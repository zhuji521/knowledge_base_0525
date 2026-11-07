# =============================================================================
# 文件说明：processor/query_processor/nodes/a_node_item_name_confirm.py
#
# 【说明】本文件是原有代码的逐行注释版：
#   · 只新增了注释，代码逻辑与原代码 100% 一致，一行代码都没有改。
#
# 这个节点在整条查询流程里排第一个（文件名以 a_ 开头，方便排序），它干三件事：
#   1) 听懂用户问的是哪个商品：从"当前问题 + 历史对话"里抽出商品名 item_names，
#      顺便把问题改写成一句完整的话（把"这个/它"换成具体商品名）；
#   2) 拿商品名去 Milvus 向量库找最像的"标准商品名"，按相似度分数分档处理：
#        高分（>= 0.8） → 基本确定，直接认定就是它；
#        中分（>= 0.6） → 拿不准，列出候选，反问用户"您说的是哪个"；
#        没分           → 直接告诉用户没找到；
#   3) 把这一轮的问答写进 MongoDB 历史会话，供下一轮做指代消解。
#
# 方法命名规范：_step_X_xxx 表示"第 X 步"，process() 里按 1 → 4 → 5 → 6 → 7 → 8
# 的顺序调用（2、3 两步不在本节点里，所以编号看着是跳号的，属正常现象）。
# =============================================================================

# 导入 json 标准库：用于把大模型返回的 JSON 文本解析成 Python 字典
import json
# 从 typing 导入 Dict、List 两个类型标注，用来标注方法的参数/返回值类型（不影响运行）
from typing import Dict, List

# 从 LangChain 导入两种消息类型：
#   SystemMessage  = 系统提示词（给大模型定人设、定规则）
#   HumanMessage   = 用户提示词（本次具体要它干的活）
from langchain_core.messages import SystemMessage, HumanMessage

# 导入 Milvus 配置对象：从中取"商品名向量集合"的集合名
from config.milvus_config import milvus_config
# 导入节点基类 NodeBase：它统一了节点入口（__call__）和异常日志，子类只需实现 process()
from processor.query_processor.base import NodeBase
# 导入商品名抽取用的两个提示词模板：
#   ITEM_NAME_EXTRACT_TEMPLATE      = 用户提示词模板（填历史对话 + 当前问题）
#   （第二行是同一个 import 语句的续行，用反斜杠 \ 换行书写，不是两条语句）
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
# 导入流程图的状态类型 QueryGraphState（本质是个 TypedDict，即"带类型提示的字典"）
from processor.query_processor.state import QueryGraphState
# 导入项目统一的日志对象，info/warn/error 都走它，方便统一收集日志
from tool.logger import logger
# 导入向量化工具：把一段文本转成"稠密向量 dense + 稀疏向量 sparse"，适配混合检索
from utils.embedding_utils import generate_embeddings
# 导入大模型客户端工厂函数，json_mode=True 表示要求模型按 JSON 格式输出
from utils.llm_utils import get_llm_client
# 导入 Milvus 相关工具：
#   get_milvus_client            = 取全局单例的 Milvus 客户端
#   create_hybrid_search_requests= 构造"稠密 + 稀疏"两条检索请求
#   hybrid_search                = 执行混合检索，并按权重融合两路结果
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
# 导入 MongoDB 历史会话工具：
#   get_recent_messages       = 查最近的对话记录
#   save_chat_message         = 新增或更新一条对话记录
#   update_message_item_names = 批量回填商品名（本文件未直接调用，属预留导入）
from utils.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names


# 定义"商品名确认"节点类，继承 NodeBase（必须实现 process 方法）
class NodeItemNameConfirm(NodeBase):
    # 节点名称：会打在日志里，也用于在流程图中标识这个节点
    name: str = "node_item_name_confirm"

    # 构造函数：创建节点对象时执行一次
    def __init__(self):
        # 预留的占位属性（当前赋值为 None，没有实际用途，可视为历史遗留）
        self._step_8_write = None

    # -------------------------------------------------------------------------
    # 节点主流程：接收上一环节传来的状态 state，处理完再返回 state
    # 父类 NodeBase.__call__ 会调用它，并统一做"开始/成功/失败"的日志和异常兜底
    # -------------------------------------------------------------------------
    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 打一条 info 日志，标记本节点开始干活
        logger.info(f"[{self.name}]节点逻辑")

        # 第 1 步：参数校验。取出会话 ID 和用户原始问题，缺一个就直接抛异常
        session_id, original_query = self._step_1_validate_param(state)

        # 查这个会话最近的对话记录（默认最近 10 条，按时间从旧到新）
        # 用途：用户说"这个""它"时，要靠历史才知道指哪个商品
        history = get_recent_messages(session_id)
        # 把历史记录放进状态，后续节点（以及其他检索节点）也能用
        state["history"] = history

        # 先把本轮"用户问题"存进 MongoDB，此时只存原始问题
        # 返回的是这条记录的 _id（字符串），第 8 步会拿它去"补全"这条记录
        # 先存的好处：即使后面检索/抽取失败，用户的提问也不会丢
        message_id = save_chat_message(session_id, "user", original_query)

        # 第 4 步：调用大模型，从"当前问题 + 历史"里抽商品名，并把问题改写完整
        extract_res = self._step_4_extract_info(original_query, history)
        # 取出抽到的商品名列表，例如 ["hak180"]；没抽到就是 []
        item_names = extract_res["item_names"]
        # 取出改写后的问题，例如把"这玩意儿咋鼓捣"改成"hak180 怎么使用"
        rewritten_query = extract_res["rewritten_query"]
        # 把改写后的问题写回状态，供下游做检索
        state["rewritten_query"] = rewritten_query
        # 把抽到的商品名写回状态
        state["item_names"] = item_names
        # 调试打印：看看大模型到底抽了什么、改写了什么（正式环境可去掉）
        print("extract_res:", extract_res)

        # 预置"对齐结果"的默认值：确认名称为空、候选项为空
        # 这样即使下面整段 if 没进去，第 7 步也不会因为缺 key 而报错
        align_result = {"confirmed_item_names": [], "options": []}
        # 预置向量检索结果为空列表（没抽到商品名时它就是空的）
        query_results = []

        # 只有真的抽出商品名，才值得去向量库检索；没抽到就直接跳过，省一次 IO
        if len(item_names) > 0:
            # 第 5 步：把每个商品名向量化，去 Milvus 做混合检索，拿回"库里最像的商品名 + 分数"
            query_results = self._step_5_vectorize_and_query(item_names)
            # 下面三行是旧的调试打印，代码里本来就被注释掉了，这里原样保留
            # print(f"milvus根据提取到的item_namede匹配结果:{query_results}")
            # for result in query_results:
            #     print(query_results)

            # 第 6 步：把检索结果按分数分档对齐 —— 高分算"确认命中"，中分算"候选让用户选"
            align_result = self._step_6_align_item_names(query_results)
            # 调试打印：说明下面要输出的是哪两份东西
            # （高分 confirmed_item_name / 中分 options）
            print(f"milvus根据提取到的item_namede匹配结果,并且经过了分数整理(结果对齐)高分:"
                  f"confirmed_item_name,中分:options")

            # 取出"高分确认"的商品名列表（可以直接认定是它）
            names_list = align_result["confirmed_item_names"]
            # 取出"中分候选"的商品名列表（需要反问用户选哪个）
            options = align_result["options"]
            # 把两份结果打印出来，方便对账排查
            print(f"{names_list},{options}")

        # 没抽到商品名：只打日志，不做检索
        else:
            # 记录 info 日志：本节点没提取到商品名，跳过向量检索
            logger.info("Node: 未提取到商品名，跳过向量检索")

        # 把 Milvus 的原始召回明细写进状态（下游要看"召回了哪些、各多少分"时用）
        state["item_name_matches"] = query_results
        # 第 7 步：根据对齐结果决定走哪条分支（确认 / 反问 / 没找到），并把答案写进 state
        state = self._step_7_check_confirmation(state, align_result, history)

        # 第 8 步：把本轮结果落库 —— 助手回答写一条，用户消息按 message_id 补全
        self._step_8_write_history(state, session_id, rewritten_query, message_id)

        # 返回处理后的状态，流程图会把它交给下一个节点
        return state

    # -------------------------------------------------------------------------
    # 第 1 步：参数校验
    # 作用：保证后面逻辑一定拿得到 session_id 和 original_query，缺了就"快速失败"
    # -------------------------------------------------------------------------
    def _step_1_validate_param(self, state):
        # 调试打印：标识当前执行到第 1 步
        print("step_1:参数检验")

        # 从状态字典里取会话 ID（用 get 取值，取不到返回 None，不会抛 KeyError）
        session_id = state.get("session_id")
        # 会话 ID 为空就抛 ValueError
        if not session_id:
            # 没有会话 ID 就查不了历史、存不了记录，直接报错中断
            raise ValueError("核心参数session_id缺失")

        # 从状态字典里取用户原始问题
        original_query = state.get("original_query")
        # 原始问题为空就抛 ValueError
        if not original_query:
            # 没有问题文本，抽取和检索都无从谈起，直接报错中断
            raise ValueError("核心参数original_query缺失")

        # 校验通过，把两个值一起返回，供调用方解包使用
        return session_id, original_query

    # -------------------------------------------------------------------------
    # 第 4 步：调用大模型抽取商品名 + 改写问题
    # 入参：original_query 用户原始问题；history 最近的历史对话
    # 返回：{"item_names": [...], "rewritten_query": "..."} 这样的字典
    # -------------------------------------------------------------------------
    def _step_4_extract_info(self, original_query, history) -> Dict:
        # 调试打印：标识当前执行到第 4 步
        print("step_4:模型提取意图主体")

        # 获取大模型客户端；json_mode=True 要求模型直接吐 JSON，便于后面 json.loads 解析
        ai_client = get_llm_client(json_mode=True)

        # 先准备一个空字符串，用来把历史对话拼成一段纯文本
        history_text = ""
        # 逐条遍历历史消息（每条是 MongoDB 里的一条记录，本质是字典）
        for msg in history:
            # 取这条消息的角色："user"（用户）或 "assistant"（助手）
            role = msg.get("role")
            # 取这条消息的正文内容
            content = msg.get("text")
            # 拼成"角色:内容 + 换行"的形式，一路追加，最终得到整段历史
            history_text += f"{role}:{content}\n"

        # 把历史文本和当前问题填进提示词模板，得到最终发给模型的"用户提示词"
        user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            # 模板里的 {history_text} 占位符 → 上面拼好的历史对话
            history_text=history_text,
            # 模板里的 {query} 占位符 → 用户当前问题
            query=original_query,
        )

        # 组装成 LangChain 的消息列表：一条系统提示词 + 一条用户提示词
        messages = [
            # 系统提示词：设定"你是一个专业的客服助手……"的人设与输出要求
            SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
            # 用户提示词：历史对话 + 当前问题 + 抽取/改写的具体要求
            HumanMessage(content=user_prompt)
        ]

        # 调用大模型，同步拿到返回结果（这一步会真正发起网络请求，耗时一般在几百毫秒到几秒）
        response = ai_client.invoke(messages)
        # 从返回对象里取出文本内容（AIMessage.content）
        response_content = response.content
        # 兼容处理：有些模型会把 JSON 包在 ```json ... ``` 代码块里返回
        if response_content.startswith("```json"):
            # 把开头的 ```json 和结尾的 ``` 都去掉，只留下纯 JSON 文本
            response_content = response_content.replace("```json", "").replace("```", "")

        # 把 JSON 文本解析成 Python 字典
        result = json.loads(response_content)
        # 兜底：模型万一没返回 item_names 字段，就补一个空列表，避免后面 KeyError
        if "item_names" not in result:
            # 补上默认空列表
            result["item_names"] = []
        # 兜底：模型万一没返回 rewritten_query 字段，就用原始问题顶上
        if "rewritten_query" not in result:
            # 用原始问题作为改写结果
            result["rewritten_query"] = original_query
        # 如果确实抽到了商品名，就做一次"去空白"清洗
        if result["item_names"]:
            # 逐个商品名，把其中的空格、换行、制表符、回车全部删掉
            # （模型偶尔会在商品名里加空格或换行，不清掉会导致向量检索匹配不上）
            result["item_names"] = [name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "") for name
                                    in result["item_names"]]

        # 下面几行是早期的调试残骸，代码里本来就是注释状态，原样保留不启用
        # ai_client.invoke(user_prompt)
        #
        # for ms in history:
        #     role=msg.get("")

        # 返回解析、兜底、清洗之后的结果字典
        return result

    # -------------------------------------------------------------------------
    # 第 5 步：向量化 + 混合检索
    # 入参：item_names 抽到的商品名列表
    # 返回：[{"extracted_name": 抽到的名字, "matches": [{"item_name": 库里的名字, "score": 分数}, ...]}, ...]
    # -------------------------------------------------------------------------
    def _step_5_vectorize_and_query(self, item_names) -> List[Dict]:
        # 调试打印：说明本步在干什么
        print("step_5:向量化并检索,检索出来对应item_name的向量库中的相似商品名称的列表")

        # 先准备一个空列表，用来装"每个商品名的召回结果"
        result: List[Dict] = []

        # 获取 Milvus 客户端（全局单例，重复调用不会重复建连接）
        milvus_client = get_milvus_client()
        # 从配置里取"商品名向量集合"的集合名（相当于数据库里的表名）
        collection_name = milvus_config.item_name_collection

        # 一次性把整批商品名转成向量，返回结构形如 {"dense": [...], "sparse": [...]}
        # 两个列表的下标与 item_names 一一对应
        embeddings = generate_embeddings(item_names)
        # 按商品名下标逐个检索（有几个商品名就查几轮）
        for i in range(len(item_names)):
            # 取第 i 个商品名对应的稠密向量（语义相似度，适合理解"意思接近"）
            dense_vector = embeddings.get("dense")[i]
            # 取第 i 个商品名对应的稀疏向量（关键词匹配，适合型号、专有名词精确命中）
            sparse_vector = embeddings.get("sparse")[i]
            # 构造两条检索请求：一条查稠密字段，一条查稀疏字段；每条各取 top5
            regs = create_hybrid_search_requests(
                # 稠密向量
                dense_vector=dense_vector,
                # 稀疏向量
                sparse_vector=sparse_vector,
                # 单路召回条数：每种向量各取前 5 条
                limit=5
            )
            # 执行混合检索：两路结果按权重融合后排序
            search_res = hybrid_search(
                # Milvus 客户端
                client=milvus_client,
                # 集合名（商品名向量表）
                collection_name=collection_name,
                # 上面构造的两条检索请求
                reqs=regs,
                # 融合权重：稠密 0.8、稀疏 0.2，即更看重语义相似
                ranker_weights=(0.8, 0.2),
                # 融合前先把两路分数归一化到 0~1，避免量级不同导致权重失真
                norm_score=True,
                # 结果里除了主键，还要带回 item_name 字段（标准商品名）
                output_fields=["item_name"],
            )

            # 准备一个空列表，装"这一个商品名"的所有命中
            matches = []
            # 检索成功且确实有结果时才处理（hybrid_search 失败会返回 None）
            if search_res and len(search_res) > 0:
                # 调试打印：本次向量检索的原始返回
                print("此处是意图识别的向量搜索结果")
                # 打印原始结果，排查分数/召回问题时看这里
                print(search_res)

                # search_res 是"按查询分组"的二维结构：[第1个查询的结果, 第2个查询的结果...]
                # 这里每轮只查了 1 个商品名，所以取 [0] 这一组
                for hit in search_res[0]:
                    # 把每条命中整理成统一的小字典，追加进 matches
                    matches.append(
                        {
                            # 命中记录里的标准商品名
                            "item_name": hit.entity.get("item_name"),
                            # 融合后的相似度分数（0~1，越大越像）
                            "score": hit.get("distance"),
                        }
                    )

                # 把"抽到的名字 + 它的全部命中"作为一条记录存进总结果
                result.append({
                    # 本轮检索用的商品名（用户侧抽出来的）
                    "extracted_name": item_names[i],
                    # 该名字在库里的所有命中
                    "matches": matches
                })

        # 返回所有商品名的召回结果
        return result

    # -------------------------------------------------------------------------
    # 第 6 步：结果对齐（按分数分档）
    # 核心规则：
    #   · 高分（>= 0.8）只有 1 个      → 直接确认
    #   · 高分（>= 0.8）有多个         → 优先选与抽出名字完全相同的那个，否则取第一个
    #   · 没有高分但有中分（>= 0.6）   → 放进 options，交给第 7 步反问用户
    # -------------------------------------------------------------------------
    def _step_6_align_item_names(self, query_results: List[Dict]) -> Dict:
        # 调试打印：说明本步在做什么
        print("step_6:对齐结果:高分结果和低分对齐结果整理")
        # 准备两个列表：confirmed = 确认命中的标准名；options = 待用户选择的候选名
        confirmed_item_names: List[str] = []
        # 候选项列表（中分结果）
        options: List[str] = []

        # 逐个处理"每个抽出商品名的召回结果"
        for res in query_results:
            # 取出用户侧抽到的名字；取不到就用空串兜底，再 strip 去掉首尾空白
            extracted_name = (res.get("extracted_name", "") or "").strip()
            # 取出该名字的命中列表；取不到就用空列表兜底
            matches = res.get("matches", []) or []
            # 一条命中都没有，这个商品名就没法判断，直接看下一个
            if not matches:
                # 跳过本轮
                continue

            # 高分集合：分数 >= 0.8 的命中，基本可以认定是同一个商品
            # （用 m.get("score",0) 取值，缺分数按 0 处理，避免 TypeError）
            high_results = [m for m in matches if m.get("score", 0) >= 0.8]
            # 中分集合：分数 >= 0.6 的命中（注意它包含高分段，是"至少 0.6"）
            mid_results = [m for m in matches if 0.8 > m.get("score", 0) >= 0.6]

            # 情况一：只有 1 个高分 → 直接确认是它
            if len(high_results) == 1:
                # 【代码原样，注意这里有疑似 Bug】
                # append() 的返回值是 None，这里再链式调用 .get(...) 会抛
                # AttributeError: 'NoneType' object has no attribute 'get'
                # 推测原意是：confirmed_item_names.append(high_results[0].get("item_name"))
                confirmed_item_names.append(high_results[0].get("item_name"))
                # 本轮处理完，继续下一个商品名
                continue

            # 情况二：有多个高分 → 需要"挑一个"
            if len(high_results) > 1:
                # 先假设没挑中
                picked = None
                # 如果用户侧抽到了名字，就尝试"精确匹配"：找库里名字与抽到的名字完全相同的那个
                if extracted_name:
                    # 遍历所有高分候选
                    for hr in high_results:
                        # 名字完全相同，就认为命中了用户真实想要的那个
                        if hr.get("item_name") == extracted_name:
                            # 记下这个候选
                            picked = hr
                            # 找到就跳出循环，不用再比
                            break
                # 精确匹配没找到（picked 仍是 None），就退而取第一个高分候选
                if not picked:
                    # 用第一个高分候选兜底
                    picked = high_results[0]

                # 【代码原样，注意这里有疑似 Bug】字段名写成了 "items_name"（多了个 s）
                # 正确应为 "item_name"，否则取到的是 None，会让确认结果变成 [None]
                confirmed_item_names.append(picked.get("item_name"))
                # 本轮处理完，继续下一个商品名
                continue

            # 情况三：没有高分，但有中分 → 说明"有点像但不确定"，交给用户选
            if len(mid_results) > 0:
                # 最多取前 5 个中分候选，逐个放进 options
                for mr in mid_results[:5]:
                    # 把候选的标准商品名加入待选列表
                    options.append(mr.get("item_name"))

        # 返回对齐结果
        return {
            # 确认命中：用 set 去重，再转回列表（避免同一个商品名被重复确认）
            "confirmed_item_names": list(set(confirmed_item_names)),
            # 候选项：保持原有顺序，不做去重
            "options": options
        }

    # -------------------------------------------------------------------------
    # 第 7 步：根据对齐结果，决定节点走向并填充 state
    # 三个分支：A 确认命中 / B 有候选需反问 / C 啥也没有
    # -------------------------------------------------------------------------
    def _step_7_check_confirmation(self, state, align_result: Dict, history):
        # 调试打印：说明本步在做什么
        print("step_7:状态state信息,根据第六步高分低分对齐结果整理")

        # 取出高分确认结果
        confirmed = align_result.get("confirmed_item_names")
        # 取出中分候选结果
        options = align_result.get("options")

        # 分支 A：有高分确认的商品名 → 直接认定，继续往下走检索流程
        if confirmed:
            # 准备一个"待更新记录 ID"的列表（当前为空，属于未完成的预留逻辑）
            ids_to_update = []
            # ... 是 Python 的省略号占位符，表示"此处逻辑待补充"，运行时什么都不做
            ...
            # 把确认后的标准商品名写回状态（覆盖掉第 4 步抽出来的原始名字）
            state["item_names"] = confirmed
            # 把答案置空：表示本节点不直接回复用户，交给下游节点继续检索作答
            state["answer"] = ""
            return state

        # 分支 B：没有确认结果、但有候选 → 反问用户"你说的是哪个"
        elif options:
            # 候选项不唯一，先清空商品名，避免下游拿错名字去检索
            state["item_names"] = []
            # 用顿号把候选名拼成一句话，例如 "hak180、HAK-180"
            options_str = "、".join(options)
            # 给用户的提示语：让他明确型号
            state["answer"] = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"
            return state

        # 分支 C：既没确认、也没候选 → 直接告诉用户没找到
        else:
            # 给用户的兜底提示
            state["answer"] = "抱歉,未找到相关产品"
            # 商品名保持空列表
            state["item_names"] = []
        # 返回更新后的状态
        return state

    # -------------------------------------------------------------------------
    # 第 8 步：把本轮结果写进历史会话
    # 两件事：
    #   1) 如果 state 里有 answer（分支 B/C 的反问/兜底），存一条 assistant 消息；
    #   2) 用第 4 步拿到的 message_id，把用户那条消息补全（改写问题 + 商品名）。
    # -------------------------------------------------------------------------
    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        # 调试打印：说明本步在做什么
        print("step_8:写入历史会话,更新")

        # 若会话状态中有助手答案（分支B/C），写入助手消息到历史
        if state.get("answer"):
            # 调用工具写入一条"助手"消息
            save_chat_message(
                session_id=session_id,  # 会话ID，关联所属会话
                role="assistant",  # 消息角色：助手
                text=state["answer"],  # 消息内容：向用户确认的提示语/无结果提示语
                rewritten_query="",  # 助手消息无需改写查询，设为空
                item_names=state.get("item_names", [])  # 关联的商品名列表（分支B/C均为空）
            )

        # 强制更新本次用户原始问题的关联信息（核心：补充改写查询、商品名）
        save_chat_message(
            session_id=session_id,  # 会话ID，关联所属会话
            role="user",  # 消息角色：用户
            text=state["original_query"],  # 消息内容：用户原始查询
            rewritten_query=rewritten_query,  # 补充step3改写后的完整问题
            item_names=state.get("item_names", []),  # 补充关联的商品名列表
            message_id=message_id  # 消息ID，指定更新已存在的用户消息（而非新增）
        )

        # 返回最终会话状态，供下游节点使用
        return state


# 直接运行本文件时（python a_node_item_name_confirm.py）才执行的单元测试代码
if __name__ == "__main__":
    # 初始化图状态
    init_state = {
        # 模拟用户一句很口语化的问题
        "original_query": "hak180这玩儿意咋鼓捣呀？",
        # 模拟一个会话 ID（这里写成了 int，实际业务里是字符串）
        "session_id": 1234567
    }

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()
    # 执行节点的单元测试（会真正调用大模型、Milvus、MongoDB）
    result = node_item_name_confirm(init_state)
    # 将返回的图状态进行json序列化（下面两行是注释状态，原样保留）
    # json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logger.info(result)

# =============================================================================
# 附：阅读时值得留意的几处（仅为注释说明，代码未做任何修改）
#
#   1. 第 6 步 high_results 只有 1 个时：
#        confirmed_item_names.append(high_results[0]).get("item_name")
#      append() 返回 None，再 .get() 会抛 AttributeError。
#   2. 第 6 步多高分时：picked.get("items_name") 字段名多了一个 s，
#      取到的是 None，最终可能得到 [None]。
#   3. 第 4 步代码块清理只处理了以 ```json 开头的情况，
#      若模型返回 ```（不带 json）或前后有空格，会解析失败。
#   4. 第 7 步 ids_to_update = [] 与 ... 是未完成的占位逻辑。
#   5. 第 5 步若某商品名检索无结果，该名字不会出现在 query_results 里，
#      第 6 步也就不会为它产生任何结论（自然落入"未找到"）。
#   6. 第 8 步注释里提到"step3 改写"，实际改写发生在第 4 步，注释为历史遗留。
# =============================================================================
