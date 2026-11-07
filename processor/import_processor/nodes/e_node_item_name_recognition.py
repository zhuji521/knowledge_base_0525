# ============================================================================
# 主体识别节点  e_node_item_name_recognition.py
# ----------------------------------------------------------------------------
# 这个节点的作用：
#   用大模型从「文件标题 + 正文切片」中识别出商品名称（即主体），
#   然后把名称回填到 state 和每个切片里，
#   最后把商品名称向量化并写入 Milvus 向量库，供后续检索使用。
#
# 一句话流程：
#   取输入 -> 拼上下文 -> 调模型 -> 回填 -> 向量化 -> 入库
# ============================================================================

# ---------------------------- 标准库 ----------------------------
import json                                  # 读取 json 文件（只在文件末尾的本地测试代码里用）
from pathlib import Path                     # 拼接备份 json 路径
from typing import List, Dict, Tuple         # 类型注解：列表 / 字典 / 元组

# ---------------------------- 第三方库 ----------------------------
from langchain_core.messages import SystemMessage, HumanMessage  # 大模型消息：System 定角色，Human 传内容
from pymilvus import DataType                                    # Milvus 的字段数据类型枚举

# ---------------------------- 项目内部模块 ----------------------------
from config.lm_config import lm_config                             # 大模型配置：模型名、api_key、base_url、temperature
from config.milvus_config import milvus_config                     # Milvus 配置：服务地址、集合（表）名
from processor.import_processor.base import BaseNode               # 导入流程节点的基类
from processor.import_processor.exceptions import StateFieldError  # 状态字段校验失败时抛的异常
from processor.import_processor.state import ImportGraphState      # 流程图状态对象（各节点共享的数据）
from utils.embedding_utils import generate_embeddings              # 向量化工具：一次返回稠密向量 + 稀疏向量
from utils.milvus_utils import get_milvus_client, escape_milvus_string  # 获取 Milvus 客户端 / 转义字符串中的特殊符号
from utils.llm_utils import get_llm_client                         # 带缓存的 LLM 客户端工厂


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取

    节点输入：
        state["file_title"] —— 文件标题，例如 "H3C LA2608室内无线网关 用户手册-6W100-整本手册"
        state["chunks"]     —— 切片列表，每个切片含 title、content 等字段

    节点输出：
        state["item_name"]              —— 识别出的商品名称
        每个 chunk 里的 item_name 字段   —— 同样写入商品名称
        Milvus 中新增（或覆盖）一条商品名称记录

    process() 的 6 个步骤：
        1. 取出并校验输入
        2. 把前几个切片拼成上下文（控制长度，避免超模型上限）
        3. 调用大模型识别商品名称
        4. 把商品名称回填到 state 和 chunks
        5. 把商品名称转成稠密向量和稀疏向量
        6. 写入 Milvus 向量库
    """

    # 节点名：在流程图 / 配置里引用这个节点时用的标识
    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState):
        # 主流程：按顺序执行 6 个步骤，每步单独一个 _step_x 方法，方便阅读和排查问题

        # 第 1 步：校验并取出输入（文件标题、切片列表）
        file_title, chunks = self._step_1_get_inputs(state)

        # 第 2 步：把靠前的若干个切片拼成一段上下文，并限制总字数
        context = self._step_2_build_context(file_title, chunks)
        # print(f"上下文长度:{len(context)}字符")

        # 第 3 步：调用大模型，从「标题 + 上下文」中识别商品名称
        item_name = self._step_3_call_llm(file_title, context)
        print(f"item_name:{item_name}")

        # 第 4 步：把识别结果回填到 state 和每个切片里
        self._step_4_update_chunks(state, item_name, chunks)

        # 第 4.5 步：备份「识别 item_name 后」的切片到本地
        # 命名：{file_title}_new_chunks.json，例如 B530_new → B530_new_new_chunks.json
        self._step_5_backup_chunks(state, chunks)

        # 第 5 步：把商品名称向量化，得到稠密向量和稀疏向量
        dense_vector, sparse_vector = self._step_5_generate_vectors(item_name)
        print(f"dense_vector:{dense_vector}")
        print(f"sparse_vector:{sparse_vector}")

        # 第 6 步：把商品名称和两个向量写入 Milvus
        self._step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        # 返回更新后的 state，交给下一个节点继续使用
        return state

    def _step_1_get_inputs(self, state) -> Tuple[str, List[Dict]]:
        """
        第 1 步：取出并校验输入参数。

        返回：(file_title, chunks)
        校验不通过时抛 StateFieldError，避免后面用错数据、报更难懂的错。
        """
        print("node_item_name_recognition:步骤1:参数处理")
        # 1) 从 state 中取文件标题
        file_title = state["file_title"]

        # 2) 标题不能为空，否则没有识别依据
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        # 3) 用 get 取切片，key 不存在时返回 None，避免直接报 KeyError
        chunks = state.get("chunks")
        # 4) 切片不能为空（None 或 [] 都算空）
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        # 5) 切片必须是 list 类型，否则后面的遍历会出错
        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        # 6) 校验通过，返回两个输入
        return file_title, chunks

    def _step_2_build_context(self, file_title, chunks):
        """
        第 2 步：拼接上下文。

        目的：大模型看标题之外，再看一点正文，识别会更准。
        做法：按顺序取切片，逐个拼成「第N个切片:标题\n内容」，
              同时累计字数，超过上限就停止，最后再整体截断一次。
        """
        print("node_item_name_recognition:步骤2:上下文拼接")
        # 1) 最多取前 k 个切片（k 来自节点配置）
        k = self.config.item_name_chunk_k

        # 2) 上下文字符数上限（来自节点配置）
        chunk_size = self.config.item_name_chunk_size

        parts: List[str] = []    # 存放每个切片拼好的文本
        total_chars = 0          # 累计已拼接的字数

        # 3) 遍历前 k 个切片：index 从 1 开始，只用于显示"第N个切片"
        for index, chunk in enumerate(chunks[:k], start=1):
            # 3.1 取切片标题和正文，strip 去掉首尾空白
            chunk_title = chunk.get("title", "").strip()
            chunk_content = chunk.get("content", "").strip()

            # 3.2 拼成一段："第N个切片:标题" + 换行 + 正文
            piece = f"第{index}个切片:{chunk_title}\n{chunk_content}\n"
            parts.append(piece)
            total_chars += len(piece)

            # 3.3 字数够了就提前结束，别拼太长
            if total_chars > chunk_size:
                break
        # 4) 用空行把各段连接起来，并去掉首尾空白
        context = "\n\n".join(parts).strip()
        # 5) 再硬截断一次，确保绝不超过上限
        final_context = context[:chunk_size]
        return final_context

    def _step_3_call_llm(self, file_title, context):
        """
        第 3 步：调用大模型识别商品名称。

        规则：
          - 没有上下文时，直接把文件标题当商品名称；
          - 有上下文则让模型结合「标题 + 正文」输出商品名称；
          - 模型输出会清理空白字符；如果模型什么都没返回，同样退回用文件标题。
        """
        print("node_item_name_recognition:步骤3:模型识别")

        # 1) 没有上下文就没必要调模型，直接用文件标题兜底
        if not context:
            return file_title

        # 2) 创建 模型客户端（参数全部来自大模型配置）
        llm = get_llm_client(lm_config.item_model)

        # 3) 拼提示词：告诉模型要识别什么、输出格式是什么
        prompt = f"""请从以下信息中识别出商品名称与型号：
                文件名：{file_title}
                
                正文切片（用于辅助识别）：
                {context}
                
                要求：
                1. 返回内容为字符串形式，最好型号和名称的完整商品名称。比如：5000W大功率电磁炉；
                2. 返回结果应该只包含商品名称，不要添加任何解释或其他内容；
                3. 如果无法识别商品名称,请返回空字符串
        """
        # 4) 组成消息列表：System 设定模型角色，Human 放具体任务
        message = [
            SystemMessage("你是一个专业的商品名称识别模型,请根据提供的信息,识别商品名称,名称最好不要超过20个字"),
            HumanMessage(content=prompt)
        ]
        # 5) 调用模型，拿到回复
        response = llm.invoke(message)

        # 6) 取出回复文本（getattr 防止没有 content 属性），并去掉首尾空白
        item_name = getattr(response, "content", "").strip()
        # 7) 去掉名称里的空格、换行、制表符等，保证入库数据干净
        item_name = (item_name.replace(" ", "")
                     .replace("\n", "")
                     .replace("\t", "")
                     .replace("\r", ""))
        # 8) 模型没识别出来（空字符串）时，退回用文件标题
        if not item_name:
            item_name = file_title

        # 9) 返回最终商品名称
        return item_name

    def _step_4_update_chunks(self, state, item_name, chunks):
        """
        第 4 步：回填数据。

        把识别出的商品名称写到两个地方：
          - state["item_name"]，供后续节点使用；
          - 每个 chunk 的 item_name 字段，供后续切片入库使用。
        """
        print("node_item_name_recognition:步骤4:回填数据")
        # 1) 写入整个流程的状态
        state["item_name"] = item_name
        # 2) 逐个切片写入
        for chunk in chunks:
            chunk["item_name"] = item_name
        # 3) 同步回 state，保证下游读到带 item_name 的切片
        state["chunks"] = chunks
        return state

    def _step_5_backup_chunks(self, state, chunks):
        """
        备份识别 item_name 后的文档切片。

        输出文件（与课程图示一致）：
          {file_title}_new_chunks.json
          例如 file_title=B530_new → B530_new_new_chunks.json
          例如 file_title=H3C LA2608室内无线网关 用户手册-6W100-整本手册_new → H3C LA2608室内无线网关 用户手册-6W100-整本手册_new_new_chunks.json
        保存目录优先取 md_path 所在目录，其次 file_dir。
        """
        print("node_item_name_recognition:步骤4.5:备份识别后的chunks")
        backup_dir = self._resolve_backup_dir(state)
        file_title = state.get("file_title") or "unknown"
        path = backup_dir / f"{file_title}_new_chunks.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        self.logger.info(f"识别 item_name 后的文档已备份: {path}")
        return str(path)

    @staticmethod
    def _resolve_backup_dir(state) -> Path:
        md_path = state.get("md_path")
        if md_path:
            return Path(md_path).parent
        file_dir = state.get("file_dir")
        if file_dir:
            return Path(file_dir)
        raise StateFieldError(
            field_name="md_path/file_dir",
            message="无法确定备份目录，请提供 md_path 或 file_dir",
            expected_type=str,
        )

    def _step_5_generate_vectors(self, item_name):
        """
        第 5 步：向量化。

        把商品名称文本转成两个向量：
          - dense（稠密向量）：擅长语义相似检索；
          - sparse（稀疏向量）：擅长关键词精确匹配。
        """
        print("node_item_name_recognition:步骤5:主体名称向量化,返回稠密和稀疏向量")
        # 1) 传入列表批量向量化，返回 {"dense": [...], "sparse": [...]}
        embedding = generate_embeddings([item_name])
        # 2) 只处理了一个名称，所以取第 0 个
        dense = embedding["dense"][0]
        sparse = embedding["sparse"][0]

        # 3) 返回两个向量
        return dense, sparse

    def _step_6_save_to_milvus(self, state, file_title, item_name, dense_vector, sparse_vector):
        """
        第 6 步：写入 Milvus 向量库。

        要点：
          - 先检查配置；
          - 定义表结构（字段 + 类型）；
          - 给向量字段建索引；
          - 集合不存在就创建；
          - 先按商品名称删掉旧记录，再插入新记录（同名覆盖，避免重复）；
          - 最后把集合加载到内存，才能被检索。
        """
        print("node_item_name_recognition:步骤6:存入milvus向量库")

        # 1) 读取 Milvus 配置：服务地址、集合（表）名
        milvus_url = milvus_config.milvus_url
        collection_name = milvus_config.item_name_collection

        # 2) 配置缺失直接报错，避免连到错误的库
        if not milvus_url or not collection_name:
            raise Exception("Milvus配置错误")

        # 3) 获取 Milvus 客户端
        client = get_milvus_client()

        # 4) 创建表结构：auto_id=True 表示主键由 Milvus 自动生成；开启动态字段
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        # 4.1 主键字段 pk：自增的 64 位整数
        schema.add_field(
            field_name="pk",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )
        # 4.2 文件标题字段，最长 65535 个字符
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=65535
        )

        # 4.3 商品名称字段，最长 65535 个字符
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        # 4.4 稠密向量字段，维度固定 1024（要和向量模型输出维度一致）
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024
        )

        # 4.5 稀疏向量字段，维度不固定
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        # 5) 准备索引参数（每个向量字段一个索引）
        index_params = client.prepare_index_params()
        # 5.1 稠密向量索引：IVF_FLAT 聚类索引，COSINE 余弦相似度，nlist=128 个聚类中心
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128}
        )

        # 5.2 稀疏向量索引：倒排索引，IP 内积度量
        index_params.add_index(
            field_name = "sparse_vector",  # 字段名
            index_name = "sparse_vector_index",  # 索引名
            index_type = "SPARSE_INVERTED_INDEX",  # 索引类型
            metric_type = "IP",  # 相似度计算方式（内积）
            params = {
                "inverted_index_algo": "DAAT_MAXSCORE",
                # 高效的稀疏检索算法

                "normalize": True,
                # ↑ L2 归一化，让内积 (IP) 等价于余弦相似度

                "quantization": "none"
                # ↑ 关闭量化，保持原始精度：模型生成的向量已经压缩的一半的精度了（BGE_FP16=1），这里就不再压缩了
                # "quantization": "none" → 存储原始向量，不压缩
                # "quantization": "sq8" → 存储压缩后的向量（8-bit 量化
        })

        # collection_name = milvus_config.item_name_collection
        #
        # 6) 集合（表）不存在时才创建，已存在则直接复用
        if not client.has_collection(collection_name):
            client.create_collection(collection_name=collection_name,schema= schema, index_params=index_params)

        # 7) 转义名称里的特殊字符，避免过滤表达式语法出错
        safe_item_name = escape_milvus_string(item_name)
        # 8) 过滤条件：商品名称相同
        filter_expr = f'item_name=="{safe_item_name}"'
        # 9) 先删除同名旧记录，实现"同名覆盖"，避免重复数据
        client.delete(collection_name=collection_name, filter=filter_expr)

        # 10) 组装要插入的一条数据
        data = {
            "file_title": file_title,      # 文件标题
            "item_name": item_name,        # 商品名称
            "dense_vector": dense_vector,  # 稠密向量
            "sparse_vector": sparse_vector # 稀疏向量
        }
        # 11) 插入数据（传列表，表示批量插入）
        client.insert(collection_name, [data])
        # 12) 把集合加载进内存，之后才能被检索
        client.load_collection(collection_name)

        # 13) 再次确保 state 里有商品名称，并返回
        state["item_name"]=item_name
        print(f"state:{state}")
        return state

# ============================================================================
# 本地测试入口（直接运行本文件时才会执行，被别的代码 import 时不会执行）
# ============================================================================
if __name__ == "__main__":
    # 1) 创建节点实例
    node = NodeItemNameRecognition()

    # 2) 准备一份切片 json 文件路径（d 节点产出：{file_title}_chunks.json）
    path = r"D:\output\H3C LA2608室内无线网关 用户手册-6W100-整本手册\H3C LA2608室内无线网关 用户手册-6W100-整本手册_new_chunks.json"

    # 3) 读取 json 文件内容
    with open(path, "r", encoding="utf-8") as f:
        chunks_json_data = f.read()
    # 4) 组装初始 state：file_title 需与 d 节点一致（H3C LA2608室内无线网关 用户手册-6W100-整本手册_new），才会写出 H3C LA2608室内无线网关 用户手册-6W100-整本手册_new_new_chunks.json
    init_state = {
        "file_title": "H3C LA2608室内无线网关 用户手册-6W100-整本手册_new",
        "md_path": r"D:\output\H3C LA2608室内无线网关 用户手册-6W100-整本手册\H3C LA2608室内无线网关 用户手册-6W100-整本手册_new.md",
        "file_dir": r"D:\output\H3C LA2608室内无线网关 用户手册-6W100-整本手册",
        "chunks": json.loads(chunks_json_data),
    }
    print(f"init_state:{init_state}")
    # 5) 执行节点主流程（会备份到 D:\output\H3C LA2608室内无线网关 用户手册-6W100-整本手册\H3C LA2608室内无线网关 用户手册-6W100-整本手册_new_new_chunks.json）
    process = node.process(init_state)
    print(f"process:{process}")
    # 6) 打印每个切片的标题和识别出的商品名称
    for chunk in process["chunks"]:
        print(f'{chunk.get("title")}章节:{chunk.get("item_name")}')
    # 7) 打印完整的返回结果
    print(process)
