# 导入 json 模块，用于把结果转成 JSON 字符串打印
import json

# 从配置里拿 Milvus 相关配置（比如集合名）
from config.milvus_config import milvus_config
# 查询流水线节点的基类
from processor.query_processor.base import NodeBase
# 普通向量检索节点（本文件未直接使用，保留原导入）
from processor.query_processor.nodes.b_node_search_embedding import NodeSearchEmbedding
# HyDE 用的提示词模板
from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
# 查询工作流的状态类型
from processor.query_processor.state import QueryGraphState
# 日志工具
from tool.logger import logger
# 生成稠密/稀疏向量的工具函数
from utils.embedding_utils import generate_embeddings
# 获取大模型客户端的工具函数
from utils.llm_utils import get_llm_client
# Milvus 客户端、混合检索请求构造、混合检索执行
from utils.milvus_utils import (
    get_milvus_client,
    create_hybrid_search_requests,
    hybrid_search,
    escape_milvus_string,
)

# HyDE 向量检索节点：继承节点基类
class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    # 节点主入口：接收状态，处理后返回更新后的状态
    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        # 打一条日志，标明当前执行的是哪个节点
        logger.info(f"【{self.name}】节点逻辑")

        # 从状态里取出物品名称列表（用于过滤检索范围）
        item_names = state.get("item_names")
        # 从状态里取出改写后的用户问题
        rewritten_query = state.get("rewritten_query")

        # 步骤1：让大模型根据问题生成“假设性答案文档”
        hyde_doc = self._step_1_create_embedding_hyde_doc(rewritten_query)
        # 打印假设性答案，方便调试查看
        print(f"假设性回答:{hyde_doc}")

        # 步骤2：用「原问题 + 假设性答案」去做向量混合检索
        response = self._step_2_search_embedding_hyde(
            rewritten_query=rewritten_query,  # 改写后的问题
            hyde_doc=hyde_doc,                # 假设性答案文档
            item_names=item_names             # 物品名称过滤条件
        )

        # json_response = json.dumps(response, ensure_ascii=False, indent=4)
        # 打印检索结果的第一条，方便调试
        print(f"hyde hits: {len(response) if response else 0}")
        if response:
            print(response[0])

        # return state
        # 把 HyDE 检索到的 chunk 列表，以及假设性文档，写回状态
        return {"hyde_embedding_chunks": response if response else [], "hyde_doc": hyde_doc}

    # 步骤1：调用大模型，生成假设性文档
    def _step_1_create_embedding_hyde_doc(self, rewritten_query: str) -> str:
        # 打印当前步骤说明
        print("步骤1:基于大模型生成假设性文档")

        # 获取大模型客户端
        ai_client = get_llm_client()

        # 把改写后的问题填进 HyDE 提示词模板
        hyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)

        # 调用大模型，取出返回内容作为假设性文档
        hyde_doc = ai_client.invoke(hyde_prompt).content

        # 返回生成好的假设性文档
        return hyde_doc

    # 步骤2：对「问题 + 假设性文档」做向量混合检索
    def _step_2_search_embedding_hyde(self, rewritten_query: str, hyde_doc: str, item_names: list):
        # 打印当前步骤说明
        print("步骤2:根据假设性文档向量搜索")

        # 把改写问题和假设性文档拼成一段文本，作为待编码内容
        embedding_context = rewritten_query + " "+hyde_doc
        # 对这段文本生成稠密向量和稀疏向量
        embeddings =generate_embeddings([embedding_context])
        # 取出第一个样本的稠密向量
        dense_vector=embeddings.get("dense")[0]
        # 取出第一个样本的稀疏向量
        sparse_vector=embeddings.get("sparse")[0]

        # 获取已连接好的 Milvus 客户端
        milvus_client=get_milvus_client()
        # 从配置读取要检索的 chunks 集合名
        collection_name=milvus_config.chunks_collection

        # 构造混合检索请求（稠密 + 稀疏），并用 item_name 过滤
        safe_names = [f'"{escape_milvus_string(name)}"' for name in (item_names or [])]
        expr = f'item_name in [{", ".join(safe_names)}]' if safe_names else None
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=expr,
        )
        # 执行混合检索，取 Top5，并指定要返回的字段
        response=hybrid_search(
            client=milvus_client,                   # Milvus 客户端
            collection_name=collection_name,       # 集合名
            reqs=reqs,                             # 混合检索请求
            limit=5,                               # 最多返回 5 条
            output_fields=["chunk_id","content","item_name"]  # 需要的字段
        )

        # 有结果就把第一条查询的命中转成字典列表；否则返回空列表
        return [hit.to_dict() for hit in response[0]] if response else []

# 直接运行本文件时，做一次本地自测
if __name__ == '__main__':
    # 创建本节点实例
    node_search_embedding = NodeSearchEmbeddingHyde()
    # 构造测试用的初始状态
    init_state = {
        "item_names": ["H3CLA2608室内无线网关"],  # 测试用物品名
        "rewritten_query": "帮我查一下H3CLA2608室内无线网关是什么"  # 测试用问题
    }
    # 调用 process 跑一遍节点逻辑
    process = node_search_embedding.process(init_state)
    # 把最终结果以 JSON 格式漂亮打印出来
    print(json.dumps(process, ensure_ascii=False, indent=4))
