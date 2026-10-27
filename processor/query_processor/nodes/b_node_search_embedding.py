# 导入 json 模块，用于把检索结果转成可读的 JSON 字符串并打印
import json

# 从配置里拿到 Milvus 相关配置（例如集合名）
from config.milvus_config import milvus_config
# 导入查询流水线节点的基类，本节点继承它
from processor.query_processor.base import NodeBase
# 导入查询图状态类型，约定 process 的输入/输出结构
from processor.query_processor.state import QueryGraphState
# 导入日志工具，方便打印节点执行信息
from tool.logger import logger
# 导入生成向量的工具：把文本变成稠密/稀疏向量
from utils.embedding_utils import generate_embeddings
# 导入 Milvus 相关工具：建连接、构造混合检索请求、执行检索、转义字符串
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search, escape_milvus_string


# 定义「向量检索」节点类，继承通用节点基类
class NodeSearchEmbedding(NodeBase):
    """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

    # 覆盖基类的 name 属性，标识节点名称;
    name: str = "node_search_embedding"

    # 节点主处理函数：接收上游状态，做检索后把结果写回状态
    def process(self, state: QueryGraphState) -> QueryGraphState:

        # TODO
        # 打一条日志，标明当前正在执行本节点
        logger.info(f"【{self.name}】节点逻辑")
        # 从状态里取出已确认的主体名列表；没有则用空列表，避免后面报错
        item_names=state.get("item_names") or []
        # 从状态里取出改写后的用户问题，作为本次检索的查询文本
        query=state.get("rewritten_query")

        # 对查询文本生成向量；入参是列表，所以包一层 [query]
        query_embeddings=generate_embeddings([query])
        # 取出第 1 条查询对应的稠密向量（语义相似度用）
        dense_vector=query_embeddings["dense"][0]
        # 取出第 1 条查询对应的稀疏向量（关键词匹配用）
        sparse_vector=query_embeddings["sparse"][0]

        # 获取 Milvus 客户端，后续用它发检索请求
        milvus_client =get_milvus_client()
        # 从配置读取「文档块」集合名，指定去哪个集合里搜
        chunks_collection=milvus_config.chunks_collection

        # Milvus 过滤表达式需要双引号字符串字面量，不能直接塞 Python list / None
        # 把每个主体名转义并包上双引号，拼成 Milvus 可识别的字符串列表
        safe_names = [f'"{escape_milvus_string(name)}"' for name in item_names]
        # 有主体名时：过滤条件为 item_name in ["名1", "名2", ...]；没有则不过滤（expr=None）
        expr = f'item_name in [{", ".join(safe_names)}]' if safe_names else None

        # 用稠密向量 + 稀疏向量 + 过滤条件，构造混合检索请求
        reqs=create_hybrid_search_requests(
            dense_vector=dense_vector,   # 语义向量检索请求
            sparse_vector=sparse_vector, # 稀疏向量检索请求
            expr=expr                    # 主体名过滤条件（可为空）
        )

        # 调用混合检索：稠密/稀疏两条路一起搜，再按权重融合排序
        response = hybrid_search(
            client=milvus_client,                 # Milvus 客户端
            collection_name=chunks_collection,    # 要检索的集合名
            reqs=reqs,                            # 上面构造好的混合检索请求
            ranker_weights=(0.8,0.2),              # 融合权重：稠密 0.8，稀疏 0.2
            output_fields=["chunk_id","content","item_name"]  # 只要这些字段返回
        )

        # Hit.to_dict() 得到: {chunk_id, distance, entity:{chunk_id, content, item_name}}
        # 把检索命中列表转成字典列表；若没有结果则返回空列表
        chunks = [hit.to_dict() for hit in response[0]] if response else []
        # 把检索到的文档块写回状态，供下游节点使用
        return {"embedding_chunks": chunks}

# 仅当本文件被直接运行时才进入下面的测试代码（被 import 时不会执行）
if __name__ =='__main__':
    # 实例化本节点，方便本地单独调试
    node_search_embedding=NodeSearchEmbedding()
    # 构造一份最小测试状态：主体名 + 改写后的问题
    init_state={
        "item_names":["H3CLA2608室内无线网关"],
        "rewritten_query":"帮我查一下H3CLA2608室内无线网关是什么"
    }
    # 调用 process，跑一遍检索流程
    process=node_search_embedding.process(init_state)
    # 把 embedding_chunks 结果以中文可读的 JSON 格式打印出来
    print(json.dumps(process["embedding_chunks"], ensure_ascii=False, indent=4))
