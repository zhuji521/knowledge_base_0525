import json

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search, escape_milvus_string


class NodeSearchEmbedding(NodeBase):
    """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

    # 覆盖基类的 name 属性，标识节点名称;
    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        # TODO
        logger.info(f"【{self.name}】节点逻辑")
        item_names=state.get("item_names") or []
        query=state.get("rewritten_query")

        query_embeddings=generate_embeddings([query])
        dense_vector=query_embeddings["dense"][0]
        sparse_vector=query_embeddings["sparse"][0]

        milvus_client =get_milvus_client()
        chunks_collection=milvus_config.chunks_collection

        # Milvus 过滤表达式需要双引号字符串字面量，不能直接塞 Python list / None
        safe_names = [f'"{escape_milvus_string(name)}"' for name in item_names]
        expr = f'item_name in [{", ".join(safe_names)}]' if safe_names else None

        reqs=create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=expr
        )

        response = hybrid_search(
            client=milvus_client,
            collection_name=chunks_collection,
            reqs=reqs,
            ranker_weights=(0.8,0.2),
            output_fields=["chunk_id","content","item_name"]
        )

        # Hit.to_dict() 得到: {chunk_id, distance, entity:{chunk_id, content, item_name}}
        chunks = [hit.to_dict() for hit in response[0]] if response else []
        return {"embedding_chunks": chunks}

if __name__ =='__main__':
    node_search_embedding=NodeSearchEmbedding()
    init_state={
        "item_names":["H3CLA2608室内无线网关"],
        "rewritten_query":"帮我查一下H3CLA2608室内无线网关是什么"
    }
    process=node_search_embedding.process(init_state)
    print(json.dumps(process["embedding_chunks"], ensure_ascii=False, indent=4))
