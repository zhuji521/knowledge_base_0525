import json

from torch.nn.functional import embedding

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")
        embedding_chunks=state.get("embedding_chunks")
        hyde_embedding_chunks=state.get("hyde_embedding_chunks")

        embedding_chunks_list=[doc.get("entity") for doc in embedding_chunks]
        hyde_embedding_chunks_list=[doc.get("entity") for doc in hyde_embedding_chunks]

        rrf_inputs=[
            (embedding_chunks_list,1.0),
            (hyde_embedding_chunks_list,1.0),
        ]

        rrf_merge_result=self._rrf_merge(rrf_inputs)

        state["rrf_chunks"]=None
        return state

if __name__ == '__main__':

    # 模拟两路检索结果
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#3"}},
        ]
    }

    node_rrf = NodeRrf()
    result = node_rrf(mock_state)
    logger.info(json.dumps(result, indent=4))


















        return state