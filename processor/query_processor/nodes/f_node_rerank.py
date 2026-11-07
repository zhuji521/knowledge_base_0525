import json
from typing import List, Dict, Any

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.reranker_http_utils import rerank_documents


class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        merged_multi_docs: List[Dict[str, Any]] = self._step_1_merge_multi_sourse_docs(state)
        print(merged_multi_docs)

        reranked_docs: List[Dict[str, Any]] = self._step_2_rerank_merged_docs(state, merged_multi_docs)

        cutoff_docs = self._step_3_cliff_cutoff(reranked_docs)

        print("步骤4:返回结果封装")

        state["reranked_docs"] = cutoff_docs
        print(cutoff_docs)

        return state

    def _step_1_merge_multi_sourse_docs(self, state):
        print("步骤1:将web_doc和rrf_chunks节点的数据合并")

        final_docs = []
        rrf_chunks = state.get("rrf_chunks") or []
        web_docs = state.get("web_search_docs") or []

        for rrf_doc in rrf_chunks:
            final_docs.append({
                "content": rrf_doc.get("content"),
                "title": rrf_doc.get("title"),
                "chunks_id": rrf_doc.get("chunk_id"),
                "url": None,
                "source": "local",
            })

        for web_doc in web_docs:
            final_docs.append({
                "content": web_doc.get("content") or web_doc.get("snippet"),
                "title": web_doc.get("title"),
                "chunks_id": None,
                "url": web_doc.get("url"),
                "source": "web",
            })

        return final_docs

    def _step_2_rerank_merged_docs(self, state, merged_multi_docs):
        print("步骤1:调用排序模型对合并后结果进行排序")
        if not merged_multi_docs:
            return []
        rewritten_query = state.get("rewritten_query")
        contents = [doc.get("content") or "" for doc in merged_multi_docs]
        rerank_scores = rerank_documents(rewritten_query, contents)
        scored_docs = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]
        return sorted(scored_docs, key=lambda x: x["score"], reverse=True)

    def _step_3_cliff_cutoff(self, reranked_docs):
        print("步骤3:断崖测试(分数差距,分数优化)去掉过低的匹配结果")
        upper_bound = min(10, len(reranked_docs))
        lower_bound = min(3, len(reranked_docs))

        cutoff_pos = upper_bound
        for index in range(lower_bound - 1, upper_bound - 1):
            current_score = reranked_docs[index].get("score")
            next_score = reranked_docs[index + 1].get("score")

            abs_gap = current_score - next_score

            rel_gap=abs_gap/(abs(current_score)+1e-6)

            if abs_gap >= 0.5 or rel_gap >=0.25:
                cutoff_pos=index +1
                break

        return reranked_docs[:cutoff_pos]

if __name__ == "__main__":
    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {
                "chunk_id": "local_1",
                "title": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"
            },
            {
                "chunk_id": "local_2",
                "title": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
            },
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/repair",
                "title": "短路查修指南",
                "content": "主板通电前先打各主供电电感对地阻值，阻值偏低就是短路。"
            },
            {
                "url": "https://example.com/news",
                "title": "科技新闻",
                "content": "苹果发布新款手机，A系列芯片性能提升20%。"
            },
        ],
    }

    node_rerank = NodeRerank()
    result = node_rerank(mock_state)
    logger.info(json.dumps(result, indent=4))
