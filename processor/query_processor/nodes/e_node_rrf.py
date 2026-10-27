# 导入 json：用于把结果格式化成易读的 JSON 字符串（测试时打印用）
import json
# 从 typing 导入类型注解：List 表示列表，Tuple 表示元组
from typing import List, Tuple

# 从 torch 导入 embedding（当前文件未实际使用，可能是遗留导入）
from torch.nn.functional import embedding

# 导入查询流水线的节点基类，本节点继承它
from processor.query_processor.base import NodeBase
# 导入查询图的状态类型，用来约定 state 的数据结构
from processor.query_processor.state import QueryGraphState
# 导入日志工具，用于打印节点运行信息
from tool.logger import logger


# 定义 RRF（倒数排名融合）节点类，继承 NodeBase
class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

    # 节点主入口：接收上游状态，做融合后返回更新后的状态
    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        # 打印日志，标明当前正在执行本节点
        logger.info(f"【{self.name}】节点逻辑")
        # 从状态中取出「普通向量检索」得到的 chunk 列表
        embedding_chunks=state.get("embedding_chunks")
        # 从状态中取出「HyDE 假设性文档向量检索」得到的 chunk 列表
        hyde_embedding_chunks=state.get("hyde_embedding_chunks")

        # 从每条向量检索结果里取出真正的文档实体 entity，组成列表
        embedding_chunks_list=[doc.get("entity") for doc in embedding_chunks]
        # 从每条 HyDE 检索结果里取出真正的文档实体 entity，组成列表
        hyde_embedding_chunks_list=[doc.get("entity") for doc in hyde_embedding_chunks]

        # 组装 RRF 输入：每一项是 (文档列表, 权重)
        rrf_inputs=[
            # 第一路：普通向量检索结果，权重 1.0
            (embedding_chunks_list,1.0),
            # 第二路：HyDE 检索结果，权重 1.0
            (hyde_embedding_chunks_list,1.0),
        ]

        # 调用内部融合函数，按 RRF 规则合并多路结果并重新排序
        rrf_merge_results=self._rrf_merge(rrf_inputs)

        # 只要融合后的文档本身，丢掉分数；doc 是文档，_ 是分数
        rrf_chunks=[doc for doc,_ in rrf_merge_results]
        # 把融合后的结果写回状态，供下游节点使用
        state["rrf_chunks"]=rrf_chunks
        # 打印融合结果，方便调试查看
        print(f"rrf_chunks:{rrf_chunks}")
        # 返回更新后的状态，继续走后续节点
        return state

    # RRF 融合核心方法：多路结果按排名加权打分，再按总分排序截断
    # rrf_inputs: [(文档列表, 权重), ...]；k: 平滑常数（默认 60）；max_results: 最多返回条数
    def _rrf_merge(self,rrf_inputs:List[Tuple],k:int=60,max_results:int=5):
        # 提示当前进入向量检索与假设性检索的融合逻辑
        print("向量搜索和假设性搜索的融合函数")
        # 用字典累计每个 chunk 的 RRF 总分：key=chunk_id，value=分数
        chunk_scores={}
        # 用字典保存每个 chunk 的完整文档内容，避免重复存多份
        chunk_data={}

        # 遍历每一路召回结果及其权重
        for rrf_input,weight in rrf_inputs:
            # 遍历该路结果中的每一篇文档；enumerate 得到排名 rank（从 0 开始）
            for rank,doc in enumerate(rrf_input):
                # 取出当前文档的唯一标识 chunk_id，用于跨路去重与累加分
                chunk_id= doc.get("chunk_id")
                # RRF 公式：分数 += 权重 / (k + 排名)；排名越靠前，加分越多
                chunk_scores[chunk_id]=chunk_scores.get(chunk_id,0.0) +weight /(k+rank)
                # 若该 chunk 还没存过文档内容，则存入一份（setdefault 不覆盖已有值）
                chunk_data.setdefault(chunk_id,doc)

        # 把「文档 + 分数」拼成列表，便于后续排序
        unsorted_result=[(chunk_data[cid],score) for cid,score in chunk_scores.items()]

        # 按分数从高到低排序（reverse=True 表示降序）
        sorted_results= sorted(
            unsorted_result,
            key=lambda x:x[1],
            reverse=True
        )
        # 若指定了 max_results，只取前 N 条；否则返回全部
        return sorted_results[:max_results] if max_results else sorted_results

# 仅在直接运行本文件时执行下面的测试代码（被其他模块 import 时不会跑）
if __name__ == '__main__':

    # 模拟两路检索结果
    mock_state = {
        # 模拟普通向量检索返回的 3 条结果（外层带 entity 包装）
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        # 模拟 HyDE 检索返回的 3 条结果（chunk_1、chunk_2 与上路重叠，会累加分）
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#3"}},
        ]
    }

    # 实例化 RRF 节点
    node_rrf = NodeRrf()
    # 调用节点（会走到 process），对 mock_state 做融合
    result = node_rrf(mock_state)
    # 把最终状态以缩进 JSON 形式打日志，方便肉眼检查
    logger.info(json.dumps(result, indent=4))
