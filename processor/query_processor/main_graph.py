from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.query_processor.nodes.a_node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.b_node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.c_node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_processor.nodes.d_node_web_search_mcp import NodeWebSearchMcp
from processor.query_processor.nodes.e_node_rrf import NodeRrf
from processor.query_processor.nodes.f_node_rerank import NodeRerank
from processor.query_processor.nodes.g_node_answer_output import NodeAnswerOutput
from processor.query_processor.state import QueryGraphState

from tool.logger import logger
load_dotenv()

class KBQueryWorkflow:

    def __init__(self):
        self.workflow=StateGraph(QueryGraphState)

        self._init_nodes()

        self._register_node()

        self._setup_routes()

        self._compiled_app=None

    def _init_nodes(self):
        print("实例化所有节点...")
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()

    def _register_node(self):
        print("注册所有节点...")
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm)  # 确认主体
        self.workflow.add_node("node_search_embedding", self.node_search_embedding)  # 向量搜索
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde)  # 假设性答案向量搜索
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp)  # 联网搜索
        self.workflow.add_node("node_rrf", self.node_rrf)  # 排序
        self.workflow.add_node("node_rerank", self.node_rerank)  # 重排
        self.workflow.add_node("node_answer_output", self.node_answer_output)  # 生成

    def _setup_routes(self):
        print("设置路由规则...")
        self.workflow.set_entry_point("node_item_name_confirm")

        self.workflow.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,
            path_map={
                "node_search_embedding":"node_search_embedding",
                "node_search_embedding_hyde":"node_search_embedding_hyde",
                "node_web_search_mcp":"node_web_search_mcp",
                "node_answer_output":"node_answer_output"
            }
        )

        self.workflow.add_edge("node_search_embedding","node_rrf")
        self.workflow.add_edge("node_search_embedding_hyde","node_rrf")
        self.workflow.add_edge("node_web_search_mcp","node_rrf")

        self.workflow.add_edge("node_rrf","node_rerank")
        self.workflow.add_edge("node_rerank","node_answer_output")
        self.workflow.add_edge("node_answer_output",END)

    def _route_after_item_name_confirm(self,state:QueryGraphState) ->str:
        print("路由-条件...")
        if state.get("answer"):
            return "node_answer_output"

        return ["node_search_embedding","node_search_embedding_hyde","node_web_search_mcp"]

    def compile(self):
        print("编译图...")
        if not self._compiled_app:
            self._compiled_app=self.workflow.compile()
        return self._compiled_app

    def run(self, initial_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        print("调用图...")
        if not self._compiled_app:
            self.compile()
        if stream:
            return self._compiled_app.stream(initial_state)
        else:
            return self._compiled_app.invoke(initial_state)

if __name__ == "__main__":
    workflow = KBQueryWorkflow()
    response = workflow.run(initial_state={"original_query": "哥们儿，B530这个东东咋鼓捣上啊？"}, stream=True)
    print(response)
    # 画图
    print(workflow.compile().get_graph().draw_ascii())



