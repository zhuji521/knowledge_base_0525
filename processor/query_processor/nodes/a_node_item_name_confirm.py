import json
from typing import Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.llm_utils import get_llm_client
from utils.mongo_history import get_recent_messages, save_chat_message


class NodeItemNameConfirm(NodeBase):
    name: str = "node_item_name_confirm"

    def __init__(self):
        self._step_8_write = None

    def process(self, state: QueryGraphState) -> QueryGraphState:
        logger.info(f"[{self.name}]节点逻辑")

        session_id, original_query = self._step_1_validate_param(state)

        history = get_recent_messages(session_id)
        state["history"] = history

        message_id=save_chat_message(session_id,"user",original_query)

        extract_res=self._step_4_extract_info(original_query,history)
        item_names=extract_res["item_names"]
        rewritten_query=extract_res["rewritten_query"]
        state["rewritten_query"]=rewritten_query
        state["item_names"]=item_names
        print("extract_res:",extract_res)

        align_result = {}
        if len(item_names) >= 0:
            query_results = self._step_5_vectorize_and_query(item_names)
            align_result = self._step_6_align_item_names(query_results)
        else:
            logger.info("Node: 未提取到商品名，跳过向量检索")

        state=self._step_7_check_confirmation(state,align_result,history)

        self._step_8_write_history(state,session_id,rewritten_query,message_id)

        return state

    def _step_1_validate_param(self, state):
        print("step_1:参数检验")

        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("核心参数session_id缺失")

        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("核心参数original_query缺失")

        return session_id, original_query

    def _step_4_extract_info(self, original_query, history) ->Dict:
        print("step_4:模型提取意图主")





        ai_client= get_llm_client()
        history_text=""
        for msg in history:
            role =msg.get("role")
            content=msg.get("text")
            history_text += f"{role}:{content}\n"

        user_prompt=ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text,
            query=original_query,
        )

        messages=[
            SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        response=ai_client.invoke(messages)
        response_content=response.content
        if response_content.startswith("```json"):
            response_content = response_content.replace("```json", "").replace("```", "")

        result=json.loads(response_content)
        if "item_names" not in result:
            result["item_names"]=[]
        if "rewritten_query" not in result:
            result["rewritten_query"]=original_query
        if result["item_names"]:
            result["item_names"]=[name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "") for name in result["item_names"]]

        # ai_client.invoke(user_prompt)
        #
        # for ms in history:
        #     role=msg.get("")
        #
        result={
            "item_names":[],
            "rewritten_query":original_query,
        }
        return result

    def _step_5_vectorize_and_query(self, item_names) ->List[Dict]:
        print("step_5:向量化并检索,检索出来对应item_name的向量库中的相似商品名称的列表")

        result:List[Dict]=[]
        return result

    def _step_6_align_item_names(self, query_results: List[Dict])->Dict:
        print("step_6:对齐结果")

        return{
            "confirmed_item_name":[],
            "options":[],
        }

    def _step_7_check_confirmation(self, state, align_result:Dict, history):
        print("step_7:状态state信息,根据第六步高分低分对齐结果整理")

        return state

    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        print("step_8:写入历史会话,更新")
        pass

if __name__ == "__main__":
    # 初始化图状态
    init_state = {
        "original_query": "hak180这玩儿意咋鼓捣呀？",
        "session_id":1234567
    }

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()
    # 执行节点的单元测试
    result = node_item_name_confirm(init_state)
    # 将返回的图状态进行json序列化
    # json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logger.info(result)
