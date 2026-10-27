import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        query = state.get("rewritten_query")
        docs = []

        result = asyncio.run(self._mcp_call(query))

        json_text =result.content[0].text
        result_json_obj=json.loads(json_text)
        pages=result_json_obj.get("pages")
        for page in pages:
            snippet=(page.get("snippet") or "").strip()
            url =(page.get("url") or "").strip()
            title=(page.get("title") or "").strip()
            docs.append({"snippet":snippet,"url":url,"title":title})

        print(docs)
        return {"web_search_docs": docs}

    async def _mcp_call(self, query):
        search_mcp_tool = MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                "timeout": 10
            },
            cache_tools_list=True,
            max_retry_attempts=3
        )

        await search_mcp_tool.connect()
        result = await search_mcp_tool.call_tool(
            tool_name="bailian_web_search",
            arguments={"query":query,"count":3}
        )
        await search_mcp_tool.cleanup()
        return result

if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？"
    }

    # 执行节点的业务调用
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(json.dumps(result, indent=4))