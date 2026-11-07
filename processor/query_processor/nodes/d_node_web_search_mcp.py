# 导入 asyncio：用于在同步方法里运行异步 MCP 调用（asyncio.run）
import asyncio
# 导入 json：用于把 MCP 返回的文本解析成 Python 字典，以及测试时格式化打印
import json

# 导入 MCP 的 HTTP 流式客户端，用来连接外部 MCP 服务并调用工具
from agents.mcp import MCPServerStreamableHttp

# 导入百炼 MCP 的配置（服务地址、API Key 等）
from config.bailian_mcp_config import mcp_config
# 导入查询流水线的节点基类，本节点继承它
from processor.query_processor.base import NodeBase
# 导入查询图的状态类型，用来约定 state 的数据结构
from processor.query_processor.state import QueryGraphState
# 导入日志工具，用于打印节点运行信息
from tool.logger import logger


# 定义「Web 搜索 MCP」节点类，继承 NodeBase
class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息。
    未配置/连不上 MCP（例如不用阿里云 key）时，返回空结果，不中断整条流水线。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    # 节点主入口：接收上游状态，做 Web 搜索后返回更新后的状态
    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        logger.info(f"【{self.name}】节点逻辑")

        query = state.get("rewritten_query")
        docs = []

        # 未配置 MCP 地址或 Key 时直接跳过（纯 / 本地知识库场景）
        if not (mcp_config.mcp_base_url and mcp_config.api_key):
            logger.warning(f"【{self.name}】未配置 MCP，跳过联网搜索")
            return {"web_search_docs": docs}

        try:
            result = asyncio.run(self._mcp_call(query))
            if not result or not getattr(result, "content", None):
                logger.warning(f"【{self.name}】MCP 无返回内容，跳过")
                return {"web_search_docs": docs}

            json_text = result.content[0].text
            result_json_obj = json.loads(json_text)
            pages = result_json_obj.get("pages") or []
            for page in pages:
                snippet = (page.get("snippet") or "").strip()
                url = (page.get("url") or "").strip()
                title = (page.get("title") or "").strip()
                docs.append({"snippet": snippet, "url": url, "title": title})
        except Exception as e:
            #  方案下通常没有可用的百炼 MCP：失败不抛出，让后面节点继续
            logger.error(f"【{self.name}】联网搜索失败，已跳过: {e}")
            docs = []

        print(docs)
        return {"web_search_docs": docs}

    # 异步方法：连接 MCP 服务，调用联网搜索工具，并清理连接
    async def _mcp_call(self, query):
        search_mcp_tool = MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                "timeout": 10,
            },
            cache_tools_list=True,
            max_retry_attempts=3,
        )

        await search_mcp_tool.connect()
        try:
            result = await search_mcp_tool.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": query, "count": 3},
            )
        finally:
            await search_mcp_tool.cleanup()
        return result


if __name__ == "__main__":
    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？"
    }
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(json.dumps(result, indent=4, ensure_ascii=False))
