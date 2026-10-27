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
    节点功能，调用外部搜索引擎补充信息
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

        # TODO
        # 打印日志，标明当前正在执行本节点
        logger.info(f"【{self.name}】节点逻辑")

        # 从状态中取出改写后的查询词（用于联网搜索）
        query = state.get("rewritten_query")
        # 初始化空列表，用来存放整理后的网页搜索结果
        docs = []

        # 同步方式启动异步方法 _mcp_call，把查询发给 MCP 并拿到原始结果
        result = asyncio.run(self._mcp_call(query))

        # 从 MCP 返回对象里取出第一条内容的文本（一般是 JSON 字符串）
        json_text =result.content[0].text
        # 把 JSON 字符串解析成 Python 字典，方便按字段取值
        result_json_obj=json.loads(json_text)
        # 从解析结果中取出 pages 列表（每一项是一条网页结果）
        pages=result_json_obj.get("pages")
        # 遍历每一条网页结果，提取并整理字段
        for page in pages:
            # 取摘要片段；没有则用空字符串，并去掉首尾空白
            snippet=(page.get("snippet") or "").strip()
            # 取网页链接；没有则用空字符串，并去掉首尾空白
            url =(page.get("url") or "").strip()
            # 取网页标题；没有则用空字符串，并去掉首尾空白
            title=(page.get("title") or "").strip()
            # 把整理好的一条结果追加到 docs 列表
            docs.append({"snippet":snippet,"url":url,"title":title})

        # 打印整理后的搜索结果（调试用）
        print(docs)
        # 把 Web 搜索结果写回状态，供下游节点使用
        return {"web_search_docs": docs}

    # 异步方法：连接 MCP 服务，调用联网搜索工具，并清理连接
    async def _mcp_call(self, query):
        # 创建 MCP HTTP 客户端实例，指定服务名与连接参数
        search_mcp_tool = MCPServerStreamableHttp(
            # MCP 服务在客户端侧的名称标识
            name="search_mcp",
            # 连接参数字典
            params={
                # MCP 服务的基础 URL（来自配置）
                "url": mcp_config.mcp_base_url,
                # 请求头：用 Bearer Token 做鉴权（API Key 来自配置）
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                # 请求超时时间（秒）
                "timeout": 10
            },
            # 缓存工具列表，减少重复拉取工具定义的开销
            cache_tools_list=True,
            # 调用失败时的最大重试次数
            max_retry_attempts=3
        )

        # 建立与 MCP 服务的连接
        await search_mcp_tool.connect()
        # 调用名为 bailian_web_search 的工具，传入查询词和返回条数
        result = await search_mcp_tool.call_tool(
            # 要调用的 MCP 工具名称
            tool_name="bailian_web_search",
            # 工具参数：query 为搜索词，count 为希望返回的结果条数
            arguments={"query":query,"count":3}
        )
        # 清理 MCP 连接与相关资源，避免泄漏
        await search_mcp_tool.cleanup()
        # 把工具调用的原始结果返回给调用方
        return result

# 仅当本文件被直接运行时才执行下面的测试代码（被 import 时不会跑）
if __name__ == "__main__":

    # 构造测试用的初始状态：只给改写后的查询词
    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？"
    }

    # 执行节点的业务调用
    # 实例化本节点
    node_web_search_mcp = NodeWebSearchMcp()
    # 以可调用对象方式执行节点（内部会走到 process）
    result = node_web_search_mcp(init_state)
    # 把节点返回结果格式化成 JSON 并写入日志，方便查看
    logger.info(json.dumps(result, indent=4))
