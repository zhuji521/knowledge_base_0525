# web/api/query_service.py

import asyncio
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from processor.query_processor.main_graph import KBQueryWorkflow
from utils.mongo_history_utils import clear_history, get_recent_messages
from utils.sse_utils import SSEEvent, bind_sse_stream, create_sse_queue, has_terminal_error, push_to_session, sse_generator
from utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_task_result,
    get_task_status,
    set_task_result,
    update_task_status,
)

app = FastAPI(
    title="掌柜智库-查询API",
    description="此文档是掌柜智库查询流程的API接口说明",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """查询请求数据结构"""

    query: str = Field(..., description="查询内容")
    session_id: str | None = Field(default=None, description="会话ID")
    is_stream: bool = Field(default=False, description="是否流式返回")


@app.get("/chat.html")
async def chat():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    html_path = current_dir_parent_path / "page" / "chat.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{html_path}")
    return FileResponse(html_path)


def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    print(f"开始流程图处理...{session_id} {user_query} {is_stream}")
    init_state = {
        "original_query": user_query,
        "session_id": session_id,
        "is_stream": is_stream,
    }
    try:
        if is_stream:
            bind_sse_stream(session_id)
        workflow = KBQueryWorkflow()
        result = workflow.run(init_state, stream=is_stream)
        # stream=True 时 run 返回生成器，消费后图才会执行，节点里的推送才会发生
        if is_stream:
            for _ in result:
                pass
        if is_stream and has_terminal_error(session_id):
            update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        else:
            update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        print(f"流程执行异常: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            if not has_terminal_error(session_id):
                push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
        else:
            set_task_result(session_id, "answer", f"查询失败：{e}")


@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    1 解析参数
    2 更新任务状态
    3 调用处理流程图
    4 返回结果
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    is_stream = request.is_stream
    if is_stream:
        create_sse_queue(session_id)
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
    print("开始处理流程... 是否流式:", is_stream, f"其他参数:{user_query}, session_id:{session_id}")

    if is_stream:
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        print("开始处理结果....")
        return {
            "message": "结果正在处理中...",
            "session_id": session_id,
        }

    await asyncio.to_thread(run_query_graph, session_id, user_query, is_stream)
    answer = get_task_result(session_id, "answer", "")
    status = get_task_status(session_id)
    return {
        "message": "处理失败" if status == TASK_STATUS_FAILED else "处理完成！",
        "session_id": session_id,
        "answer": answer,
        "done_list": get_done_task_list(session_id),
    }


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    print("调用流式/stream...")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """清空指定会话的历史记录。"""
    count = clear_history(session_id)
    return {"message": "历史会话已清空", "deleted_count": count}


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """查询当前会话历史记录。"""
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for record in records:
            items.append({
                "_id": str(record.get("_id")) if record.get("_id") is not None else "",
                "session_id": record.get("session_id", ""),
                "role": record.get("role", ""),
                "text": record.get("text", ""),
                "rewritten_query": record.get("rewritten_query", ""),
                "item_names": record.get("item_names", []),
                "ts": record.get("ts"),
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.get("/health")
async def health():
    """检查服务是否正常。"""
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
