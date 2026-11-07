# web/api/import_service.py

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config.minio_config import minio_config
from processor.import_processor.main_graph import KBImportWorkflow
from tool.logger import logger
from utils.minio_utils import get_minio_client
from utils.task_utils import (
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status,
)

load_dotenv()

app = FastAPI(
    title="掌柜智库-导入API",
    description="此文档是掌柜智库导入流程的API接口说明",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/import.html")
async def get_import_page():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    html_path = current_dir_parent_path / "page" / "import.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{html_path}")
    return FileResponse(html_path)


def run_graph_task(task_id: str, file_dir: str, import_file_path: str):
    """后台跑导入图。节点进度由基类写入任务表，这里再消费流式事件。"""
    try:
        update_task_status(task_id, "processing")
        init_state = {
            "task_id": task_id,
            "file_dir": file_dir,
            "import_file_path": import_file_path,
        }
        workflow = KBImportWorkflow()
        for event in workflow.graph.stream(init_state, stream_mode="updates"):
            if not isinstance(event, dict):
                continue
            for node_name in event:
                if isinstance(node_name, str) and node_name != "__end__":
                    add_done_task(task_id, node_name)
        update_task_status(task_id, "completed")
    except Exception as e:
        update_task_status(task_id, "failed")
        logger.info(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)


@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传，自动触发知识库导入全流程")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    接收 PDF/MD，按日期和任务 ID 存到本地，再尝试上传 MinIO，并为每个文件启动导入图。
    """
    data_based_root_dir = os.getenv("DATA_BASED_ROOT_DIR")
    if not data_based_root_dir:
        raise HTTPException(status_code=500, detail="未配置 DATA_BASED_ROOT_DIR")

    data_dir = os.path.join(data_based_root_dir, datetime.now().strftime("%Y%m%d"))
    task_ids = []

    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}")

        add_running_task(task_id, "upload_file")

        file_dir = os.path.join(data_dir, task_id)
        os.makedirs(file_dir, exist_ok=True)
        original_name = os.path.basename(file.filename or "upload.bin")
        import_file_path = os.path.join(file_dir, original_name)

        try:
            with open(import_file_path, "wb") as file_buffer:
                shutil.copyfileobj(file.file, file_buffer)
        finally:
            await file.close()
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{import_file_path}")

        minio_object_name = f"pdf_files/{datetime.now().strftime('%Y%m%d')}/{original_name}"
        try:
            minio_client = get_minio_client()
            minio_bucket_name = minio_config.bucket_name
            if minio_client is None or not minio_bucket_name:
                raise RuntimeError("MinIO 客户端未就绪")
            minio_client.fput_object(
                bucket_name=minio_bucket_name,
                object_name=minio_object_name,
                file_path=import_file_path,
                content_type=file.content_type or "application/octet-stream",
            )
            logger.info(f"[{task_id}] 文件已成功上传至MinIO，桶名：{minio_bucket_name}，对象名：{minio_object_name}")
        except Exception as e:
            logger.warning(f"[{task_id}] 文件上传MinIO失败，将继续执行本地处理流程，异常信息：{str(e)}", exc_info=True)

        add_done_task(task_id, "upload_file")
        background_tasks.add_task(run_graph_task, task_id, file_dir, import_file_path)
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")

    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")
    return {
        "code": 200,
        "message": f" 文件上传成功, total: {len(files)}",
        "task_ids": task_ids,
    }


@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    }
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}"
    )
    return task_status_info


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="127.0.0.1", port=8000)
