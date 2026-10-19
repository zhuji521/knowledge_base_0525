import base64
import json
import logging
import os
import time
from ast import Tuple, List, Dict
from collections import deque
from pathlib import Path
from typing import re, Deque

from django.contrib.messages.context_processors import messages
from django.db.models.expressions import result
from langchain_deepseek import ChatDeepSeek

from config.lm_config import lm_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState
from utils.llm_utils import get_llm_client


class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行...")

        md_content, md_path_obj, images_dir = self._step_1_get_content(state)

        target_images = self._step_2_scan_images(md_content, images_dir)
        print(f"target_images:{target_images}")

        summaries = self._step_3_generate_summaries(md_path_obj.stem, target_images)

        new_md_content = self._step_4_upload_replace(md_path_obj.stem, target_images, summaries, md_content)

        new_md_file_name = self._self_5_backup_new_md_file(state["md_path"], new_md_content)

        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name

        return state

    def _step_1_get_content(self, state):
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(file_name="md_path", expected_type=str)
        md_path_obj = Path(md_path)

        if not md_path_obj.exists():
            raise FileProcessingError(message=f"输入文件不存在:{md_path}")

        md_content = state["md_content"]
        if not md_content:
            with open(md_path_obj, "r", encoding="utf-8") as md_file:
                md_content = md_file.read()

        images_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, images_dir

    def _step_2_scan_images(self, md_content, images_dir):

        target_images = []

        for image_file in os.listdir(images_dir):
            file_ext = os.path.splitext(image_file)[1].lower()
            if file_ext not in self.config.image_extensions:
                self.logger.warning(f"图片格式不支持,跳过:{image_file}")
                continue

            img_path = images_dir / image_file
            context = self._find_image_in_md(md_content, image_file)

            target_images.append((image_file, img_path, context))

        return target_images

    def _find_image_in_md(self, md_content: str, image_file: str) -> Tuple[str, str]:

        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
        match = pattern.search(md_content)

        if not match:
            return None

        start, end = match.span()

        context_len = self.config.img_content_length
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content)), end + context_len]

        return pre_text, post_text

    def _step_3_generate_summaries(self, doc_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[
        str, str]:

        summaries = {}
        request_deque = deque()

        for img_file, image_path, context in target_images:
            self._apply_api_rate_limit(request_deque, max_requests=10)
            summaries[img_file] = self._summarize_image(image_path, root_folder=doc_stem, image_context=context)
        return summaries

    def _apply_api_rate_limit(
            self,
            request_times: Deque[float],
            max_requests: int,
            window_seconds: int = 60
    ) -> None:
        """
        通用滑动窗口API速率限制器（抽离为公共工具）
        核心逻辑：维护请求时间戳双端队列，窗口内请求数超上限则自动等待，防止触发第三方API限流
        :param request_times: 存储请求时间戳的双端队列，需外部初始化（全局/单例），跨调用复用
        :param max_requests: 速率限制窗口内的最大允许请求次数
        :param window_seconds: 速率限制滑动窗口时长，默认60秒（1分钟）
        :return: None，超出限制时会阻塞等待
        """
        current_time = time.time()

        # 1. 清理滑动窗口外的过期请求时间戳，保证队列仅存窗口内的请求
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()

        # 2. 窗口内请求数达上限，计算并阻塞等待剩余时间
        if len(request_times) >= max_requests:
            # 计算需要等待的时长（窗口总时长 - 最早请求已存在的时长）
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                logging.getLogger().info(
                    f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f} 秒")
                time.sleep(sleep_duration)
                # 等待后更新当前时间，重新清理过期请求（避免等待期间有请求过期）
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()

        # 3. 记录当前请求时间戳，加入滑动窗口队列
        request_times.append(current_time)
        logging.getLogger().info(f"API请求时间戳已记录，当前{window_seconds}秒窗口内请求数：{len(request_times)}")

    def _summarize_image(self, image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:

        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")


        ai = ChatDeepSeek(
            model=lm_config.vl_model, temperature=lm_config.llm_temperature, base_url=lm_config.base_url,
            api_key=lm_config.api_key, model_kwargs={"response_format": {"type": "json_object"}}
        )

        vl_ai = get_llm_client(lm_config.vl_model)

        messages=[
            {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": f"""这是"{root_folder}"文件中的一张图片，图片上文部分为"{image_content[0]}"，下文部分为"{image_content[1]}"，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]

            }
        ]
        response = vl_ai.invoke(messages)
        return response.content.strip().replace("\n","")

        vl_ai.invoke()

        pass


if __name__ == "__main__":
    setup_logging()

    init_state = {
        "md_path": "D:\output\hak180.md",
        "md-content": None
    }

    node = NodeMDImg()
    result = node(init_state)

    dumps = json.dumps(result)
    print(dumps)
