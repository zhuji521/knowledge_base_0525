"""
【文件说明】c_node_md_img.py —— Markdown 图片处理节点

这个节点只做一件事：
    把 Markdown 里的本地图片，换成「中文摘要 + MinIO 公网链接」。

在整条导入流程里的位置：
    b_node_pdf_to_md（PDF 转 md） → c_node_md_img（本节点） → d_node_document_split（文档切分）

5 个处理步骤（对应下面 process 方法）：
    步骤1  读出 md 正文、md 路径、图片目录
    步骤2  扫描 images 目录，筛出支持的图片，并找到每张图在 md 里的前后文
    步骤3  调用多模态模型，给每张图生成一句中文摘要
    步骤4  把图片上传到 MinIO，再用「摘要 + URL」替换 md 里的图片引用
    步骤5  另存为 xxx_new.md（原文件不动，防止误覆盖）

阅读提示：
    本文件是「逐行注释版」，只增加注释，没有改动任何代码逻辑。
"""

# =============================================================================
# 一、依赖导入
# =============================================================================

import base64                                        # 标准库：图片二进制 -> base64 文本，才能塞进多模态消息
import json                                          # 标准库：只在文件末尾调试时把 state 转成 JSON 打印
import logging                                       # 标准库：日志，取根 logger
import os                                            # 标准库：列目录、拆扩展名、拼路径
import time                                          # 标准库：取时间戳、限流时 sleep
import re                                            # 标准库：用正则匹配和替换 md 里的图片语法
from ast import Tuple, List, Dict                    # 无效导入：ast 里的同名类是语法树节点，会被下面的 typing 导入覆盖
from collections import deque                        # 标准库：双端队列，用它的 popleft 做滑动窗口限流
from pathlib import Path                             # 标准库：面向对象的路径操作，比字符串拼路径安全
from typing import Deque, Dict, List, Tuple          # 类型注解：双端队列、字典、列表、元组（覆盖上面 ast 的同名类）

from django.contrib.messages.constants import SUCCESS                        # 未使用：Django 消息级别常量（本项目并未安装 Django）
from django.contrib.messages.context_processors import messages              # 未使用：Django 消息上下文处理器
from django.db.models.expressions import result                              # 未使用：Django ORM 表达式
from langchain_deepseek import ChatDeepSeek          # LangChain 的 DeepSeek 客户端，用来调多模态模型
from minio import Minio                             # MinIO 客户端类型，只用于类型注解
from minio.deleteobjects import DeleteObject        # 批量删除 MinIO 对象时，用它包装对象名

from config.lm_config import lm_config               # 大模型配置：base_url、api_key、vl_model（视觉模型名）
from config.minio_config import minio_config         # MinIO 配置：endpoint、bucket_name、img_dir
from processor.import_processor.base import BaseNode, setup_logging   # 节点基类（提供 self.config / self.logger）、日志初始化函数
from processor.import_processor.exceptions import StateFieldError, FileProcessingError   # 自定义异常：state 字段缺失 / 文件处理失败
from processor.import_processor.state import ImportGraphState   # 流程图的状态类型（TypedDict）
from utils.llm_utils import get_llm_client           # 带缓存的 LLM 客户端工厂
from utils.minio_utils import get_minio_client       # 全局单例的 MinIO 客户端


# =============================================================================
# 二、节点实现
# =============================================================================

class NodeMDImg(BaseNode):                                    # 继承 BaseNode：自动获得 config、logger、统一的异常包装
    """
    Markdown 图片处理节点：
    1. 读出 md 和图片
    2. 用多模态模型给每张图写中文摘要
    3. 把摘要写回 md（后续步骤负责上传/替换）
    """

    name = "node_md_img"                                      # 节点名：基类用它生成 logger 名 "import.node_md_img"

    def process(self, state: ImportGraphState):
        """主流程：按步骤处理当前 state 里的 markdown 图片。"""
        logging.info(f"{self.name}节点开始执行...")            # 打一条「开始执行」日志

        # 步骤1：拿到 md 正文、路径、图片目录
        md_content, md_path_obj, images_dir = self._step_1_get_content(state)   # 拆包：正文、Path 对象、图片目录

        # 步骤2：扫描图片，并找到每张图在 md 里的上下文
        target_images = self._step_2_scan_images(md_content, images_dir)        # 列表，每个元素是 (文件名, 图片路径, (上文, 下文))
        print(f"target_images:{target_images}")                                 # 调试打印：本次要处理的图片清单

        # 步骤3：调用视觉模型，生成每张图的中文摘要
        summaries = self._step_3_generate_summaries(md_path_obj.stem, target_images)   # md_path_obj.stem = 去掉 .md 的文件名

        # 步骤4：上传图片并替换 md 中的图片引用
        new_md_content = self._step_4_upload_replace(md_path_obj.stem, target_images, summaries, md_content)   # 返回替换后的新正文

        # 步骤5：备份并写出新的 md 文件
        new_md_file_name = self._step_5_backup_new_md_file(state["md_path"], new_md_content)   # 写到 xxx_new.md，原文件保留

        # 把结果写回 state，供后续节点使用
        state["md_content"] = new_md_content                   # 更新正文：下游切分节点用的就是这一份
        state["md_path"] = new_md_file_name                    # 指向新文件，下游按这个路径读

        return state                                           # LangGraph 约定：节点返回更新后的 state

    def _step_1_get_content(self, state):
        """从 state 读取 md 路径和内容；没有内容就从文件读。"""
        md_path = state.get("md_path")                          # 从 state 取 md 路径，取不到返回 None（不会 KeyError）
        if not md_path:                                         # 空字符串、None 都算没给路径
            raise StateFieldError(file_name="md_path", expected_type=str)   # 抛自定义异常，提示缺字段
        md_path_obj = Path(md_path)                             # 转成 Path 对象，后面拼路径更方便

        if not md_path_obj.exists():                            # 先确认文件真的存在
            raise FileProcessingError(message=f"输入文件不存在:{md_path}")   # 不存在直接报错，避免后面出现更难懂的错

        # 优先用 state 里已有的内容，没有再读磁盘
        md_content = state["md_content"]                        # 直接按下标取； key 不存在会抛 KeyError
        if not md_content:                                      # 空字符串 / None 都算「还没内容」
            with open(md_path_obj, "r", encoding="utf-8") as md_file:   # with 打开文件，用完自动关闭
                md_content = md_file.read()                     # 一次性把整个 md 读进内存

        # 约定：图片放在 md 同级的 images 目录
        images_dir = md_path_obj.parent / "images"              # Path 支持用 / 拼路径：同级目录下的 images

        return md_content, md_path_obj, images_dir              # 一次性返回三个结果，避免上层再算一遍

    def _step_2_scan_images(self, md_content, images_dir):
        """
        遍历 images 目录，筛出支持的图片，
        并记录：文件名、本地路径、在 md 中的前后文。
        """
        target_images = []                                      # 结果列表：只放「格式支持 + 在 md 里被引用」的图片

        for image_file in os.listdir(images_dir):               # 列出 images 目录下的所有文件名（只列一层，不递归）
            file_ext = os.path.splitext(image_file)[1].lower()  # 取扩展名并转小写，如 ".PNG" -> ".png"
            # 扩展名不在白名单里就跳过
            if file_ext not in self.config.image_extensions:    # 白名单来自 ImportConfig.image_extensions（jpg/png/gif...）
                self.logger.warning(f"图片格式不支持,跳过:{image_file}")   # 记警告，方便排查为什么这张图没被处理
                continue                                        # 跳过本次循环，直接看下一张

            img_path = images_dir / image_file                  # 拼出这张图的完整本地路径

            # 在 md 正文里找这张图，并截取前后文
            context = self._find_image_in_md(md_content, image_file)   # 返回 (上文, 下文)；md 里没引用则返回 None

            if not context:                                     # 没找到引用
                self.logger.warning(f"图片未在MD中引用,跳过处理:{image_file}")   # 记警告：说明跳过原因
                continue                                        # 跳过这张图

            target_images.append((image_file, img_path, context))   # 收集：文件名、本地路径、上下文

        return target_images                                    # 返回待处理图片列表

    def _find_image_in_md(self, md_content: str, image_file: str) -> Tuple[str, str]:
        """
        在 md 里用正则找 `![...](...图片名...)`，
        返回图片前后各一段文字，方便模型结合上下文总结。
        """
        # 匹配 markdown 图片语法，且路径里包含该图片文件名
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")   # 拼正则：re.escape 转义文件名里的 . 等特殊字符
        match = pattern.search(md_content)                      # 只找第一处匹配

        if not match:                                           # 正文里没有这张图
            return None                                         # 返回 None，上层据此跳过

        start, end = match.span()                               # 取出匹配到的图片语法的「起始下标、结束下标」

        # 前后文各截取配置长度
        context_len = self.config.img_content_length            # 上下文长度，默认 200 个字符
        pre_text = md_content[max(0, start - context_len):start]            # 上文：起点不能小于 0，终点是图片语法开头
        post_text = md_content[end:min(len(md_content), end + context_len)] # 下文：终点不能超过正文长度

        return pre_text, post_text                              # 返回 (上文, 下文) 二元组

    def _step_3_generate_summaries(self, doc_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[
        str, str]:
        """对每张图调用视觉模型，返回 {图片文件名: 中文摘要}。"""
        # 入参：doc_stem 文档名；target_images 待处理图片列表；返回值：{图片文件名: 中文摘要}
        summaries = {}                                          # 结果字典：{文件名: 摘要}
        # 用双端队列记录请求时间，做滑动窗口限流
        request_deque = deque()                                 # 存放最近若干次请求的时间戳

        for img_file, image_path, context in target_images:     # 逐张图处理（文件名、本地路径、上下文）
            # 先限流，再调模型，避免触发 API 频率限制
            self._apply_api_rate_limit(request_deque, max_requests=10)   # 1 分钟窗口内最多 10 次，超了就阻塞等待
            summaries[img_file] = self._summarize_image(image_path, root_folder=doc_stem, image_content=context)   # 调模型拿摘要
        return summaries                                        # 返回 {文件名: 摘要}

    def _apply_api_rate_limit(
            self,
            request_times: Deque[float],                        # 入参：存放历史请求时间戳的队列，跨调用复用
            max_requests: int,                                  # 入参：窗口内允许的最大请求次数
            window_seconds: int = 60                            # 入参：滑动窗口时长（秒），默认 60 秒
    ) -> None:                                                  # 返回值：无，超限时原地阻塞等待
        """
        通用滑动窗口API速率限制器（抽离为公共工具）
        核心逻辑：维护请求时间戳双端队列，窗口内请求数超上限则自动等待，防止触发第三方API限流
        :param request_times: 存储请求时间戳的双端队列，需外部初始化（全局/单例），跨调用复用
        :param max_requests: 速率限制窗口内的最大允许请求次数
        :param window_seconds: 速率限制滑动窗口时长，默认60秒（1分钟）
        :return: None，超出限制时会阻塞等待
        """
        current_time = time.time()                              # 取当前时间戳（秒），本次判断的基准时间

        # 1. 清理滑动窗口外的过期请求时间戳，保证队列仅存窗口内的请求
        while request_times and current_time - request_times[0] >= window_seconds:   # 队首就是最早的那次请求
            request_times.popleft()                             # 已经超出窗口时长，从左边弹出

        # 2. 窗口内请求数达上限，计算并阻塞等待剩余时间
        if len(request_times) >= max_requests:                   # 窗口内请求数已达上限
            # 计算需要等待的时长（窗口总时长 - 最早请求已存在的时长）
            sleep_duration = window_seconds - (current_time - request_times[0])   # 等最早那次请求「滑出」窗口
            if sleep_duration > 0:                               # 只有算出正数才需要等
                logging.getLogger().info(
                    f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f} 秒")   # 日志：打印要等多久
                time.sleep(sleep_duration)                       # 阻塞等待，让最早的请求滑出窗口
                # 等待后更新当前时间，重新清理过期请求（避免等待期间有请求过期）
                current_time = time.time()                       # 等待结束，重新取时间
                while request_times and current_time - request_times[0] >= window_seconds:   # 再清一次过期时间戳
                    request_times.popleft()                      # 弹出已过期的

        # 3. 记录当前请求时间戳，加入滑动窗口队列
        request_times.append(current_time)                       # 把本次请求时间入队，供下次判断
        logging.getLogger().info(f"API请求时间戳已记录，当前{window_seconds}秒窗口内请求数：{len(request_times)}")   # 日志：当前窗口内请求数

    def _summarize_image(self, image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
        """把图片转成 base64，连同前后文一起发给视觉模型，得到一句中文摘要。"""
        # 读图片并编码，方便塞进多模态消息
        with open(image_path, "rb") as img_file:                 # 以二进制方式打开图片
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")   # 读全部字节 -> base64 字节 -> utf-8 字符串

        ai = ChatDeepSeek(                                       # 新建一个 DeepSeek 客户端实例
            model=lm_config.vl_model, temperature=lm_config.llm_temperature, base_url=lm_config.base_url,   # 模型名、温度、服务地址
            api_key=lm_config.api_key, model_kwargs={"response_format": {"type": "json_object"}}            # 密钥；要求返回 JSON
        )

        # 获取视觉大模型客户端
        vl_ai = get_llm_client(lm_config.vl_model)               # 真正发请求的是这个（工厂内部带缓存，不会重复创建）

        # 构造「文字说明 + 图片」的消息
        pre_text = ""                                            # 图片上文，先给默认空串
        post_text = ""                                           # 图片下文，先给默认空串
        if image_content:                                        # 有上下文才解包
            pre_text, post_text = image_content                  # 解包成「上文、下文」
        messages = [                                             # 多模态消息列表：一条 user 消息，内容包含文字 + 图片
            {
                "role": "user",                                  # 角色：用户
                "content": [                                     # content 是一个「多段内容」列表
                    {
                        "type": "text",                          # 第一段：文本
                        # 提示词：告诉模型图片属于哪个文件、前后文是什么、要什么样的摘要
                        "text": f"""这是"{root_folder}"文件中的一张图片，图片上文部分为"{pre_text}"，下文部分为"{post_text}"，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。"""
                    },
                    {
                        "type": "image_url",                     # 第二段：图片
                        "image_url": {                           # 图片内容对象：模型只认 URL 字段
                            "url": f"data:image/jpeg;base64,{image_base64}"   # 用 data URL 内联图片（这里固定写成 jpeg）
                        }
                    }
                ]
            }
        ]
        response = vl_ai.invoke(messages)                        # 同步调用模型，拿到完整响应对象
        # 去掉换行，方便直接当图片标题用
        return response.content.strip().replace("\n", "")        # 取文本内容 -> 去首尾空白 -> 删掉换行符

    def _step_4_upload_replace(self, doc_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]],
                               summaries: Dict[str, str], md_content: str):
        # 入参：文档名、待处理图片列表、图片摘要表、原 md 正文；返回替换后的 md 正文

        minio_client = get_minio_client()                        # 取全局单例 MinIO 客户端（模块加载时已确保桶存在）
        minio_img_dir = minio_config.img_dir                     # 图片在 MinIO 里的根目录（来自环境变量 MINIO_IMG_DIR）
        upload_dir = f"{minio_img_dir}/{doc_stem}"               # 本文档的图片目录：根目录/文档名
        print(f"将图片存入{upload_dir}目录下")                    # 打印一下上传位置，方便排查

        self.clean_minio_directory(minio_client, upload_dir)     # 先清空该目录，避免重跑时留下旧图

        urls = self.upload_image_batch(minio_client, upload_dir, target_images)   # 批量上传，返回 {文件名: 公网 URL}

        image_info=self.merge_summery_and_url(summaries,urls)    # 合并摘要和 URL -> {文件名: (摘要, URL)}

        md_content=self.process_md_file(md_content,image_info)   # 用「摘要 + URL」替换 md 里的图片引用
        return md_content                                        # 返回替换后的新正文

    def clean_minio_directory(self, minio_client: Minio, upload_dir):
        try:                                                     # 清理失败不应该阻断主流程，所以整体包一层 try
            objects_to_delete = minio_client.list_objects(minio_config.bucket_name, prefix=upload_dir, recursive=True)   # 列出该前缀下所有对象
            # 构造删除列表
            delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]   # 转成 DeleteObject 列表
            if delete_list:                                      # 有对象才需要删
                errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)   # 批量删除，返回「错误」迭代器
                for error in errors:                             # 迭代才会真正执行删除，并逐个拿到失败信息
                    self.logger.error(f"删除失败：{error}")       # 记录删除失败的对象
        except Exception as e:                                   # 兜底：网络/权限等任何异常
            self.logger.error(f"清理minio目录失败：{e}")          # 只记日志，不向上抛，避免影响后面的上传

    def upload_image_batch(self, minio_client: Minio, upload_dir: str,
                           target_images: List[Tuple[str, str, Tuple[str, str]]]):
        urls = {}                                                # 结果字典：{文件名: 公网 URL}
        for img_file, img_path, _ in target_images:              # 第三项（上下文）这里用不到，用 _ 忽略
            object_name = f"{upload_dir}/{img_file}"             # MinIO 里的对象名：目录 + 文件名
            print(f"上传文件: {object_name}")                     # 打印正在上传的对象
            urls[img_file] = self.upload_to_minio(minio_client, img_path, object_name)   # 单张上传，拿到 URL

        return urls                                              # 返回 {文件名: URL}

    def merge_summery_and_url(self, summaries: Dict[str, str], urls) -> Dict[str, Tuple[str, str]]:
        image_info = {}                                          # 结果：{文件名: (摘要, URL)}
        for image_file, summary in summaries.items():            # 遍历每张图的摘要
            if url := urls.get(image_file):                      # 海象运算符：只有上传成功拿到 URL 才合并
                image_info[image_file] = (summary, url)          # 摘要和 URL 配对存放

        return image_info                                        # 返回合并结果

    def process_md_file(self, md_content: str, image_info: Dict[str, Tuple[str, str]]):
        for image_file, (summary, url) in image_info.items():    # 逐张图替换 md 里的引用
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")   # 和查找时用的是同一条正则
            md_content=pattern.sub(lambda m: f"![{summary}]({url})",md_content)         # 全部替换成「摘要 + 新 URL」；用 lambda 免去反斜杠转义

        return md_content                                        # 返回替换后的正文

    def upload_to_minio(self, minio_client: Minio, img_path: str, object_name: str) -> str:
        ifSuccess = minio_client.fput_object(                    # 上传本地文件到 MinIO（返回值其实是对象，这里当标志用）
            bucket_name=minio_config.bucket_name,                # 桶名
            object_name=object_name,                             # 对象名（含目录）
            file_path=img_path,                                  # 本地图片路径
            content_type=f"image/{os.path.splitext(img_path)[1][1:]}"   # Content-Type：image/扩展名
        )
        print(f"上传结果: {ifSuccess}")                           # 打印上传结果

        url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{object_name}"   # 拼出公网访问地址

        print(f"图片url:{url}")                                   # 打印最终 URL

        return url                                               # 返回 URL，供替换 md 使用

    def _step_5_backup_new_md_file(self, origin_md_path: str, md_content: str) -> str:
        """
        步骤5：将处理后的MD内容保存为新文件（原文件不变，避免数据丢失）
        新文件命名规则：原文件名 + _new.md（如test.md → test_new.md）
        :param origin_md_path: 原始MD文件完整路径
        :param md_content: 处理后的新MD内容
        :return: 新MD文件的完整路径
        """
        # 构造新文件路径：替换原后缀为 _new.md
        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"   # test.md -> test_new.md

        # 写入新MD内容（覆盖写入，若文件已存在则更新）
        with open(new_md_file_name, "w", encoding="utf-8") as f:  # 文本写模式，UTF-8 编码
            f.write(md_content)                                   # 一次性写入全部内容

        self.logger.info(f"处理后MD文件已保存，新文件路径：{new_md_file_name}")   # 记录新文件路径，便于排查

        return new_md_file_name                                   # 返回新文件完整路径


# =============================================================================
# 三、本地调试入口（直接 python c_node_md_img.py 就能单独跑这个节点）
# =============================================================================

if __name__ == "__main__":                                       # 只有直接运行本文件时才走调试逻辑
    # 本地单独跑这个节点做调试
    setup_logging()                                              # 初始化日志格式（时间、logger 名、级别）

    init_state = {                                               # 构造一个最小可用的 state
        "md_path": "D:\output\hak180\hak180.md",                 # 要调试的 md 文件完整路径
        "md_content": None                                       # 正文先给 None，让节点自己从磁盘读
    }

    node = NodeMDImg()                                           # 实例化节点
    result = node(init_state)                                    # 调用基类的 __call__，内部再走 process

    dumps = json.dumps(result)                                   # 把处理后 state 序列化成 JSON 字符串，方便查看
    print(dumps)                                                 # 打印结果
