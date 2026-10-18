import json
import logging
import time
import zipfile
from http.client import responses
from pathlib import Path
from urllib.parse import urlsplit

import requests
from django.utils.archive import extract

from config.mineru_config import mineru_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, PdfConversionError
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析

    在整条导入流水线中排在入口节点之后。入口判断文件是 PDF 时，
    会走到这里：确认本地 PDF 和输出目录都存在，再交给 MinerU 解析成
    Markdown，最后把正文和文件路径写回状态，交给下一个节点。
    """

    # LangGraph 里这个节点的名字，日志和报错都会带上它
    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        """
        节点主流程，基类 __call__ 会调用这里。

        输入 state 至少要有 pdf_path、file_dir。
        输出时会补上 md_content（Markdown 正文）和 md_path（md 文件路径）。
        """
        logging.info(f"{self.name}节点开始执行...")

        # 第 1 步：检查状态里的路径是否齐全、文件是否真的在磁盘上
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 第 2 步：把 PDF 交给 MinerU，等到解析完成，拿到结果压缩包的下载地址
        # 目前是占位实现，直接返回字符串 "下载url"，还没有真正上传和轮询
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)
        print(f"已经获得下载地址url{zip_url}")

        # 第 3 步：下载压缩包、解压，按 PDF 文件名（不含后缀）得到 md 路径
        # 目前也是占位，直接返回字符串 "md_path"
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        # 读出 Markdown 全文。读失败时只打印提示，不会把异常抛出去
        try:
            with open(md_path, "r", encoding="utf-8") as md_file:
                md_content = md_file.read()
        except:
            print("其实你的程序已经成功了!!!")
        finally:
            # 无论读成功还是失败，都会执行到这里
            print("over")

        # 把正文和路径写回状态，后续节点（图片处理、切分等）从这里取
        # 如果上面 open 失败，md_content 还没被赋值，这里会再抛 NameError
        state["md_content"] = md_content
        state["md_path"] = md_path

        return state

    def _step_1_validate_paths(self, state: ImportGraphState):
        """
        校验输入。

        返回两个 Path 对象：PDF 文件路径、输出目录路径。
        缺字段抛 StateFieldError，路径在磁盘上不存在抛 FileProcessingError。
        """
        # 状态里必须带上待转换的 PDF 路径
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path", expected_type=str)

        # 状态里必须带上结果要落到哪个目录
        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir", expected_type=str)

        # 转成 Path，后面可以用 .exists()、.stem 这类方法
        pdf_path_obj = Path(pdf_path)
        output_dir_obj = Path(file_dir)

        # 文件或目录不存在就停下来，避免后面上传一个不存在的文件
        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"输入文件不存在:{pdf_path}")
        if not output_dir_obj.exists():
            raise FileProcessingError(message=f"输出目录不存在:{output_dir_obj}")

        return pdf_path_obj, output_dir_obj

    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        """
        上传 PDF 到 MinerU，并轮询直到解析完成。
        设计上应返回结果 zip 的下载地址。当前未实现，固定返回 "下载url"。
        """
        logging.info(f"_step_2_upload_and_poll上传文件到mineru服务器...")

        api_token = mineru_config.api_token
        base_url = mineru_config.base_url

        if not api_token:
            raise FileProcessingError(message="api_token未配置")
        if not base_url:
            raise FileProcessingError(message="base_url未配置")

        header = {
            "Content-type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }

        url = f"{base_url}/file-urls/batch"
        response = requests.post(url, headers=header, json=data)
        if response.status_code != 200:
            raise FileProcessingError(message=f"申请上传文件失败:{response.text}")
        result = response.json()
        if result.get("code") != 0:
            raise FileProcessingError(message=f"申请上传文件失败:{result.get('message')}")
        batch_id = result["data"]["batch_id"]
        signed_urls = result["data"]["file_urls"][0]

        with open(pdf_path_obj, "rb") as pdf_file:
            res_upload = requests.put(signed_urls, data=pdf_file)
            if res_upload.status_code != 200:
                raise PdfConversionError(f"文件上传失败,状态码:{res_upload.status_code},响应结果:{res_upload}")
            self.logger.info(f"文件上传成功!")

            poll_url = f"{base_url}/extract-results/batch/{batch_id}"
            start_time = time.time()
            timeout_seconds = 600
            poll_interval = 3

            while True:
                end_time = time.time() - start_time
                if end_time > timeout_seconds:
                    raise FileProcessingError(message="获得下载地址超时")
                try:
                    res_poll = requests.get(url=poll_url, headers=header, timeout=10)
                except Exception as e:
                    self.logger.error(f"轮询接口异常:{e}")
                    time.sleep(poll_interval)
                    continue

                if res_poll.status_code != 200:
                    raise PdfConversionError(
                        f"[任务轮换]HTTP请求失败,状态码:{res_poll.status_code},相应内容:{res_poll}")

                poll_data = res_poll.json()

                if poll_data.get("code") != 0:
                    raise PdfConversionError(f"[任务轮询]任务失败,错误信息:{poll_data.get('message')}")

                extract_results = poll_data["data"]["extract_result"]
                extract_result = extract_results[0]
                extract_state = extract_result["state"]

                if extract_state == "done":
                    full_zip_url = extract_result["full_zip_url"]
                    return full_zip_url
                elif extract_state == "failed":
                    err_msg = extract_result.get("err_msg", "未知错误,无具体信息")
                    raise PdfConversionError(f"[任务轮询]解析任务失败!batch_id:{batch_id},错误信息:{err_msg}")
                else:
                    self.logger.info(
                        f"[任务轮询]处理中...已耗时{int(end_time)}s,状态:{extract_state},,batch_id:{batch_id}"
                    )
                    time.sleep(poll_interval)

        return "下载url"

    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
        """
        下载 zip、解压，并把 Markdown 放到输出目录。

        pdf_stem 是 PDF 文件名去掉后缀，例如 "hak180产品安全手册"。
        设计上的落盘位置类似：
        D:\\output\\hak180产品安全手册\\hak180产品安全手册.md
        当前未实现，固定返回字符串 "md_path"。
        """
        logging.info(f"_step_3_download_and_extract下载并解压改名")

        response = requests.get(zip_url)
        if response.status_code != 200:
            raise FileProcessingError(message=f"获得下载文件失败:{response.text}")
        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        with open(zip_save_path, "wb") as f:
            f.write(response.content)

        extract_target_dir = output_dir_obj / pdf_stem
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_save_path, mode="r") as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)
        self.logger.info(f"[ZIP已解压]已清空旧的解压目录:{extract_target_dir}")

        self.logger.info(f"[MD重命名]找到MinerU生成的full.md文件")
        target_md_file = extract_target_dir / "full.md"

        self.logger.info(f"[MD重命名]开始将full.md文件进行重命名")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")

        target_md_file.rename(new_md_path)
        self.logger.info(f"[MD重命名]重命名成功,文件名:{pdf_stem}.md")

        return str(new_md_path.absolute())


# 直接运行本文件时走这里，不经过整张导入图，只单独测这个节点
if __name__ == "__main__":
    # 打开控制台日志，能看到基类打出的「开始 / 完成」
    setup_logging()

    # 最小测试状态：一份本地 PDF，结果目录是 D:\output
    init_state = {
        "pdf_path": r"D:\hak180.pdf",
        "file_dir": r"D:\output"
    }

    # node(...) 会进基类 __call__，再进上面的 process
    node = NodePDFToMD()
    result = node(init_state)

    # 把更新后的状态打成中文可读的 JSON（ensure_ascii=False 保留中文）
    dumps = json.dumps(result, ensure_ascii=False, indent=4)
    print(dumps)
