# 模块说明：导入流程的入口节点，负责校验文件路径并识别文件类型
import logging  # 日志模块，用于记录节点运行过程
from pathlib import Path  # 路径工具，用于判断文件是否存在、获取文件名和后缀

from processor.import_processor.base import BaseNode  # 节点基类，提供统一的节点接口
from processor.import_processor.exceptions import StateFieldError, FileProcessingError  # 自定义异常
from processor.import_processor.state import ImportGraphState  # 图中流转的状态数据类型


class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """

    # 节点名称：用于日志输出，也用于在流程图中注册和定位该节点
    name = "node_entry"

    def process(self, state: ImportGraphState):
        # 第 1 步：记录日志，表示入口节点开始执行
        logging.info(f"{self.name}节点开始执行...")

        # 第 2 步：从状态中取出待导入文件的路径
        import_file_path = state.get("import_file_path")

        # 第 3 步：校验路径不能为空，否则抛出状态字段异常
        if not import_file_path:
            raise StateFieldError(field_name="import_file_path", expected_type=str)

        # 第 4 步：把字符串路径转换为 Path 对象，方便后续操作
        import_file_path_obj = Path(import_file_path)
        # 第 5 步：校验文件是否真实存在，不存在则抛出文件处理异常
        if not import_file_path_obj.exists():
            raise FileProcessingError(message=f"文件不存在:{import_file_path}")

        # 第 6 步：根据文件后缀判断文件类型，并开启对应的读取开关
        if import_file_path_obj.suffix == ".md":
            # Markdown 文件：启用 MD 读取
            state["is_md_read_enabled"] = True
        elif import_file_path_obj.suffix==".pdf":
            # PDF 文件：启用 PDF 读取
            state["is_pdf_read_enabled"]=True
        else:
            # 其他格式：直接报错，不支持处理
            raise FileProcessingError(message=f"不支持的文件格式:{import_file_path}")


        # 第 7 步：写入后续节点要用的文件信息
        state["file_title"]=import_file_path_obj.stem  # 文件名（不含扩展名）
        state["file_dir"]=r"D:outpput"  # 输出目录

        # 第 8 步：返回更新后的状态，交给下一个节点继续处理
        return state
