# ==================== 标准库 ====================
# json: 备份切分结果为 JSON
# re: 正则匹配 Markdown 标题
# Path: 拼接备份文件路径
# List/Dict: 类型注解
import json
import re
from dataclasses import replace
from pathlib import Path
from sys import prefix, path
from typing import List, Dict

# ==================== 第三方库 ====================
# RecursiveCharacterTextSplitter: 超长文本按长度递归再切
import sections
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==================== 项目内部 ====================
# BaseNode: 节点基类
# StateFieldError: 状态字段缺失异常
# get_config: 读取 max_content_length 等配置
# ImportGraphState: 导入流程图状态
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.import_config import get_config
from processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """

    # 节点在图中的注册名
    name = "node_document_split"

    def process(self, state: ImportGraphState):
        # ---------- 主流程共 6 步 ----------
        # 步骤1：取出 Markdown 正文、文件标题
        content, file_title = self._step_1_get_inputs(state)

        # 步骤2：按 # 标题初切，得到 sections，并统计标题数、行数
        sections, title_count, lines_count = self._step_2_split_by_title(content, file_title)

        # 步骤3：无标题时整篇兜底成一块（标题=「无标题」）
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)

        # 步骤4：过长再切、过短合并
        sections = self._step_4_refine_chunks(sections)

        # 调试打印：每个切块的标题与内容
        for section in sections:
            print(f"{section['title']}")
            print(f"{section['content']}")
            print("===============================================================================")

            print(f"标题数量:{title_count}")
            print(f"行数:{lines_count}")

        # 步骤5：输出切分统计日志
        self._step_5_print_step(lines_count, sections)

        # 步骤6：把 chunks 备份成 JSON
        self._step_6_backup(state, sections)

        # 结果写入 state，供后续节点使用
        state["chunks"] = sections
        return state

    def _step_1_get_inputs(self, state):
        # 从 state 取 md 正文和文件标题，并统一换行符
        print("node_document_split:步骤1:参数处理")
        # 取 Markdown 全文；缺失则报错
        content = state.get("md_content")
        if not content:
            raise StateFieldError(field_name="md_content", expected_type=str)

        # 取文件标题；缺失则报错
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="md_content", expected_type=str)

        # 统一换行：\r\n、\r → \n
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        return content, file_title

    def _step_2_split_by_title(self, content, file_title):
        # 按 Markdown 标题切成多个 section；代码块内的 # 不当标题
        print("node_document_split:步骤2:标题切(初切)")
        # sections: 切分结果
        # title_count: 标题命中次数
        # lines: 按行拆分后的全文
        # current_lines: 当前段落已收集的行
        # in_code_block: 是否处于代码块内
        # current_title: 当前段落标题
        sections: List[Dict[str, str]] = []
        title_count = 0
        lines = content.split("\n")
        current_lines = []
        in_code_block = False
        current_title = ""

        # 匹配 1~6 级标题，如：## 概述
        title_pattern = r"\s*#{1,6}\s+.+"

        def _flush_section():
            # 把 current_lines 收成一个 section；空则跳过
            if not current_lines:
                return
            # title / content / parent_title / file_title
            sections.append(
                {
                    "title": current_title,
                    "content": "\n".join(current_lines),
                    "parent_title": "",
                    "file_title": file_title
                }
            )

        # 逐行扫描全文
        for line in lines:
            striped_line = line.strip()

            # 代码围栏：切换 in_code_block，本行仍写入当前段
            if striped_line.startswith("```") or striped_line.startswith("~~~"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            # 非代码块内且是标题：冲刷上一段，开启新段
            if (not in_code_block) and (re.match(title_pattern, line)):
                _flush_section()
                current_title = striped_line
                current_lines = [current_title]
                title_count += 1

            else:
                # 普通正文：追加到当前段
                current_lines.append(line)
        # 收尾：最后一段也要冲刷
        _flush_section()
        print(f"标题初切:sections:{sections}")
        return sections, title_count, len(lines)

    def _step_3_handle_no_title(self, content, sections, title_count, file_title):
        # 全文无标题时，整篇做成一块，标题固定为「无标题」
        print("node_document_split:步骤3:无标题兜底(默认标题)")
        if title_count == 0:
            return [{"title": "无标题", "content": content, "file_title": file_title}]

        return sections

    def _step_4_refine_chunks(self, sections):
        # 精细化：长切 → 短合 → 补全 parent_title
        print("node_document_split:步骤4:块精细化处理(长切短合)")

        # 过长的 section 再切成多块
        refind_split = []
        for sec in sections:
            refind_split.extend(self.split_long_section(sec))

        # 过短且同父标题的相邻块合并
        final_sections = self.merge_short_section(refind_split)

        # 没有 parent_title 时用自身 title 补上
        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""

        return final_sections

    def _step_5_print_step(self, lines_count, sections):
        # 打印原始行数、最终 chunk 数量
        print("node_document_split:步骤5:打印日志")

        chunk_num=len(sections)

        self.logger.info("-"*50+"文档切分统计信息"+"-"*50)
        self.logger.info(f"MD原始文本总行数:{lines_count}")
        self.logger.info(f"最终生成Chunk数量:{chunk_num}")

    def _step_6_backup(self, state, sections):
        # 在 md 同目录写出 {file_title}_chunks.json
        print("node_document_split:步骤6:备份")
        path=Path(state.get("md_path")).parent/f'{state.get("file_title")}_chunks.json'
        # ensure_ascii=False 保留中文；indent=2 便于阅读
        with open(path,"w",encoding="utf-8") as f:
            json.dump(
                sections,
                f,
                ensure_ascii=False,
                indent=2
            )

    def split_long_section(self, section: Dict[str, str]):
        # 单块超过 max_content_length 时再切；子块标题形如「原标题-1」
        print("node_document_split:步骤4方法1长切")
        content = section.get("content", "")
        content_len = len(content)

        # 未超长：原样返回（包成列表方便 extend）
        if content_len <= get_config().max_content_length:
            return [section]

        title = section.get("title")

        # 每块会挂「标题\n\n」，可用长度要扣掉标题占用
        prefix = f"{title}\n\n" if title else ""
        available_len = get_config().max_content_length - len(prefix)

        # 正文若已含标题行，先剥掉，避免拼回时重复
        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title):].lstrip()

        # 按段落/句子等分隔符递归切分，块间无重叠
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
        )

        sub_sections = []
        for index, chunk in enumerate(splitter.split_text(body), start=1):
            text = chunk.strip()
            if not text:
                continue
            # 标题前缀 + 正文
            full_text = (prefix + text).strip()

            # title: 子标题带序号；parent_title: 原章节；part: 第几段
            sub_sections.append(
                {
                    "title": f"{title}-{index}" if title else f"chunk - {index}",
                    "content": full_text,
                    "parent_title": title,
                    "part": index,
                    "file_title": section.get("file_title")
                }
            )
        return sub_sections

    def merge_short_section(self, sections):
        # 过短且同 parent_title 的相邻块合并
        print("node_docment_split:步骤4方法2短合")
        if not sections:
            self.logger.debug("待合并Chunk列表为空，直接返回")
            return []

        merged_sections = []  # 最终合并结果
        current_chunk = None  # 迭代累加器：保存当前待合并的Chunk

        for sec in sections:
            # 初始化：第一个Chunk直接作为当前待合并块
            if current_chunk is None:
                current_chunk = sec
                continue

            # 合并条件：1.当前块长度不足阈值 2.与下一块同父标题（同属一个原章节）
            is_current_short = len(current_chunk["content"]) < self.config.min_content_length
            is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

            if is_current_short and is_same_parent:
                # 合并前清理：去掉下一块开头重复的父标题，避免内容冗余
                parent_title = sec.get("parent_title", "")
                next_content = sec["content"]
                if parent_title and next_content.startswith(parent_title):
                    next_content = next_content[len(parent_title):].lstrip()
                # 合并内容：空行分隔，保证格式整洁
                current_chunk["content"] += "\n\n" + next_content
                # 更新子Chunk序号：保留最新序号，便于溯源
                if "part" in sec:
                    current_chunk["part"] = sec["part"]
                self.logger.debug(
                    f"合并短Chunk：{current_chunk.get('parent_title')} → 累计长度{len(current_chunk['content'])}")
            else:
                # 不满足合并条件：将当前块加入结果，切换为新的待合并块
                merged_sections.append(current_chunk)
                current_chunk = sec

        # 循环结束后，将最后一个待合并块加入结果
        if current_chunk is not None:
            merged_sections.append(current_chunk)

        self.logger.debug(f"短Chunk合并完成：原{len(sections)}个 → 合并后{len(merged_sections)}个")
        return merged_sections

# ==================== 本地调试入口 ====================
if __name__ == "__main__":
    node = NodeDocumentSplit()
    with open(r"D:\output\hak180\hak180_new.md", "r", encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "md_path": r"D:\output\hak180\hak180_new.md",
        "md_content": md_content,
        "file_title": "hak180_new"
    }
    process = node.process(init_state)
    print(f"且分界点执行流程:{process}")
