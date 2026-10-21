import json
import re
from dataclasses import replace
from pathlib import Path
from sys import prefix, path
from typing import List, Dict

import sections
from langchain_text_splitters import RecursiveCharacterTextSplitter

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.import_config import get_config
from processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):
        content, file_title = self._step_1_get_inputs(state)

        sections, title_count, lines_count = self._step_2_split_by_title(content, file_title)

        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)

        sections = self._step_4_refine_chunks(sections)

        for section in sections:
            print(f"{section['title']}")
            print(f"{section['content']}")
            print("===============================================================================")

            print(f"标题数量:{title_count}")
            print(f"行数:{lines_count}")

        self._step_5_print_step(lines_count, sections)

        self._step_6_backup(state, sections)

        state["chunks"] = sections
        return state

    def _step_1_get_inputs(self, state):
        print("node_document_split:步骤1:参数处理")
        content = state.get("md_content")
        if not content:
            raise StateFieldError(field_name="md_content", expected_type=str)

        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="md_content", expected_type=str)

        content = content.replace("\r\n", "\n").replace("\r", "\n")

        return content, file_title

    def _step_2_split_by_title(self, content, file_title):
        print("node_document_split:步骤2:标题切(初切)")
        sections: List[Dict[str, str]] = []
        title_count = 0
        lines = content.split("\n")
        current_lines = []
        in_code_block = False
        current_title = ""

        title_pattern = r"\s*#{1,6}\s+.+"

        def _flush_section():
            if not current_lines:
                return
            sections.append(
                {
                    "title": current_title,
                    "content": "\n".join(current_lines),
                    "parent_title": "",
                    "file_title": file_title
                }
            )

        for line in lines:
            striped_line = line.strip()

            if striped_line.startswith("```") or striped_line.startswith("~~~"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            if (not in_code_block) and (re.match(title_pattern, line)):
                _flush_section()
                current_title = striped_line
                current_lines = [current_title]
                title_count += 1

            else:
                current_lines.append(line)
        _flush_section()
        print(f"标题初切:sections:{sections}")
        return sections, title_count, len(lines)

    def _step_3_handle_no_title(self, content, sections, title_count, file_title):
        print("node_document_split:步骤3:无标题兜底(默认标题)")
        if title_count == 0:
            return [{"title": "无标题", "content": content, "file_title": file_title}]

        return sections

    def _step_4_refine_chunks(self, sections):
        print("node_document_split:步骤4:块精细化处理(长切短合)")

        refind_split = []
        for sec in sections:
            refind_split.extend(self.split_long_section(sec))

        final_sections = self.merge_short_section(refind_split)

        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""

        return final_sections

    def _step_5_print_step(self, lines_count, sections):
        print("node_document_split:步骤5:打印日志")

        chunk_num=len(sections)

        self.logger.info("-"*50+"文档切分统计信息"+"-"*50)
        self.logger.info(f"MD原始文本总行数:{lines_count}")
        self.logger.info(f"最终生成Chunk数量:{chunk_num}")

    def _step_6_backup(self, state, sections):
        print("node_document_split:步骤6:备份")
        path=Path(state.get("md_path")).parent/f'{state.get("file_title")}_chunks.json'
        with open(path,"w",encoding="utf-8") as f:
            json.dump(
                sections,
                f,
                ensure_ascii=False,
                indent=2
            )

    def split_long_section(self, section: Dict[str, str]):
        print("node_document_split:步骤4方法1长切")
        content = section.get("content", "")
        content_len = len(content)

        if content_len <= get_config().max_content_length:
            return [section]

        title = section.get("title")

        prefix = f"{title}\n\n" if title else ""
        available_len = get_config().max_content_length - len(prefix)

        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title):].lstrip()

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
            full_text = (prefix + text).strip()

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
