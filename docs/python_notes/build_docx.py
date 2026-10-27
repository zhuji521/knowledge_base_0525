# -*- coding: utf-8 -*-
"""把 docs/python_notes/part*.md 合成一份排版好的 Word 文档。"""
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BASE = Path(r"D:\project\knowledge_base_0525\docs\python_notes")
OUT_DOCX = Path(r"D:\project\knowledge_base_0525\docs\Python知识整理_导入流水线七节点.docx")
OUT_MD = Path(r"D:\project\knowledge_base_0525\docs\Python知识整理_导入流水线七节点.md")

CHINESE_BODY = "宋体"
CHINESE_HEAD = "微软雅黑"
CODE_FONT = "Consolas"
CODE_CN = "等线"

DARK = RGBColor(0x1F, 0x38, 0x64)
MID = RGBColor(0x2E, 0x5C, 0x8A)
GREY = RGBColor(0x59, 0x59, 0x59)
CODE_COLOR = RGBColor(0x9C, 0x27, 0x1B)


# ---------------------------------------------------------------- 解析 markdown
def parse_blocks(text):
    blocks = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            while code and not code[0].strip():
                code.pop(0)
            while code and not code[-1].strip():
                code.pop()
            blocks.append(("code", "\n".join(code)))
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            rows = []
            header = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            blocks.append(("table", (header, rows)))
            continue

        if stripped.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append(("callout", "\n".join(quote)))
            continue

        if stripped.startswith("### "):
            blocks.append(("h3", stripped[4:].strip()))
            i += 1
            continue
        if stripped.startswith("## "):
            blocks.append(("h2", stripped[3:].strip()))
            i += 1
            continue
        if stripped.startswith("# "):
            blocks.append(("h1", stripped[2:].strip()))
            i += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
            blocks.append(("number", (m.group(1), m.group(2))))
            i += 1
            continue

        if stripped.startswith("- "):
            blocks.append(("bullet", stripped[2:].strip()))
            i += 1
            continue

        if stripped in ("---", "***", "___"):
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        blocks.append(("p", stripped))
        i += 1
    return blocks


# ---------------------------------------------------------------- 字体工具
def set_run_fonts(run, ascii_font, cn_font, size=None, bold=None, color=None):
    run.font.name = ascii_font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn("w:" + attr), ascii_font if attr in ("ascii", "hAnsi", "cs") else cn_font)
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        if rfonts.get(qn("w:" + attr)) is not None:
            del rfonts.attrib[qn("w:" + attr)]
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def set_style_font(style, ascii_font, cn_font, size=None, bold=None, color=None):
    style.font.name = ascii_font
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn("w:" + attr), ascii_font if attr in ("ascii", "hAnsi", "cs") else cn_font)
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        if rfonts.get(qn("w:" + attr)) is not None:
            del rfonts.attrib[qn("w:" + attr)]
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color


PPR_AFTER_SHD = (
    "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap", "w:overflowPunct",
    "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN", "w:bidi", "w:adjustRightInd",
    "w:snapToGrid", "w:spacing", "w:ind", "w:contextualSpacing", "w:mirrorIndents",
    "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment", "w:textboxTightWrap",
    "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr", "w:pPrChange",
)


def shade(element_pr, fill, paragraph_level=True):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    if paragraph_level:
        element_pr.insert_element_before(shd, *PPR_AFTER_SHD)
    else:
        element_pr.append(shd)


def para_border(paragraph, side="bottom", color="D9D9D9", size=6, space=4):
    ppr = paragraph._p.get_or_add_pPr()
    pbdr = ppr.find(qn("w:pBdr"))
    if pbdr is None:
        pbdr = OxmlElement("w:pBdr")
        ppr.insert_element_before(pbdr, "w:shd", *PPR_AFTER_SHD)
    el = OxmlElement("w:" + side)
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(size))
    el.set(qn("w:space"), str(space))
    el.set(qn("w:color"), color)
    pbdr.append(el)


INLINE_RE = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`")


def add_inline(paragraph, text, size=10.5, color=None, bold_all=False):
    pos = 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            run = paragraph.add_run(text[pos:m.start()])
            set_run_fonts(run, "Calibri", CHINESE_BODY, size=size, bold=True if bold_all else None, color=color)
        if m.group(1) is not None:
            run = paragraph.add_run(m.group(1))
            set_run_fonts(run, "Calibri", CHINESE_BODY, size=size, bold=True, color=color or DARK)
        else:
            run = paragraph.add_run(m.group(2))
            set_run_fonts(run, CODE_FONT, CODE_CN, size=size - 1, bold=False, color=CODE_COLOR)
        pos = m.end()
    if pos < len(text):
        run = paragraph.add_run(text[pos:])
        set_run_fonts(run, "Calibri", CHINESE_BODY, size=size, bold=True if bold_all else None, color=color)


# ---------------------------------------------------------------- 页面元素
def add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("第 ")
    set_run_fonts(run, "Calibri", CHINESE_BODY, size=9, color=GREY)
    _add_field(p, "PAGE")
    run = p.add_run(" 页")
    set_run_fonts(run, "Calibri", CHINESE_BODY, size=9, color=GREY)


def _add_field(paragraph, field):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = field
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    txt = OxmlElement("w:t")
    txt.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(txt)
    run._r.append(fld_end)


def add_header(section, text):
    header = section.header
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run(text)
    set_run_fonts(run, "Calibri", CHINESE_BODY, size=8.5, color=GREY)
    para_border(p, "bottom", "D9D9D9", 4, 2)


def add_cover(doc):
    for _ in range(3):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)

    p = doc.add_paragraph()
    run = p.add_run("技 术 学 习 笔 记")
    set_run_fonts(run, "Calibri", CHINESE_HEAD, size=11, color=MID)
    p.paragraph_format.space_after = Pt(18)

    p = doc.add_paragraph()
    run = p.add_run("知识库导入流水线")
    set_run_fonts(run, "Calibri", CHINESE_HEAD, size=30, bold=True, color=DARK)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.1

    p = doc.add_paragraph()
    run = p.add_run("Python 知识整理")
    set_run_fonts(run, "Calibri", CHINESE_HEAD, size=30, bold=True, color=DARK)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.line_spacing = 1.1

    p = doc.add_paragraph()
    run = p.add_run("按 a → g 七个节点的代码顺序，梳理一条 RAG 导入链路上的 Python 知识点")
    set_run_fonts(run, "Calibri", CHINESE_BODY, size=12, color=GREY)
    p.paragraph_format.space_after = Pt(18)
    para_border(p, "bottom", "2E5C8A", 12, 8)

    meta = [
        ("覆盖文件", "a_node_entry.py、b_node_pdf_to_md.py、c_node_md_img.py、d_node_document_split.py、\ne_node_item_name_recognition.py、f_node_bge_embedding.py、g_node_import_milvus.py"),
        ("内容范围", "语法基础 · 文件与路径 · 字符串与正则 · 网络请求 · 面向对象与类型注解 · 向量库与对象存储"),
        ("适用版本", "Python 3.9+（g 节点使用 PEP 585 内置泛型）"),
        ("整理日期", "2026 年 10 月 27 日"),
    ]
    for label, value in meta:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(7)
        p.paragraph_format.line_spacing = 1.25
        run = p.add_run(label + "　")
        set_run_fonts(run, "Calibri", CHINESE_HEAD, size=10, bold=True, color=MID)
        run = p.add_run(value)
        set_run_fonts(run, "Calibri", CHINESE_BODY, size=10, color=GREY)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(22)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("摘要")
    set_run_fonts(run, "Calibri", CHINESE_HEAD, size=10, bold=True, color=MID)
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.4
    run = p.add_run(
        "本文档把七个节点的源码拆成 10 章：第 1—7 章按代码顺序逐节点讲解，第 8 章按主题横向重排，"
        "第 9 章汇总代码中的问题与改进建议，第 10 章给出学习路径与自测题。每个知识点都标注了"
        "所在文件与行号，便于对照源码阅读。"
    )
    set_run_fonts(run, "Calibri", CHINESE_BODY, size=10.5, color=RGBColor(0x33, 0x33, 0x33))


def add_toc(doc, headings):
    p = doc.add_paragraph()
    run = p.add_run("目　录")
    set_run_fonts(run, "Calibri", CHINESE_HEAD, size=18, bold=True, color=DARK)
    p.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph()
    run = p.add_run("（结构大纲，按章节顺序排列；正文页码见页脚）")
    set_run_fonts(run, "Calibri", CHINESE_BODY, size=9, color=GREY)
    p.paragraph_format.space_after = Pt(10)
    para_border(p, "bottom", "D9D9D9", 6, 6)

    for level, text in headings:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        if level == 1:
            pf.left_indent = Cm(0.0)
            pf.space_before = Pt(4)
            pf.space_after = Pt(0)
            pf.line_spacing = Pt(13.2)
            run = p.add_run(text)
            set_run_fonts(run, "Calibri", CHINESE_HEAD, size=10.5, bold=True, color=DARK)
        else:
            pf.left_indent = Cm(0.8)
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
            pf.line_spacing = Pt(11.5)
            run = p.add_run(text)
            set_run_fonts(run, "Calibri", CHINESE_BODY, size=9.2, color=RGBColor(0x33, 0x33, 0x33))

# ---------------------------------------------------------------- 内容元素
def add_h1(doc, text):
    p = doc.add_paragraph(style="Heading 1")
    p.paragraph_format.page_break_before = True
    run = p.add_run(text)
    set_run_fonts(run, "Calibri", CHINESE_HEAD, size=19, bold=True, color=DARK)
    para_border(p, "bottom", "2E5C8A", 10, 6)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(14)
    p.paragraph_format.keep_with_next = True


def add_h2(doc, text):
    p = doc.add_paragraph(style="Heading 2")
    run = p.add_run(text)
    set_run_fonts(run, "Calibri", CHINESE_HEAD, size=14, bold=True, color=MID)
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True


def add_h3(doc, text):
    p = doc.add_paragraph(style="Heading 3")
    run = p.add_run(text)
    set_run_fonts(run, "Calibri", CHINESE_HEAD, size=11.5, bold=True, color=RGBColor(0x26, 0x26, 0x26))
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True


def add_paragraph(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.38
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_inline(p, text)


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.32
    add_inline(p, text)


def add_number(doc, num, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(0.95)
    pf.first_line_indent = Cm(-0.55)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.32
    run = p.add_run(num + ". ")
    set_run_fonts(run, "Calibri", CHINESE_BODY, size=10.5, bold=True, color=MID)
    add_inline(p, text)


def add_code(doc, code):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(0.2)
    pf.right_indent = Cm(0.1)
    pf.space_before = Pt(4)
    pf.space_after = Pt(8)
    pf.line_spacing = 1.05
    pf.keep_together = len(code.split("\n")) <= 22
    ppr = p._p.get_or_add_pPr()
    shade(ppr, "F6F8FA")
    para_border(p, "left", "4A7EBB", 18, 6)
    longest = max(_effective_len(line) for line in code.split("\n"))
    size = 8.3 if longest <= 92 else (8.0 if longest <= 104 else 7.6)
    run = p.add_run(code)
    set_run_fonts(run, CODE_FONT, CODE_CN, size=size, color=RGBColor(0x1F, 0x1F, 0x1F))


def add_callout(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(0.25)
    pf.right_indent = Cm(0.1)
    pf.space_before = Pt(6)
    pf.space_after = Pt(10)
    pf.line_spacing = 1.35
    ppr = p._p.get_or_add_pPr()
    shade(ppr, "EAF1FB")
    para_border(p, "left", "2E75B6", 18, 8)
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if idx:
            p.add_run().add_break()
        add_inline(p, line, size=9.8, color=RGBColor(0x1A, 0x33, 0x55))


def _set_cell_margins(table, top=30, bottom=30, left=55, right=55):
    tbl_pr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side, val in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement("w:" + side)
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tbl_pr.insert_element_before(mar, "w:tblLook", "w:tblCaption", "w:tblDescription")


def _effective_len(text):
    return sum(2 if ord(ch) > 127 else 1 for ch in text)


def _fit_table(table, header, rows, total_cm=16.75):
    n = len(header)
    weights = []
    for i in range(n):
        values = [header[i]] + [r[i] for r in rows if i < len(r)]
        avg = sum(_effective_len(v) for v in values) / max(1, len(values))
        weights.append(min(max(avg, 5.0), 55.0))
    total = sum(weights)
    widths = [total_cm * w / total for w in weights]
    widths = [max(1.5, w) for w in widths]
    scale = total_cm / sum(widths)
    widths = [w * scale for w in widths]

    table.autofit = False
    for idx, width in enumerate(widths):
        table.columns[idx].width = Cm(width)
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            if idx < n:
                cell.width = Cm(widths[idx])


def add_table(doc, header, rows):
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    _set_cell_margins(table)
    tbl_size = 9 if len(header) <= 4 else 8.6

    hdr = table.rows[0]
    tr_pr = hdr._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.insert_element_before(cant_split, "w:trHeight", "w:tblHeader", "w:tblCellSpacing", "w:jc", "w:hidden")
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)
    for idx, text in enumerate(header):
        cell = hdr.cells[idx]
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.1
        shade(cell._tc.get_or_add_tcPr(), "DCE6F1", paragraph_level=False)
        add_inline(p, text, size=tbl_size, bold_all=True)

    for row in rows:
        new_row = table.add_row()
        tr_pr = new_row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.insert_element_before(cant_split, "w:trHeight", "w:tblHeader", "w:tblCellSpacing", "w:jc", "w:hidden")
        cells = new_row.cells
        for idx, text in enumerate(row[: len(header)]):
            cell = cells[idx]
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.2
            add_inline(p, text, size=tbl_size)

    _fit_table(table, header, rows)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ---------------------------------------------------------------- 主流程
def main():
    merged = "\n".join((BASE / f"part{i}.md").read_text(encoding="utf-8") for i in (1, 2, 3))
    OUT_MD.write_text(merged, encoding="utf-8")

    # 校验：每行 ** 与反引号数量应为偶数，避免格式串行
    problems = []
    for no, line in enumerate(merged.split("\n"), 1):
        if line.strip().startswith("```"):
            continue
        if line.count("**") % 2:
            problems.append((no, "**", line[:60]))
        if line.count("`") % 2:
            problems.append((no, "`", line[:60]))
    if problems:
        print("格式警告：")
        for item in problems[:20]:
            print("  行", item[0], item[1], item[2])

    blocks = parse_blocks(merged)
    headings = [(1 if kind == "h1" else 2, text) for kind, text in blocks if kind in ("h1", "h2")]

    doc = Document()
    doc.core_properties.title = "知识库导入流水线：Python 知识整理"
    doc.core_properties.subject = "按 a-g 七个节点梳理 Python 知识点"
    doc.core_properties.author = "学习笔记"
    doc.core_properties.comments = "配套源码：processor/import_processor/nodes/*.py"

    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.0)
    section.header_distance = Cm(1.1)
    section.footer_distance = Cm(1.1)
    section.different_first_page_header_footer = True

    normal = doc.styles["Normal"]
    set_style_font(normal, "Calibri", CHINESE_BODY, size=10.5, color=RGBColor(0x1A, 0x1A, 0x1A))
    normal.paragraph_format.line_spacing = 1.35
    normal.paragraph_format.space_after = Pt(5)

    for name, size, color in (("Heading 1", 19, DARK), ("Heading 2", 14, MID), ("Heading 3", 11.5, RGBColor(0x26, 0x26, 0x26))):
        set_style_font(doc.styles[name], "Calibri", CHINESE_HEAD, size=size, bold=True, color=color)

    add_cover(doc)
    doc.add_page_break()
    add_toc(doc, headings)

    add_header(section, "知识库导入流水线 · Python 知识整理")
    add_page_number_footer(section)

    for kind, payload in blocks:
        if kind == "h1":
            add_h1(doc, payload)
        elif kind == "h2":
            add_h2(doc, payload)
        elif kind == "h3":
            add_h3(doc, payload)
        elif kind == "p":
            add_paragraph(doc, payload)
        elif kind == "bullet":
            add_bullet(doc, payload)
        elif kind == "number":
            add_number(doc, payload[0], payload[1])
        elif kind == "code":
            add_code(doc, payload)
        elif kind == "callout":
            add_callout(doc, payload)
        elif kind == "table":
            add_table(doc, payload[0], payload[1])

    doc.save(OUT_DOCX)
    print("已生成：", OUT_DOCX)
    print("段落数：", len(doc.paragraphs), "表格数：", len(doc.tables), "标题数：", len(headings))


if __name__ == "__main__":
    main()
