"""Build the Word report: cover + MỤC LỤC + chapters (Markdown -> pandoc -> python-docx polish).

Usage: python reports/bao-cao/build_docx.py [--allow-missing-figures] [--output PATH]
"""

import argparse
import re
import tempfile
from pathlib import Path

import pypandoc
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "BaoCao_Nhom12_SpamEmail.docx"
FONT = "Times New Roman"
COVER = [
    ("TRƯỜNG ĐẠI HỌC THỦY LỢI", 14, True), ("KHOA CÔNG NGHỆ THÔNG TIN", 14, True),
    ("", 13, False), ("", 13, False), ("", 13, False),
    ("KHAI PHÁ DỮ LIỆU", 26, True), ("", 13, False),
    ("Đề tài:", 16, True), ("Hệ thống phát hiện email spam/rác", 18, True),
    ("", 13, False), ("", 13, False),
]
TEAM = [
    "Giảng viên: Trần Mạnh Tuấn", "Nhóm sinh viên thực hiện: Nhóm 12 – Lớp 65CNTT",
    "1. Nguyễn Quốc Doanh – 2351060429", "2. Nguyễn Thái Dương – 2351060435",
    "3. Lê Hữu Hiệu – 2351060446", "4. Thân Đức Minh – 2351060467",
]
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")


def _set_font(style, size: float, bold: bool | None = None) -> None:
    style.font.name, style.font.size = FONT, Pt(size)
    style.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        style.font.bold = bold
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for key in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(key), FONT)
    for key in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        fonts.attrib.pop(qn(key), None)


def _field(paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    for kind, text in (("begin", None), (None, instruction), ("separate", None), ("end", None)):
        if kind:
            element = OxmlElement("w:fldChar")
            element.set(qn("w:fldCharType"), kind)
        else:
            element = OxmlElement("w:instrText")
            element.set(qn("xml:space"), "preserve")
            element.text = text
        run._r.append(element)


def make_reference(path: Path) -> None:
    path.write_bytes(_default_reference())
    document = Document(path)
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin, section.right_margin = Cm(3), Cm(2)
    section.top_margin = section.bottom_margin = Cm(2)
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        if name in [s.name for s in document.styles]:
            style = document.styles[name]
            _set_font(style, 13)
            style.paragraph_format.line_spacing = 1.3
            style.paragraph_format.space_after = Pt(4)
            if name != "Compact":
                style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for level, size in ((1, 15), (2, 14), (3, 13), (4, 13)):
        style = document.styles[f"Heading {level}"]
        _set_font(style, size, True)
        style.font.italic = level == 4
        style.paragraph_format.space_before, style.paragraph_format.space_after = Pt(10), Pt(4)
    by_name = {style.name: style for style in document.styles}
    if "Source Code" in by_name:
        code = by_name["Source Code"]
        code.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        code.paragraph_format.line_spacing = 1.0
        code.font.size = Pt(10)
    for name in ("Image Caption", "Table Caption", "Caption"):
        if name in by_name:
            _set_font(by_name[name], 12, False)
            by_name[name].font.italic = True
            by_name[name].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.settings.odd_and_even_pages_header_footer = True
    section.header.paragraphs[0].text = "Nhóm 12"
    section.header.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    section.even_page_header.paragraphs[0].text = "Khai phá dữ liệu"
    for footer in (section.footer, section.even_page_footer):
        paragraph = footer.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _field(paragraph, "PAGE")
    document.save(path)


def _default_reference() -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "reference.docx"
        pypandoc.convert_text("x", "docx", format="md", outputfile=str(target))
        return target.read_bytes()


def prepare_markdown(content_dir: Path, allow_missing_figures: bool) -> str:
    parts, missing = [], []
    for path in sorted(content_dir.glob("*.md")):
        def replace(match: re.Match) -> str:
            caption, target = match.group(1), match.group(2)
            resolved = (HERE / target).resolve()
            if resolved.is_file():
                return f"![{caption}]({resolved.as_posix()})"
            missing.append(target)
            return f"**[Chèn {caption}]**"
        parts.append(IMAGE.sub(replace, path.read_text(encoding="utf-8")))
    if missing and not allow_missing_figures:
        raise FileNotFoundError("Missing figures: " + ", ".join(missing))
    return "\n\n".join(parts)


def _borders(table) -> None:
    properties = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        for key, value in (("w:val", "single"), ("w:sz", "4"), ("w:space", "0"), ("w:color", "000000")):
            element.set(qn(key), value)
        borders.append(element)
    properties.append(borders)


def polish(path: Path) -> None:
    document = Document(path)
    body = document.element.body
    first = body[0]
    inserted = []

    def paragraph(text="", size=13, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, page_break=False):
        p = document.add_paragraph()
        p.alignment = align
        if text:
            run = p.add_run(text)
            run.font.size, run.font.bold, run.font.name = Pt(size), bold, FONT
        if page_break:
            p.add_run().add_break(WD_BREAK.PAGE)
        inserted.append(p._p)
        return p

    for text, size, bold in COVER:
        paragraph(text, size, bold)
    for line in TEAM:
        p = paragraph(line, 14, line.startswith(("Giảng", "Nhóm")), WD_ALIGN_PARAGRAPH.LEFT)
        p.paragraph_format.left_indent = Cm(3)
    for _ in range(4):
        paragraph()
    paragraph("HÀ NỘI, 2026", 14, True, page_break=True)
    paragraph("MỤC LỤC", 15, True)
    toc = paragraph(align=WD_ALIGN_PARAGRAPH.LEFT)
    _field(toc, 'TOC \\o "1-3" \\h \\z \\u')
    paragraph(page_break=True)
    for element in inserted:
        first.addprevious(element)
    for table in document.tables:
        _borders(table)
        for row in table.rows:
            for cell in row.cells:
                for cell_paragraph in cell.paragraphs:
                    cell_paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    document.sections[0].different_first_page_header_footer = True
    document.save(path)


def build(output: Path = DEFAULT_OUTPUT, allow_missing_figures: bool = False, content_dir: Path = HERE / "content") -> Path:
    markdown = prepare_markdown(Path(content_dir), allow_missing_figures)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        reference = Path(tmp) / "reference.docx"
        make_reference(reference)
        pypandoc.convert_text(markdown, "docx", format="markdown", outputfile=str(output),
                              extra_args=["--reference-doc", str(reference)])
    polish(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-figures", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.output, args.allow_missing_figures))
