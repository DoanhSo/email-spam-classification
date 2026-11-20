import importlib.util
from pathlib import Path

import pytest

docx = pytest.importorskip("docx")
pytest.importorskip("pypandoc")

ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_docx", ROOT / "reports/bao-cao/build_docx.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_docx_skeleton(tmp_path):
    out = tmp_path / "bao-cao.docx"
    _load_builder().build(output=out, allow_missing_figures=True)
    document = docx.Document(out)
    text = "\n".join(p.text for p in document.paragraphs)
    for needle in ("Nhóm 12", "2351060429", "2351060435", "2351060446", "2351060467", "HÀ NỘI, 2026",
                   "Hệ thống phát hiện email spam/rác", "MỤC LỤC", "LỜI CẢM ƠN"):
        assert needle in text, needle
    headings = [p.text.strip() for p in document.paragraphs if p.style.name.startswith("Heading")]
    for heading in ("I. Mô tả bài toán", "II. Quy trình khai phá dữ liệu",
                    "III. Phương pháp phân lớp và thuật toán Naive Bayes",
                    "IV. Triển khai thuật toán (Python)", "KẾT LUẬN", "TÀI LIỆU THAM KHẢO"):
        assert heading in headings, heading
    assert 'TOC \\o' in document.element.body.xml


def test_missing_figure_fails_without_flag(tmp_path):
    builder = _load_builder()
    with pytest.raises(FileNotFoundError):
        builder.build(output=tmp_path / "x.docx", allow_missing_figures=False,
                      content_dir=_content_with_missing_image(tmp_path))


def _content_with_missing_image(tmp_path):
    content = tmp_path / "content"
    content.mkdir()
    (content / "01.md").write_text("# I. Test\n\n![Hình 1: thiếu](figures/khong-co.png)\n", encoding="utf-8")
    return content
