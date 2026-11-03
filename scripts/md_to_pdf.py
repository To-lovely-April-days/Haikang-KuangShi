"""把 docs 下的 Markdown 手册导出为 PDF（给前端 / 现场发文件用）。

依赖不在 requirements 里，单独装一次即可（任意 Python ≥ 3.10 环境）：
    uv pip install markdown fpdf2
字体用 macOS 自带的冬青黑体（正文）与 Menlo（代码）；其它系统改 ``FONTS`` 里的路径。

用法：
    python scripts/md_to_pdf.py docs/06_前端接口调用手册.md            # 同目录生成同名 .pdf
    python scripts/md_to_pdf.py docs/06_前端接口调用手册.md out.pdf
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown
from fontTools.ttLib import TTCollection
from fpdf import FPDF
from fpdf.fonts import TextStyle

FONTS = {
    "body": ("/System/Library/Fonts/Hiragino Sans GB.ttc", "W3"),
    "body_bold": ("/System/Library/Fonts/Hiragino Sans GB.ttc", "W6"),
    "mono": ("/System/Library/Fonts/Menlo.ttc", "Menlo Regular"),
    "mono_bold": ("/System/Library/Fonts/Menlo.ttc", "Menlo Bold"),
    "fallback": ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", None),
}
PAGE_MARGIN_MM = 16.0
CONTENT_WIDTH_MM = 210.0 - 2 * PAGE_MARGIN_MM
ASCII_CHAR_MM = 1.9  # 9.5 pt 正文一个 ASCII 字符的大致宽度，用于估算表格列宽下限


def font_index(path: str, name: str | None) -> int:
    """TTC 字体集合里按名字找子字体序号；单个 TTF 返回 0。"""
    if name is None or not path.endswith(".ttc"):
        return 0
    names = [f["name"].getDebugName(4) or "" for f in TTCollection(path).fonts]
    for i, n in enumerate(names):
        if name.lower() in n.lower():
            return i
    raise SystemExit(f"{path}: 找不到字体 {name!r}，可选 {names}")


def visual_len(text: str) -> int:
    """中文按 2、其它按 1 估算显示宽度。"""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def add_col_widths(html: str) -> str:
    """给每个 <th> 加 width=%：先保证最长不可断 ASCII 词放得下，剩余按内容长度分配。"""

    def fix(m: re.Match[str]) -> str:
        table = m.group(0)
        want: list[int] = []
        token: list[int] = []
        for row in re.findall(r"<tr>(.*?)</tr>", table, flags=re.S):
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=re.S)
            for i, cell in enumerate(cells):
                while len(want) <= i:
                    want.append(0)
                    token.append(0)
                plain = re.sub(r"<[^>]+>", "", cell)
                want[i] = max(want[i], min(visual_len(plain), 80))
                for tok in re.findall(r"[\x21-\x7e]+", plain):
                    token[i] = max(token[i], len(tok))
        if not want:
            return table
        floor = [
            max(6.0, min(45.0, (tk * ASCII_CHAR_MM + 4) / CONTENT_WIDTH_MM * 100)) for tk in token
        ]
        rest = max(100.0 - sum(floor), 0.0)
        total = sum(want) or 1
        pct = [f + rest * w / total for f, w in zip(floor, want, strict=True)]
        scale = 100.0 / sum(pct)
        pct_int = [round(x * scale) for x in pct]
        pct_int[-1] = 100 - sum(pct_int[:-1])
        it = iter(pct_int)
        table = re.sub(r"<th>", lambda _m: f'<th width="{next(it)}%">', table)
        return table.replace("<table>", '<table width="100%">', 1)

    return re.sub(r"<table>.*?</table>", fix, html, flags=re.S)


def flatten_cells(html: str) -> str:
    """fpdf2 的表格单元格不支持嵌套行内标签：去掉 <code>/<strong>/<em>/<a>，并左对齐。"""

    def fix(m: re.Match[str]) -> str:
        open_tag = m.group(1).replace("<td", '<td align="left"', 1)
        inner = re.sub(r"</?(code|strong|em|a)\b[^>]*>", "", m.group(2))
        return f"{open_tag}{inner}{m.group(3)}"

    return re.sub(r"(<t[hd][^>]*>)(.*?)(</t[hd]>)", fix, html, flags=re.S)


class ManualPDF(FPDF):
    """带页脚的 A4 文档。"""

    footer_text = ""

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Body", size=8)
        self.set_text_color(120)
        self.cell(0, 6, f"{self.footer_text}　·　第 {self.page_no()} / {{nb}} 页", align="C")


def heading(size_pt: float, top_mm: float, bottom_mm: float) -> TextStyle:
    return TextStyle(
        font_family="Body",
        font_style="B",
        font_size_pt=size_pt,
        color=(20, 20, 20),
        t_margin=top_mm,
        b_margin=bottom_mm,
    )


def convert(src: Path, out: Path) -> int:
    """Markdown → PDF，返回页数。"""
    md_text = src.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", md_text, flags=re.M)
    title = title_match.group(1).strip() if title_match else src.stem
    html = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    html = flatten_cells(add_col_widths(html))

    pdf = ManualPDF(orientation="P", unit="mm", format="A4")
    pdf.footer_text = title
    pdf.set_margins(PAGE_MARGIN_MM, PAGE_MARGIN_MM, PAGE_MARGIN_MM)
    pdf.set_auto_page_break(auto=True, margin=18)
    body, body_name = FONTS["body"]
    bold, bold_name = FONTS["body_bold"]
    mono, mono_name = FONTS["mono"]
    mono_b, mono_b_name = FONTS["mono_bold"]
    fb, _ = FONTS["fallback"]
    pdf.add_font("Body", "", body, collection_font_number=font_index(body, body_name))
    pdf.add_font("Body", "B", bold, collection_font_number=font_index(bold, bold_name))
    pdf.add_font("Mono", "", mono, collection_font_number=font_index(mono, mono_name))
    pdf.add_font("Mono", "B", mono_b, collection_font_number=font_index(mono_b, mono_b_name))
    pdf.add_font("Fallback", "", fb)
    pdf.set_fallback_fonts(["Body", "Fallback"])
    pdf.set_title(title)
    pdf.add_page()
    pdf.set_font("Body", size=9.5)
    pdf.write_html(
        html,
        font_family="Body",
        table_line_separators=True,
        tag_styles={
            "h1": heading(18, 0, 1),
            "h2": heading(14.5, 4, 0.5),
            "h3": heading(12, 3, 0.5),
            "h4": heading(10.5, 2.5, 0.2),
            "pre": TextStyle(font_family="Mono", font_size_pt=8),
            "code": TextStyle(font_family="Mono", font_size_pt=8.5, color=(140, 20, 60)),
            "a": TextStyle(color=(0, 70, 160), underline=False),
        },
    )
    pdf.output(str(out))
    return pdf.page_no()


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    src = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 else src.with_suffix(".pdf")
    pages = convert(src, out)
    print(f"已写入 {out}（{pages} 页）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
