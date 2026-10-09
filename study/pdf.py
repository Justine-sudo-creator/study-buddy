"""Reviewer PDF, laid out from a structured StudyGuide (HTML -> PDF via xhtml2pdf).

Layout:
  cover block (title, subject, big picture, learning goals)
  1 Key Concepts        — one card per concept: explanation, why it matters, example, watch-out
  2 How It Connects
  3 Formulas            — (if any)
  4 Worked Examples     — numbered steps with the why
  5 Figures from Your Notes
  6 Common Misconceptions — wrong vs right
  7 Test Yourself       — questions grouped by level (answers at the back)
  8 Cheat Sheet
  9 Not Covered in Your Notes
  Answer Key
"""

from __future__ import annotations

import base64
import html as _html
import io
import re
from pathlib import Path

from .engines import StudyGuide
from .extract import NoteImage

_FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
BLUE, LIGHT, INK, MUTED = "#1d4ed8", "#eff6ff", "#1f2937", "#6b7280"


def _register_fonts() -> tuple[str, str]:
    """Register DejaVu (Unicode: — → Σ ≤ ²) with reportlab + xhtml2pdf's font table.

    xhtml2pdf refuses @font-face files outside the workspace, so register the TTFs
    directly and refer to them by name in CSS. Falls back to Helvetica/Courier.
    """
    try:
        from reportlab.lib.fonts import addMapping
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from xhtml2pdf.default import DEFAULT_FONT

        if "DejaVuSans" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("DejaVuSans", str(_FONT_DIR / "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(_FONT_DIR / "DejaVuSans-Bold.ttf")))
            pdfmetrics.registerFont(TTFont("DejaVuSansMono", str(_FONT_DIR / "DejaVuSansMono.ttf")))
            for bold in (0, 1):
                for italic in (0, 1):
                    addMapping("DejaVuSans", bold, italic, "DejaVuSans-Bold" if bold else "DejaVuSans")
                    addMapping("DejaVuSansMono", bold, italic, "DejaVuSansMono")
        DEFAULT_FONT["dejavusans"] = "DejaVuSans"
        DEFAULT_FONT["dejavusansmono"] = "DejaVuSansMono"
        return "DejaVuSans", "DejaVuSansMono"
    except Exception:  # noqa: BLE001
        return "Helvetica", "Courier"


CSS = f"""
@page {{
  size: a4; margin: 1.8cm 1.7cm 2cm 1.7cm;
  @frame footer {{ -pdf-frame-content: footer; bottom: 0.7cm; margin-left: 1.7cm; margin-right: 1.7cm; height: 0.8cm; }}
}}
body {{ font-family: BODYFONT; font-size: 10.5pt; line-height: 1.5; color: {INK}; }}
h1 {{ font-size: 22pt; color: {BLUE}; margin: 0 0 2px 0; }}
.subject {{ font-size: 11pt; color: {MUTED}; margin-bottom: 10px; }}
h2 {{ -pdf-keep-with-next: true; font-size: 14pt; color: #ffffff; background-color: {BLUE}; padding: 5px 8px; margin: 16px 0 8px 0; }}
h3 {{ font-size: 12pt; color: {BLUE}; margin: 4px 0 4px 0; }}
p {{ margin: 3px 0; }}
li {{ margin: 2px 0; }}
code {{ font-family: MONOFONT; font-size: 9.5pt; background-color: #f3f4f6; }}
.box {{ background-color: {LIGHT}; border-left: 4px solid {BLUE}; padding: 8px 10px; margin: 6px 0 10px 0; }}
.card {{ border: 1px solid #d1d5db; margin: 0 0 10px 0; }}
.card td {{ padding: 8px 10px; vertical-align: top; }}
.label {{ font-weight: bold; color: {BLUE}; }}
.warn {{ background-color: #fff7ed; border-left: 4px solid #ea580c; padding: 5px 8px; margin: 4px 0; }}
.ex {{ background-color: #f9fafb; border-left: 4px solid #9ca3af; padding: 5px 8px; margin: 4px 0; }}
.src {{ color: {MUTED}; font-size: 8.5pt; }}
.wrong {{ background-color: #fef2f2; border-left: 4px solid #dc2626; padding: 5px 8px; }}
.right {{ background-color: #f0fdf4; border-left: 4px solid #16a34a; padding: 5px 8px; }}
.formula {{ font-family: MONOFONT; font-size: 11pt; background-color: #f3f4f6; padding: 6px 8px; margin: 4px 0; }}
.lvl {{ font-size: 11pt; color: {INK}; font-weight: bold; margin: 8px 0 2px 0; }}
.fig {{ margin: 8px 0 12px 0; text-align: center; }}
.figcap {{ color: {INK}; font-size: 9.5pt; text-align: left; }}
.meta {{ color: {MUTED}; font-size: 8.5pt; }}
#footer {{ text-align: center; color: #9ca3af; font-size: 8pt; }}
"""


# Unicode subscript/superscript maps
_SUB_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢₖₗₘₙₒₚᵣₛₜₓ",
                         "0123456789aehiklmnoprstx")
_SUP_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵉᵏˡᵐⁿᵒᵖˢᵗᵘʷˣᶻ",
                         "0123456789+-=()aeklmnopstuwxz")

_LATEX_MAP = {
    r"\sum": "Σ", r"\times": "×", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\le": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\infty": "∞", r"\alpha": "α", r"\beta": "β", r"\gamma": "γ",
    r"\Delta": "Δ", r"\pi": "π", r"\mu": "μ", r"\sigma": "σ",
    r"\Sigma": "Σ", r"\Omega": "Ω", r"\rightarrow": "→",
    r"\in": "∈", r"\notin": "∉", r"\subset": "⊂", r"\cup": "∪",
    r"\cap": "∩", r"\emptyset": "∅", r"\sqrt": "√", r"\int": "∫",
}


def _render_math(s: str) -> str:
    """Convert Unicode subscripts/superscripts to HTML tags and strip common LaTeX."""
    for cmd, replacement in _LATEX_MAP.items():
        s = s.replace(cmd, replacement)
    s = re.sub(r"\\[a-zA-Z]+\{?", "", s)
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ord(ch) in _SUB_MAP:
            j = i
            while j < len(s) and ord(s[j]) in _SUB_MAP:
                j += 1
            if j > i:
                out.append(f"<sub>{s[i:j].translate(_SUB_MAP)}</sub>")
                i = j
                continue
        if ord(ch) in _SUP_MAP:
            j = i
            while j < len(s) and ord(s[j]) in _SUP_MAP:
                j += 1
            if j > i:
                out.append(f"<sup>{s[i:j].translate(_SUP_MAP)}</sup>")
                i = j
                continue
        out.append(ch)
        i += 1
    result = "".join(out)
    # Caret exponents on the escaped output: 2^n → 2<sup>n</sup>
    result = re.sub(r"(\w+)\s*\^\s*(\w+)", r"\1<sup>\2</sup>", result)
    # Underscore subscripts on the escaped output: H_2O → H<sub>2</sub>O, x_i → x<sub>i</sub>
    result = re.sub(r"([a-zA-Z]+)_(\d+|[a-zA-Z])", r"\1<sub>\2</sub>", result)
    return result


def _md_inline(s: str) -> str:
    """Escape, then allow **bold** and `code` (the only Markdown the model may use)."""
    s = _render_math(s)
    tokens = re.split(r'(</?\w+[^>]*>)', s)
    for i, tok in enumerate(tokens):
        if not tok.startswith('<'):
            tokens[i] = _html.escape(tok, quote=False)
    s = ''.join(tokens)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s.replace("\n", "<br/>")


def _interleave(practice):
    """Round-robin across levels so questions are mixed, not grouped."""
    by_level = {"Remember": [], "Apply": [], "Analyze": []}
    for p in practice:
        by_level[p.level].append(p)
    result, max_len = [], max(len(v) for v in by_level.values())
    for i in range(max_len):
        for level in ("Remember", "Apply", "Analyze"):
            if i < len(by_level[level]):
                result.append(by_level[level][i])
    return result


def _src(s: str) -> str:
    return f" <span class='src'>({_html.escape(s)})</span>" if s else ""


def _img_tag(im: NoteImage, max_w: int = 600, max_h: int = 400) -> str:
    from PIL import Image

    try:
        w, h = Image.open(io.BytesIO(im.data)).size
    except Exception:  # noqa: BLE001
        return ""
    scale = min(max_w / w, max_h / h, 1.0)
    b64 = base64.b64encode(im.data).decode()
    return f"<img src='data:{im.mime};base64,{b64}' width='{int(w * scale)}' height='{int(h * scale)}'/>"


def _card(inner: str) -> str:
    return f"<table class='card' repeat='0'><tr><td>{inner}</td></tr></table>"


def guide_to_html(g: StudyGuide, images: dict[str, NoteImage], meta: str) -> str:
    out: list[str] = [f"<h1>{_html.escape(g.title)}</h1>"]
    if g.subject:
        out.append(f"<div class='subject'>{_html.escape(g.subject)}</div>")
    out.append(f"<div class='box'><span class='label'>The big picture</span><br/>{_md_inline(g.big_picture)}</div>")
    if g.learning_goals:
        out.append("<p class='label'>After studying this, you should be able to:</p><ul>")
        out += [f"<li>{_md_inline(x)}</li>" for x in g.learning_goals]
        out.append("</ul>")

    n = 0

    def section(title: str) -> str:
        nonlocal n
        n += 1
        return f"<h2>{n}. {_html.escape(title)}</h2>"

    if g.concepts:
        out.append(section("Key Concepts"))
        for c in g.concepts:
            parts = [f"<h3>{_html.escape(c.name)}{_src(c.source)}</h3>", f"<p>{_md_inline(c.explanation)}</p>"]
            if c.why_it_matters:
                parts.append(f"<p><span class='label'>Why it matters:</span> {_md_inline(c.why_it_matters)}</p>")
            if c.example:
                parts.append(f"<div class='ex'><b>Example from your notes:</b> {_md_inline(c.example)}</div>")
            if c.watch_out:
                parts.append(f"<div class='warn'><b>Watch out:</b> {_md_inline(c.watch_out)}</div>")
            if c.connects_to:
                parts.append(f"<p class='src'>Related: {_html.escape(', '.join(c.connects_to))}</p>")
            out.append(_card("".join(parts)))

    if g.key_terms:
        out.append(section("Key Terms"))
        for t in g.key_terms:
            out.append(_card(f"<h3>{_html.escape(t.term)}{_src(t.source)}</h3>"
                             f"<p>{_md_inline(t.definition)}</p>"))
    if g.connections:
        out.append(section("How It All Connects"))
        out.append("<ul>" + "".join(f"<li>{_md_inline(x)}</li>" for x in g.connections) + "</ul>")

    if g.formulas:
        out.append(section("Formulas"))
        for f in g.formulas:
            out.append(_card(f"<h3>{_html.escape(f.name)}{_src(f.source)}</h3>"
                             f"<div class='formula'>{_render_math(f.formula)}</div>"
                             f"<p>{_render_math(f.meaning)}</p>"
                             f"<p><span class='label'>Use it when:</span> {_render_math(f.when_to_use)}</p>"))

    if g.worked_examples:
        out.append(section("Worked Examples"))
        for w in g.worked_examples:
            parts = [f"<h3>{_html.escape(w.title)}{_src(w.source)}</h3>",
                     f"<p><b>Problem:</b> {_md_inline(w.problem)}</p>"]
            parts += [f"<p><b>Step {k}.</b> {_md_inline(st)}</p>" for k, st in enumerate(w.steps, 1)]
            parts.append(f"<p><b>Answer:</b> {_md_inline(w.answer)}</p>")
            if w.takeaway:
                parts.append(f"<div class='box'><b>Takeaway:</b> {_md_inline(w.takeaway)}</div>")
            out.append(_card("".join(parts)))

    figs = [(images[f.label], f) for f in g.figures if f.label in images]
    if figs:
        out.append(section("Figures from Your Notes"))
        for im, fig in figs:
            tag = _img_tag(im)
            if tag:
                out.append(f"<div class='fig'>{tag}</div>")
                q = fig.question
                if q:
                    out.append(f"<p><b>Look at the figure — can you answer?</b> {_md_inline(q)}</p>")
                    out.append(f"<div class='ex'><b>Answer:</b> {_md_inline(fig.insight)}</div>")
                else:
                    out.append(f"<div class='figcap'><b>{_html.escape(im.label)}.</b> "
                               f"{_md_inline(fig.insight)}</div>")

    if g.misconceptions:
        out.append(section("Common Misconceptions"))
        for m in g.misconceptions:
            out.append(_card(f"<div class='wrong'><b>✗ Myth:</b> {_md_inline(m.wrong)}</div>"
                             f"<div class='right'><b>✓ Actually:</b> {_md_inline(m.right)}</div>"
                             f"<p>{_md_inline(m.why)}</p>"))

    if g.practice:
        out.append(section("Test Yourself"))
        out.append("<p class='src'>Try each question before checking the Answer Key at the end.</p>")
        q = 0
        for p in _interleave(g.practice):
            q += 1
            out.append(f"<p class='lvl'>{p.level}</p><ol start='{q}'>"
                       f"<li>{_md_inline(p.question)}{_src(p.source)}</li></ol>")

    if g.cheat_sheet:
        out.append(section("Cheat Sheet"))
        out.append("<div class='box'><ul>" + "".join(f"<li>{_md_inline(x)}</li>" for x in g.cheat_sheet)
                   + "</ul></div>")

    if g.gaps:
        out.append(section("Not Covered in Your Notes"))
        out.append("<p class='src'>Mentioned in your notes but not explained. Each one has a suggestion for what to do next.</p>")
        out.append("<ul>" + "".join(f"<li><b>{_md_inline(x.term)}</b> — {_md_inline(x.action)}</li>" for x in g.gaps) + "</ul>")

    if g.practice:
        out.append("<pdf:nextpage/><h2>Answer Key</h2>")
        q = 0
        for p in _interleave(g.practice):
            q += 1
            out.append(f"<p><b>{q}. {_md_inline(p.answer)}</b><br/>{_md_inline(p.explanation)}</p>")

    out.append(f"<p class='meta'><br/>{_html.escape(meta)}</p>")
    return "\n".join(out)


def guide_to_pdf(g: StudyGuide, images: dict[str, NoteImage], meta: str, pdf_path: Path) -> None:
    from xhtml2pdf import pisa

    body_font, mono_font = _register_fonts()
    css = CSS.replace("BODYFONT", body_font).replace("MONOFONT", mono_font)
    doc = (f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>"
           f"{guide_to_html(g, images, meta)}"
           "<div id='footer'>Page <pdf:pagenumber> of <pdf:pagecount></div></body></html>")
    with open(pdf_path, "wb") as fh:
        result = pisa.CreatePDF(doc, dest=fh, encoding="utf-8")
    if result.err:
        raise RuntimeError(f"PDF rendering failed ({result.err} errors)")


EASY_CSS = """
@page {
  size: a4; margin: 1.8cm 1.7cm 2cm 1.7cm;
  @frame footer { -pdf-frame-content: footer; bottom: 0.7cm; margin-left: 1.7cm; margin-right: 1.7cm; height: 0.8cm; }
}
body { font-family: BODYFONT; font-size: 12pt; line-height: 1.6; color: #1f2937; }
h1 { font-size: 24pt; color: #047857; margin: 0; }
.tag { color: #047857; font-size: 10pt; font-weight: bold; }
h2 { -pdf-keep-with-next: true; font-size: 15pt; color: #ffffff; background-color: #059669; padding: 5px 8px; margin: 16px 0 8px 0; }
h3 { font-size: 13.5pt; color: #047857; margin: 2px 0 4px 0; }
p { margin: 3px 0; }
li { margin: 3px 0; }
code { font-family: MONOFONT; font-size: 10.5pt; background-color: #f3f4f6; }
.one { font-size: 14pt; font-weight: bold; background-color: #ecfdf5; border-left: 5px solid #059669; padding: 8px 10px; margin: 8px 0; }
.card { border: 1px solid #d1d5db; margin: 0 0 10px 0; }
.card td { padding: 8px 10px; vertical-align: top; }
.like { background-color: #fefce8; border-left: 4px solid #ca8a04; padding: 5px 8px; margin: 4px 0; }
.ex { background-color: #f9fafb; border-left: 4px solid #9ca3af; padding: 5px 8px; margin: 4px 0; }
.rem { background-color: #ecfdf5; border-left: 4px solid #059669; padding: 5px 8px; margin: 4px 0; font-weight: bold; }
.top td { background-color: #ecfdf5; border: 2px solid #059669; padding: 8px 12px; font-size: 13pt; }
.meta { color: #6b7280; font-size: 8.5pt; }
#footer { text-align: center; color: #9ca3af; font-size: 8pt; }
"""


def easy_to_html(g, meta: str) -> str:
    out = [f"<div class='tag'>EASY VERSION · quick to read</div><h1>{_html.escape(g.title)}</h1>",
           f"<div class='one'>{_md_inline(g.one_sentence)}</div>"]
    if g.top_three:
        out.append("<table class='top' repeat='0'><tr><td><b>★ If you only remember 3 things:</b><ol>"
                   + "".join(f"<li>{_md_inline(x)}</li>" for x in g.top_three) + "</ol></td></tr></table>")
    out.append(f"<h2>The story in a nutshell</h2><p>{_md_inline(g.story)}</p>")
    if g.concepts:
        out.append("<h2>The ideas, one at a time</h2>")
        for c in g.concepts:
            parts = [f"<h3>{_html.escape(c.name)}</h3>", f"<p>{_md_inline(c.in_plain_words)}</p>"]
            if c.think_of_it_like:
                parts.append(f"<div class='like'><b>☛ Think of it like:</b> {_md_inline(c.think_of_it_like)}</div>")
            if c.from_your_notes:
                parts.append(f"<div class='ex'><b>In your notes:</b> {_md_inline(c.from_your_notes)}</div>")
            if c.remember:
                parts.append(f"<div class='rem'>✓ Remember: {_md_inline(c.remember)}</div>")
            if c.try_it:
                parts.append(f"<div class='ex'><b>Try it:</b> {_md_inline(c.try_it)}</div>")
            out.append(_card("".join(parts)))
    if g.key_terms:
        out.append("<h2>Key terms</h2>")
        for t in g.key_terms:
            out.append(_card(f"<h3>{_html.escape(t.term)}</h3>"
                             f"<p>{_md_inline(t.definition)}</p>"))
    if g.connections:
        out.append("<h2>How they connect</h2><ul>"
                   + "".join(f"<li>{_md_inline(c)}</li>" for c in g.connections) + "</ul>")
    if g.must_know:
        out.append("<h2>Must-know for the exam</h2><ul>"
                   + "".join(f"<li>{_md_inline(x)}</li>" for x in g.must_know) + "</ul>")
    if g.worked_examples:
        out.append("<h2>Worked example</h2>")
        for w in g.worked_examples:
            parts = [f"<h3>{_html.escape(w.title)}</h3>"]
            parts += [f"<p><b>Step {i}.</b> {_md_inline(s)}</p>" for i, s in enumerate(w.steps, 1)]
            parts.append(f"<p><b>Answer:</b> {_md_inline(w.answer)}</p>")
            if w.takeaway:
                parts.append(f"<div class='box'><b>Takeaway:</b> {_md_inline(w.takeaway)}</div>")
            out.append(_card("".join(parts)))
    if g.dont_mix_up:
        out.append("<h2>Don't mix these up</h2><ul>"
                   + "".join(f"<li>{_md_inline(x)}</li>" for x in g.dont_mix_up) + "</ul>")
    if g.memory_tricks:
        out.append("<h2>Memory tricks</h2>")
        for m in g.memory_tricks:
            out.append(f"<div class='like'><b>{_md_inline(m.trick)}</b><br/>→ {_md_inline(m.helps_remember)}</div>")
    if g.quick_check:
        out.append("<h2>Quick check</h2><p class='meta'>Cover the answers and try first.</p>")
        for i, q in enumerate(g.quick_check, 1):
            out.append(f"<p><b>{i}. {_md_inline(q.question)}</b><br/>"
                       f"<span style='color:#047857'>→ {_md_inline(q.answer)}</span></p>")
    out.append(f"<p class='meta'><br/>{_html.escape(meta)}</p>")
    return "\n".join(out)


def easy_to_pdf(g, meta: str, pdf_path: Path) -> None:
    from xhtml2pdf import pisa

    body_font, mono_font = _register_fonts()
    css = EASY_CSS.replace("BODYFONT", body_font).replace("MONOFONT", mono_font)
    doc = (f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>"
           f"{easy_to_html(g, meta)}"
           "<div id='footer'>Page <pdf:pagenumber> of <pdf:pagecount></div></body></html>")
    with open(pdf_path, "wb") as fh:
        result = pisa.CreatePDF(doc, dest=fh, encoding="utf-8")
    if result.err:
        raise RuntimeError(f"PDF rendering failed ({result.err} errors)")


def pdf_to_images(pdf_path: str, scale: float = 1.4) -> list:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        return [pdf[i].render(scale=scale).to_pil() for i in range(len(pdf))]
    finally:
        pdf.close()
