"""Turn uploaded note files into text + the images inside them.

Every extractor returns an `Extracted` with:
  text    — plain text with page/slide anchors ("--- Slide 3 ---")
  images  — NoteImage(label, data, mime) for pictures worth showing a vision model
  skipped — human-readable notes about images that could not be used
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

IMAGE_EXTS = ["png", "jpg", "jpeg", "webp"]
SUPPORTED = ["pdf", "docx", "pptx", "xlsx", "csv", "txt", "md", *IMAGE_EXTS]

MAX_IMAGES = 16          # per file; keeps free-tier/vision calls bounded
MAX_SIDE = 1280          # downscale long side to this many px
MIN_BYTES = 2500         # ignore icons, bullets, logos
MIN_SIDE_PX = 80
_EMU_PER_INCH = 914400
_USABLE_MIME = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/bmp", "image/tiff", "image/webp"}


@dataclass
class NoteImage:
    label: str            # anchor used in prompts/citations, e.g. "Slide 8 image 1"
    data: bytes           # normalised PNG/JPEG bytes
    mime: str
    description: str = ""  # filled later by a vision model


@dataclass
class Extracted:
    text: str
    images: list[NoteImage] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------
def _normalise(blob: bytes) -> tuple[bytes, str] | None:
    """Re-encode any raster image as RGB JPEG/PNG with a capped size. None if unusable."""
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(blob))
        im.load()
    except Exception:  # noqa: BLE001
        return None
    if min(im.size) < MIN_SIDE_PX:
        return None
    if getattr(im, "n_frames", 1) > 1:
        im.seek(0)
    has_alpha = im.mode in ("RGBA", "LA", "P")
    im = im.convert("RGBA" if has_alpha else "RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    if has_alpha:
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    # PNG keeps diagrams/code crisp; JPEG for photos (large PNGs)
    im.save(out, format="PNG", optimize=True)
    if out.tell() > 900_000:
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=85)
        return out.getvalue(), "image/jpeg"
    return out.getvalue(), "image/png"


def _add_image(res: Extracted, label: str, blob: bytes, mime: str) -> None:
    mime = (mime or "").lower()
    if mime and mime not in _USABLE_MIME:
        res.skipped.append(f"{label}: {mime.split('/')[-1].upper()} vector image (not supported, skipped)")
        return
    if len(blob) < MIN_BYTES:
        return
    norm = _normalise(blob)
    if norm is None:
        return
    res.images.append(NoteImage(label=label, data=norm[0], mime=norm[1]))


def _cap_images(res: Extracted) -> None:
    # de-duplicate identical pictures (logos repeated on every slide)
    seen, uniq = set(), []
    for im in res.images:
        h = hash(im.data)
        if h in seen:
            continue
        seen.add(h)
        uniq.append(im)
    if len(uniq) > MAX_IMAGES:
        # keep the largest (most information-dense) ones, in document order
        keep = set(id(x) for x in sorted(uniq, key=lambda x: len(x.data), reverse=True)[:MAX_IMAGES])
        res.skipped.append(f"{len(uniq) - MAX_IMAGES} smaller images not sent (limit {MAX_IMAGES} per file)")
        uniq = [x for x in uniq if id(x) in keep]
    res.images = uniq


# ---------------------------------------------------------------------------
# extractors
# ---------------------------------------------------------------------------
def _read_plain(data: bytes) -> str:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> Extracted:
    import pdfplumber

    parts, image_pages = [], []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            parts.append(f"--- Page {i} ---")
            parts.append(text)
            for table in page.extract_tables() or []:
                for row in table:
                    parts.append(" | ".join("" if c is None else str(c) for c in row))
            big_pics = [im for im in (page.images or [])
                        if (im.get("width", 0) or 0) * (im.get("height", 0) or 0) > 120 * 120]
            # scanned page (no text layer) or page carrying real figures -> render it
            if len(text.strip()) < 40 or big_pics:
                image_pages.append(i)
    res = Extracted(text="\n".join(parts))
    if image_pages:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(data)
        try:
            for i in image_pages:
                pil = doc[i - 1].render(scale=1.6).to_pil()
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                _add_image(res, f"Page {i}", buf.getvalue(), "image/png")
        finally:
            doc.close()
    return res


def _extract_docx(data: bytes) -> Extracted:
    import docx

    doc = docx.Document(io.BytesIO(data))
    parts = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style = (p.style.name or "").lower() if p.style is not None else ""
        if style.startswith("heading"):
            level = "".join(ch for ch in style if ch.isdigit()) or "1"
            parts.append("#" * int(level) + " " + text)
        elif "list" in style:
            parts.append("- " + text)
        else:
            parts.append(text)
    for t in doc.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    res = Extracted(text="\n".join(parts))
    n = 0
    for rel in doc.part.rels.values():
        if "image" in rel.reltype and not rel.is_external:
            n += 1
            part = rel.target_part
            _add_image(res, f"Figure {n}", part.blob, part.content_type)
    return res


def _iter_shapes(shapes):
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _iter_shapes(sh.shapes)
        else:
            yield sh


def _extract_pptx(data: bytes) -> Extracted:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    prs = Presentation(io.BytesIO(data))
    parts: list[str] = []
    res = Extracted(text="")
    for i, slide in enumerate(prs.slides, 1):
        parts.append(f"--- Slide {i} ---")
        pic_n = 0
        for shape in _iter_shapes(slide.shapes):
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(r.text for r in para.runs).strip()
                    if text:
                        parts.append("  " * para.level + text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    parts.append(" | ".join(c.text.strip() for c in row.cells))
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                # skip tiny decorations (< ~0.8 inch on a side)
                if min(shape.width or 0, shape.height or 0) < 0.8 * _EMU_PER_INCH:
                    continue
                pic_n += 1
                try:
                    img = shape.image
                    _add_image(res, f"Slide {i} image {pic_n}", img.blob, img.content_type)
                except Exception:  # noqa: BLE001 — linked/missing picture
                    res.skipped.append(f"Slide {i} image {pic_n}: could not be read")
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"[Speaker notes] {notes}")
    res.text = "\n".join(parts)
    return res


def _extract_table(data: bytes, ext: str) -> Extracted:
    import pandas as pd

    if ext == "csv":
        sheets = {"CSV": pd.read_csv(io.BytesIO(data))}
    else:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine="openpyxl")
    parts = []
    for name, df in sheets.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        parts.append(f"--- Sheet: {name} ---")
        parts.append(df.to_csv(index=False))
    return Extracted(text="\n".join(parts))


MAX_IMAGES_TOTAL = 30    # across all files combined into one reviewer


def combine(parts: list[tuple[str, Extracted]]) -> Extracted:
    """Merge several extracted files into one source for a single reviewer.

    With more than one file, every anchor is prefixed with a short file tag so citations
    and image labels stay unambiguous ("Lecture2: Slide 8 image 1").
    """
    if len(parts) == 1:
        return parts[0][1]
    texts, images, skipped = [], [], []
    for name, ex in parts:
        tag = name.rsplit(".", 1)[0][:28]
        texts.append(f"===== FILE: {name} =====\n{ex.text}")
        for im in ex.images:
            images.append(NoteImage(label=f"{tag}: {im.label}", data=im.data, mime=im.mime))
        skipped += [f"{tag}: {s}" for s in ex.skipped]
    res = Extracted(text="\n\n".join(texts), images=images, skipped=skipped)
    if len(res.images) > MAX_IMAGES_TOTAL:
        keep = set(id(x) for x in sorted(res.images, key=lambda x: len(x.data), reverse=True)[:MAX_IMAGES_TOTAL])
        res.skipped.append(f"{len(res.images) - MAX_IMAGES_TOTAL} smaller pictures not used (limit {MAX_IMAGES_TOTAL})")
        res.images = [x for x in res.images if id(x) in keep]
    return res


def extract(filename: str, data: bytes) -> Extracted:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in ("txt", "md"):
        return Extracted(text=_read_plain(data))
    if ext == "pdf":
        res = _extract_pdf(data)
    elif ext == "docx":
        res = _extract_docx(data)
    elif ext == "pptx":
        res = _extract_pptx(data)
    elif ext in ("xlsx", "csv"):
        res = _extract_table(data, ext)
    elif ext in IMAGE_EXTS:
        res = Extracted(text="")
        _add_image(res, "Uploaded image", data, "image/" + ("jpeg" if ext == "jpg" else ext))
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
    _cap_images(res)
    return res
