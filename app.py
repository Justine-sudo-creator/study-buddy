"""Study Buddy — turn lecture notes into an insightful reviewer + flashcards.

Run:  cd ~/Downloads/study_workspace && .venv/bin/streamlit run app.py
Keys: study_workspace/.env  ->  GEMINI_API_KEY=... (free), TYPESAFE_API_KEY=... (optional, Jev)

Several files can be uploaded at once; they are combined into ONE reviewer. Uploaded files are
processed in memory and NOT kept. The finished reviewer PDF + flashcards are saved to
C:\\Users\\USER\\Downloads\\study_workspace\\reviewers (download buttons too) and can be deleted
from the "My reviewers" tab.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from study import engines, quality
from study.extract import SUPPORTED, Extracted, NoteImage, combine, extract
from study.pdf import easy_to_pdf, guide_to_pdf, pdf_to_images

# Save finished reviewers where they're easy to find in Windows File Explorer
# (C:\Users\USER\Downloads\study_workspace\reviewers). Falls back to the app folder.
_WIN_DEFAULT = Path("/mnt/c/Users/USER/Downloads/study_workspace")
BASE_DIR = Path(os.environ.get("STUDY_WORKSPACE") or (_WIN_DEFAULT if _WIN_DEFAULT.parent.exists()
                                                      else Path(__file__).resolve().parent))
OUT_DIR = BASE_DIR / "reviewers"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR = OUT_DIR / "_json"
JSON_DIR.mkdir(parents=True, exist_ok=True)


def windows_path(p: Path) -> str:
    s = str(p)
    m = re.match(r"^/mnt/([a-z])/(.*)$", s)
    return f"{m.group(1).upper()}:\\" + m.group(2).replace("/", "\\") if m else s


def safe_stem(name: str) -> str:
    s = re.sub(r"[^\w\- ]+", "", name).strip().replace(" ", "_")
    return s[:80] or "Reviewer"
GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
JEV_KEY = quality.jev_key()
LOCAL_MODEL = os.environ.get("STUDY_MODEL", "qwen2.5-coder:1.5b")

st.set_page_config(page_title="Study Buddy", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .hero h1 {margin-bottom: 0;}
      .hero p {color: #6b7280; font-size: 1.05rem; margin-top: .25rem;}
      .step {font-weight: 600; font-size: 1.1rem; margin: 1.2rem 0 .3rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_file(name: str, data: bytes):
    return extract(name, data)


@st.cache_data(show_spinner=False)
def pdf_pages(path: str, mtime: float):
    return pdf_to_images(path)


@st.cache_data(ttl=600, show_spinner=False)
def gemini_models(key: str):
    return engines.gemini_available_models(key)


@st.cache_data(ttl=120, show_spinner=False)
def local_models():
    return engines.ollama_models()


@st.cache_resource(show_spinner=False)
def prewarm_local(model: str) -> None:
    import threading

    threading.Thread(target=engines.ollama_prewarm, args=(model, LOCAL_NUM_CTX), daemon=True).start()


MAX_FILES = 20
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_NOTE_CHARS_GEMINI = 200_000
MAX_NOTE_CHARS_LOCAL = 60_000
LOCAL_NUM_CTX = 4096


def saved_reviewers() -> list[dict]:
    stems: dict[str, float] = {}
    for suf in ("_Reviewer.pdf", "_Easy.pdf"):
        for pdf in OUT_DIR.glob(f"*{suf}"):
            stem = pdf.name.removesuffix(suf)
            stems[stem] = max(stems.get(stem, 0), pdf.stat().st_mtime)
    items = []
    for stem, mtime in sorted(stems.items(), key=lambda kv: kv[1], reverse=True):
        info_path = JSON_DIR / f"{stem}_info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        items.append(dict(stem=stem, info=info, title=info.get("title") or stem.replace("_", " "),
                          when=datetime.fromtimestamp(mtime)))
    return items


SRC_DIR = OUT_DIR / "_notes_cache"   # extracted text + pictures, so a missing version can be made later
SUFFIXES = ("_Reviewer.pdf", "_Easy.pdf", "_Quizlet.txt")


def delete_reviewer(stem: str) -> None:
    import shutil

    for suffix in SUFFIXES:
        (OUT_DIR / f"{stem}{suffix}").unlink(missing_ok=True)
    for suf in ("_cards.json", "_info.json"):
        (JSON_DIR / f"{stem}{suf}").unlink(missing_ok=True)
    shutil.rmtree(SRC_DIR / stem, ignore_errors=True)


def save_source(stem: str, ex: Extracted, names: list[str], analysis) -> None:
    """Keep what was read from the notes (not the original files) so the other version can be made later."""
    import shutil

    d = SRC_DIR / stem
    shutil.rmtree(d, ignore_errors=True)
    (d / "pics").mkdir(parents=True)
    pics = []
    for i, im in enumerate(ex.images):
        fn = f"{i:03d}.{'jpg' if im.mime == 'image/jpeg' else 'png'}"
        (d / "pics" / fn).write_bytes(im.data)
        pics.append(dict(file=fn, label=im.label, mime=im.mime))
    (d / "source.json").write_text(json.dumps(dict(names=names, text=ex.text, skipped=ex.skipped, pics=pics),
                                              ensure_ascii=False))
    if analysis is not None:
        (d / "analysis.json").write_text(analysis.model_dump_json())


def load_source(stem: str):
    """-> (Extracted, names, analysis|None) or None if the notes weren't kept."""
    d = SRC_DIR / stem
    if not (d / "source.json").exists():
        return None
    src = json.loads((d / "source.json").read_text())
    images = [NoteImage(label=p["label"], data=(d / "pics" / p["file"]).read_bytes(), mime=p["mime"])
              for p in src["pics"] if (d / "pics" / p["file"]).exists()]
    analysis = None
    if (d / "analysis.json").exists():
        try:
            analysis = engines.NotesAnalysis.model_validate_json((d / "analysis.json").read_text())
        except Exception:  # noqa: BLE001
            analysis = None
    return Extracted(text=src["text"], images=images, skipped=src.get("skipped", [])), src["names"], analysis


def friendly_error(e: Exception) -> str:
    msg = str(e)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
        return "You've hit the free limit for now. Wait a minute and try again (there is also a daily limit)."
    if "503" in msg or "UNAVAILABLE" in msg:
        return "The AI service is very busy right now. Please try again in a minute."
    if "API key" in msg or "401" in msg or "403" in msg or "PERMISSION" in msg:
        return "The AI key isn't working. Check GEMINI_API_KEY in the .env file."
    if "Connection" in msg and "11434" in msg:
        return "Offline mode needs Ollama running on this computer."
    return f"Something went wrong: {msg[:300]}"


def download_buttons(stem: str, key_prefix: str) -> None:
    """Prominent download buttons for every file that exists for this reviewer."""
    files = [(OUT_DIR / f"{stem}_Reviewer.pdf", "⬇ Full reviewer (PDF)", "application/pdf"),
             (OUT_DIR / f"{stem}_Easy.pdf", "⬇ Easy version (PDF)", "application/pdf"),
             (OUT_DIR / f"{stem}_Quizlet.txt", "⬇ Flashcards (TXT)", "text/plain")]
    files = [f for f in files if f[0].exists()]
    if not files:
        return
    cols = st.columns(len(files))
    for i, (path, label, mime) in enumerate(files):
        cols[i].download_button(label, path.read_bytes(), file_name=path.name, mime=mime,
                                type="primary" if i == 0 else "secondary", use_container_width=True,
                                key=f"{key_prefix}_dl_{path.name}")
    st.caption(f"📁 Also saved on your computer in: {windows_path(OUT_DIR)}")


def _pdf_preview(pdf: Path) -> None:
    pages = pdf_pages(str(pdf), pdf.stat().st_mtime)
    st.caption(f"{len(pages)} pages")
    with st.container(height=820):
        for img in pages:
            st.image(img, width="stretch")


def show_result(stem: str, key_prefix: str) -> None:
    """Download buttons + preview of a saved reviewer (full / easy PDFs) and its flashcards."""
    full, easy = OUT_DIR / f"{stem}_Reviewer.pdf", OUT_DIR / f"{stem}_Easy.pdf"
    cards_txt = OUT_DIR / f"{stem}_Quizlet.txt"
    if not (full.exists() or easy.exists()):
        st.info("This reviewer file is missing.")
        return
    download_buttons(stem, key_prefix)
    names = (["📄 Full reviewer"] if full.exists() else []) + (["🌱 Easy version"] if easy.exists() else []) \
        + ["🃏 Flashcards"]
    tabs = dict(zip(names, st.tabs(names)))
    if "📄 Full reviewer" in tabs:
        with tabs["📄 Full reviewer"]:
            _pdf_preview(full)
    if "🌱 Easy version" in tabs:
        with tabs["🌱 Easy version"]:
            _pdf_preview(easy)
    with tabs["🃏 Flashcards"]:
        lines = [l for l in (cards_txt.read_text(encoding="utf-8").splitlines() if cards_txt.exists() else []) if " # " in l]
        st.caption(f"{len(lines)} cards · click a card to see the answer")
        if lines:
            with st.container(height=520):
                for i, line in enumerate(lines, 1):
                    q, a = line.split(" # ", 1)
                    with st.expander(f"{i}. {q}"):
                        st.write(a)
            with st.expander("How to import these into Quizlet"):
                st.markdown(
                    "1. In Quizlet, create a new set and click **Import**.\n"
                    f"2. Download the flashcards above, open `{cards_txt.name}`, copy everything and paste it in.\n"
                    "3. **Between term and definition:** choose *Custom* and type ` # ` (space, hash, space).\n"
                    "4. **Between cards:** choose *New line*."
                )
        cards_json = JSON_DIR / f"{stem}_cards.json"
        if cards_json.exists():
            dropped = [r for r in json.loads(cards_json.read_text()) if not r["kept"]]
            if dropped:
                with st.expander(f"{len(dropped)} weak cards were removed"):
                    for r in dropped:
                        st.markdown(f"- {r['question']}  \n  *{r['reason']}*")


DEPTHS = ["Straight to the point", "Balanced", "Explain everything"]


def make_reviewer(stem: str, ex: Extracted, names: list[str], want_full: bool, want_easy: bool, online: bool,
                  include_pics: bool, depth: str, use_jev: bool, add_to_existing: bool = False,
                  analysis=None) -> bool:
    """Generate the requested version(s) and save them. With add_to_existing=True the other version
    and the existing flashcards are kept, and the new flashcards are merged in."""
    images = ex.images if (online and include_pics) else []
    source_label = names[0] if len(names) == 1 else f"{len(names)} files: " + ", ".join(names)
    temp = {"Straight to the point": 0.2, "Balanced": 0.35, "Explain everything": 0.5}[depth]
    style_note = {"Straight to the point": "\nSTYLE: concise; short explanations.", "Balanced": "",
                  "Explain everything": "\nSTYLE: thorough; explain each idea in depth with extra steps."}[depth]
    multi_note = ""
    if len(names) > 1:
        style_note += ("\nMULTIPLE FILES: these notes come from several files (marked '===== FILE: ...'). "
                       "Write ONE integrated reviewer: merge overlapping ideas, show how the files connect "
                       "and build on each other, and cite anchors with the file tag (e.g. 'Lecture2: Slide 4').")
        multi_note = "\nMULTIPLE FILES: combine them into ONE explanation; show how the files connect."
    label = "Making your reviewer…" if not add_to_existing else "Making the other version…"
    with st.status(label, expanded=True) as status:
        try:
            guide = easy = None
            if online:
                models = gemini_models(GEMINI_KEY)
                if analysis is None:
                    analysis, used = engines.gemini_analyse(GEMINI_KEY, models, source_label,
                                                                 ex.text, images, status.write,
                                                                 ex.skipped)
                    models = [used] + [m for m in models if m != used]
                else:
                    status.write("Re-using what we learned from your notes last time…")
                block = engines.notes_block(source_label, ex.text[:MAX_NOTE_CHARS_GEMINI],
                                                 images, ex.skipped)
                if want_full:
                    guide, _ = engines.gemini_write_full(GEMINI_KEY, models, block, analysis,
                                                              images, temp, status.write, style_note)
                if want_easy:
                    easy, _ = engines.gemini_write_easy(GEMINI_KEY, models, block, analysis,
                                                             temp, status.write, multi_note)
                made_with = "Gemini"
            else:
                local, _ = local_models()
                if not local:
                    raise RuntimeError("Connection to Ollama at 11434 failed")
                model = LOCAL_MODEL if LOCAL_MODEL in local else local[0]
                offline_text = ex.text[:MAX_NOTE_CHARS_LOCAL]
                if want_full:
                    guide = engines.ollama_study_guide(model, source_label, offline_text, [], temp, LOCAL_NUM_CTX,
                                                       status.write)
                if want_easy:
                    easy = engines.ollama_easy_guide(model, source_label, offline_text, temp, LOCAL_NUM_CTX, status.write)
                made_with = "offline mode"

            status.write("Checking your flashcards…")
            new_cards = (guide.flashcards if guide else []) + (easy.flashcards if easy else [])
            old_cards = []
            cards_json = JSON_DIR / f"{stem}_cards.json"
            if add_to_existing and cards_json.exists():
                old_cards = [engines.Flashcard(question=r["question"], answer=r["answer"], source=r.get("source", ""))
                             for r in json.loads(cards_json.read_text()) if r.get("kept")]
            verdicts = quality.local_check(old_cards + new_cards)   # also removes duplicates
            if use_jev:
                try:
                    figs = "\n".join(f"[{f.label}] {f.insight}" for f in (guide.figures if guide else []))
                    status.write(quality.jev_check(JEV_KEY, ex.text + "\n" + figs, "", verdicts, 0.5, 0.4))
                except Exception:  # noqa: BLE001
                    status.write("Extra check skipped (Jev unavailable).")

            status.write("Designing your PDF…")
            made_from = names[0] if len(names) == 1 else f"{len(names)} files ({', '.join(names)})"
            meta = f"Made from {made_from} with {made_with} · {date.today():%B %d, %Y}"
            if not add_to_existing:
                for suf in ("_Reviewer.pdf", "_Easy.pdf"):   # replace older versions with this name
                    (OUT_DIR / f"{stem}{suf}").unlink(missing_ok=True)
            if guide:
                guide_to_pdf(guide, {i.label: i for i in ex.images}, meta, OUT_DIR / f"{stem}_Reviewer.pdf")
            if easy:
                easy_to_pdf(easy, meta, OUT_DIR / f"{stem}_Easy.pdf")
            (OUT_DIR / f"{stem}_Quizlet.txt").write_text(quality.to_quizlet(verdicts) + "\n", encoding="utf-8")
            cards_json.write_text(json.dumps(
                [dict(question=v.card.question, answer=v.card.answer, source=v.card.source,
                      kept=v.ok, reason=v.reason) for v in verdicts], indent=1, ensure_ascii=False))
            info_path = JSON_DIR / f"{stem}_info.json"
            info = json.loads(info_path.read_text()) if (add_to_existing and info_path.exists()) else {}
            if guide:   # the full reviewer's title/subject are the most descriptive
                info.update(title=guide.title, subject=guide.subject)
            elif not info.get("title"):
                info["title"] = easy.title
            info.update(source=made_from, made_with=made_with,
                        kinds=sorted({*info.get("kinds", []), *(["full"] if guide else []), *(["easy"] if easy else [])}))
            info_path.write_text(json.dumps(info))
            save_source(stem, ex, names, analysis if online else None)
            status.update(label="Done!", state="complete", expanded=False)
            return True
        except Exception as e:  # noqa: BLE001
            status.update(label="Couldn't make the reviewer", state="error")
            st.error(friendly_error(e))
            return False


def add_missing_version(it: dict) -> None:
    """In 'My reviewers': offer to make the version (full or easy) this reviewer doesn't have yet."""
    stem = it["stem"]
    has_full = (OUT_DIR / f"{stem}_Reviewer.pdf").exists()
    has_easy = (OUT_DIR / f"{stem}_Easy.pdf").exists()
    if has_full and has_easy:
        return
    missing_full = not has_full
    what = "📄 full reviewer" if missing_full else "🌱 easy version"
    blurb = ("the detailed version: concepts, worked examples, common mistakes and practice questions"
             if missing_full else "a plain-words version with everyday comparisons and only the must-knows")
    with st.container(border=True):
        st.markdown(f"**Want the {what} too?** Make {blurb} from the same notes. "
                    "Your existing files stay as they are.")
        cached = load_source(stem)
        online = bool(GEMINI_KEY)
        if cached:
            ex, names, analysis = cached
            if st.button(f"✨ Make the {what}", type="primary", key=f"addver_{stem}"):
                if make_reviewer(stem, ex, names, missing_full, not missing_full, online, True, "Balanced",
                                 bool(JEV_KEY), add_to_existing=True, analysis=analysis):
                    st.toast("Added! Find it in the tabs below.")
                    st.rerun()
        else:
            src = it["info"].get("source", "")
            st.caption("Upload the same notes again to make it"
                       + (f" (this one was made from **{src}**)." if src else "."))
            files = st.file_uploader("Same notes", type=SUPPORTED, accept_multiple_files=True,
                                     key=f"reup_{stem}", label_visibility="collapsed")
            if files:
                parts = []
                for f in files:
                    try:
                        parts.append((f.name, read_file(f.name, f.getvalue())))
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Sorry, {f.name} couldn't be opened ({e}).")
                if parts:
                    names = [n for n, _ in parts]
                    if src and not any(n in src for n in names):
                        st.warning("These don't look like the same files as before — the two versions may not match.")
                    if st.button(f"✨ Make the {what}", type="primary", key=f"addver_{stem}"):
                        if make_reviewer(stem, combine(parts), names, missing_full, not missing_full, online,
                                         True, "Balanced", bool(JEV_KEY), add_to_existing=True):
                            st.toast("Added! Find it in the tabs below.")
                            st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("<div class='hero'><h1>📚 Study Buddy</h1>"
            "<p>Upload your lecture notes and get a clear reviewer and flashcards made from them.</p></div>",
            unsafe_allow_html=True)

tab_new, tab_saved, tab_help = st.tabs(["✨ New reviewer", "📁 My reviewers", "❓ Help"])

# ---------------------------------------------------------------------------
# New reviewer
# ---------------------------------------------------------------------------
with tab_new:
    st.markdown("<div class='step'>1 · Upload your notes</div>", unsafe_allow_html=True)
    ups = st.file_uploader(
        "PDF, Word, PowerPoint, Excel, text or photos — up to 20 files, 100 MB each; "
        "they'll be combined into one reviewer",
        type=SUPPORTED, accept_multiple_files=True,
        help="Your files are only read to make the reviewer. They aren't stored.",
    )

    st.markdown("<div class='step'>2 · Choose how to make it</div>", unsafe_allow_html=True)
    modes = []
    if GEMINI_KEY:
        modes.append("Best quality (online, free)")
    modes.append("Private (offline, on this computer)")
    mode = st.radio("Mode", modes, horizontal=True, label_visibility="collapsed")
    online = mode.startswith("Best")
    if online:
        st.caption("Uses Google's free Gemini AI. It reads your text **and** your diagrams, formulas and "
                   "screenshots. Your notes are sent to Google to do this.")
    else:
        st.caption("Nothing leaves your computer, but the result is much simpler and can take 5–10 minutes. "
                   "Pictures in your notes are skipped.")
        local, _ = local_models()
        if local:
            prewarm_local(LOCAL_MODEL if LOCAL_MODEL in local else local[0])
        if not GEMINI_KEY:
            st.info("Tip: add a free Gemini key (see Help) for much better reviewers that also read your diagrams.")

    st.markdown("<div class='step'>3 · Pick the kind of reviewer</div>", unsafe_allow_html=True)
    kind = st.radio(
        "Kind", ["📄 Full reviewer", "🌱 Easy version", "Both"], horizontal=True, label_visibility="collapsed",
        help="Full: detailed, for real understanding. Easy: simple words, everyday comparisons and only the "
             "must-knows — great if you're confused or cramming.")
    st.caption({
        "📄 Full reviewer": "Detailed and thorough: concepts, worked examples, common mistakes, practice questions.",
        "🌱 Easy version": "Explained like you're 12: plain words, everyday comparisons, memory tricks and only the "
                          "must-knows. Great if you're still confused or cramming before an exam.",
        "Both": "Get the detailed reviewer and the easy version together (takes a bit longer).",
    }[kind])
    want_full = kind in ("📄 Full reviewer", "Both")
    want_easy = kind in ("🌱 Easy version", "Both")

    with st.expander("More options"):
        include_pics = st.toggle("Read the pictures in my notes", value=online, disabled=not online,
                                 help="Diagrams, formulas and screenshots are often where the key ideas are.")
        depth = st.select_slider("Full reviewer detail", DEPTHS,
                                 value="Balanced", disabled=not want_full)
        use_jev = False
        if JEV_KEY:
            use_jev = st.toggle("Extra flashcard check (Jev)", value=True,
                                help="Removes flashcards that aren't clearly backed by your notes or are too general.")

    if ups:
        # enforce upload limits
        if len(ups) > MAX_FILES:
            st.error(f"Too many files — {len(ups)} uploaded, max {MAX_FILES}. Please try again.")
            st.stop()
        oversize = [f.name for f in ups if f.size > MAX_FILE_SIZE]
        if oversize:
            st.error(f"Too large: {', '.join(oversize)} (max {MAX_FILE_SIZE // (1024*1024)} MB each). Try smaller files.")
            st.stop()
        parts, bad = [], []
        for f in ups:
            try:
                parts.append((f.name, read_file(f.name, f.getvalue())))
            except Exception as e:  # noqa: BLE001
                bad.append(f"{f.name} ({e})")
        for b in bad:
            st.error(f"Sorry, this file couldn't be opened: {b}")
        if not parts:
            st.stop()
        ex = combine(parts)
        names = [n for n, _ in parts]
        pics = len(ex.images)
        words = len(ex.text.split())
        if len(parts) == 1:
            st.success(f"**{names[0]}** — about {words:,} words" + (f" and {pics} pictures" if pics else ""))
        else:
            st.success(f"**{len(parts)} files** — about {words:,} words" + (f" and {pics} pictures" if pics else "")
                       + " in total. They'll be combined into **one** reviewer.")
            for n, e in parts:
                st.caption(f"• {n} — {len(e.text.split()):,} words, {len(e.images)} pictures")
        if pics:
            with st.expander("See the pictures we found"):
                cols = st.columns(4)
                for i, im in enumerate(ex.images):
                    cols[i % 4].image(im.data, caption=im.label, width="stretch")
                if ex.skipped:
                    st.caption(f"{len(ex.skipped)} picture note(s): some are in a format we can't read or "
                               "were left out to stay within the limit.")
        if not ex.text.strip() and not (online and include_pics and pics):
            st.warning("We couldn't find any text in these files. Use **Best quality** mode so the pictures can be read.")
        if not online and len(ex.text) > MAX_NOTE_CHARS_LOCAL:
            st.info(f"⚡ Offline mode reads the first {MAX_NOTE_CHARS_LOCAL:,} characters of your notes. "
                    "For full-length textbooks, use Best quality mode.")
        if not online and len(ex.text) > engines.LOCAL_CHUNK_CHARS:
            n_sections = len(engines.split_notes(ex.text[:MAX_NOTE_CHARS_LOCAL]))
            st.caption(f"Offline mode will work through your notes in {n_sections} sections "
                       "and show progress as each one finishes.")
        if online and len(ex.text) > MAX_NOTE_CHARS_GEMINI:
            st.warning(f"Gemini mode can only read the first {MAX_NOTE_CHARS_GEMINI:,} characters of your notes.")

        st.markdown("<div class='step'>4 · Make it</div>", unsafe_allow_html=True)
        default_name = Path(names[0]).stem if len(names) == 1 else Path(names[0]).stem + f"_plus_{len(names) - 1}"
        nice_name = st.text_input("Name your reviewer", value=default_name.replace("_", " "),
                                  help="Used for the file names, e.g. 'Midterm Week 1-3'.")
        stem = safe_stem(nice_name)
        source_label = names[0] if len(names) == 1 else f"{len(names)} files: " + ", ".join(names)
        if any((OUT_DIR / f"{stem}{x}").exists() for x in ("_Reviewer.pdf", "_Easy.pdf")) \
                and st.session_state.get("just_made") != stem:
            st.caption("A reviewer with this name already exists and will be replaced.")
        if st.button("✨ Make my reviewer", type="primary", use_container_width=True):
            if make_reviewer(stem, ex, names, want_full, want_easy, online, include_pics, depth, use_jev):
                st.session_state["just_made"] = stem

        if st.session_state.get("just_made") == stem:
            st.divider()
            show_result(stem, "new")
    else:
        st.info("👆 Start by uploading one or more files.")

# ---------------------------------------------------------------------------
# My reviewers
# ---------------------------------------------------------------------------
with tab_saved:
    items = saved_reviewers()
    if not items:
        st.info("You haven't made any reviewers yet.")
    else:
        st.caption(f"Saved in {windows_path(OUT_DIR)}")
        labels = [f"{it['title']}  ·  {it['when']:%b %d, %I:%M %p}" for it in items]
        idx = st.selectbox("Open a reviewer", range(len(items)), format_func=lambda i: labels[i])
        it = items[idx]
        c1, c2 = st.columns([5, 1])
        with c1:
            st.subheader(it["title"])
            if it["info"].get("source"):
                st.caption(f"From {it['info']['source']} · {it['info'].get('made_with', '')} · "
                           f"{it['when']:%b %d, %I:%M %p}")
        with c2:
            confirm_key = f"confirm_{it['stem']}"
            if st.session_state.get(confirm_key):
                st.warning("Delete this reviewer and its flashcards?")
                y, n = st.columns(2)
                if y.button("Delete", type="primary", key=f"yes_{it['stem']}"):
                    delete_reviewer(it["stem"])
                    st.session_state.pop(confirm_key, None)
                    if st.session_state.get("just_made") == it["stem"]:
                        st.session_state.pop("just_made")
                    st.toast("Reviewer deleted")
                    st.rerun()
                if n.button("Cancel", key=f"no_{it['stem']}"):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
            elif st.button("🗑 Delete", key=f"del_{it['stem']}", use_container_width=True):
                st.session_state[confirm_key] = True
                st.rerun()
        add_missing_version(it)
        show_result(it["stem"], "saved")

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
with tab_help:
    st.markdown(
        """
**What do I get?**
- **A reviewer (PDF)** that explains *your* lecture: the big picture, key concepts with examples from
  your notes, how ideas connect, worked examples step by step, common mistakes, practice questions
  with an answer key, and a one-page cheat sheet.
- **An easy version (PDF)** — the same lecture in plain words with everyday comparisons, memory tricks
  and only the must-knows. Pick it if you're still confused or cramming before an exam (or pick *Both*).
- **Flashcards** you can study here or import into Quizlet.

**Where are my files?** Use the download buttons above each reviewer, or open
`C:\\Users\\USER\\Downloads\\study_workspace\\reviewers` in File Explorer.

**Which mode should I use?**
- **Best quality** — free, uses Google's Gemini AI, reads pictures too. Needs internet.
- **Private** — runs entirely on this computer. Simpler results, slower, ignores pictures.

**Setting up Best quality (one time)**
1. Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (no credit card).
2. Open the `.env` file in the study_workspace folder and paste it after `GEMINI_API_KEY=`.
3. Restart the app.

**Good to know**
- You can upload **several files at once** (e.g. all lectures for a quiz) — they become one combined reviewer.
- Made only the easy version (or only the full one)? Open it in *My reviewers* and click
  **Make the full reviewer / easy version** — no need to upload again.
- Your original files are not kept (only the text and pictures read from them, so the other version
  can be made later). The reviewer and flashcards are saved in
  `C:\\Users\\USER\\Downloads\\study_workspace\\reviewers` and can also be downloaded with the buttons.
  Delete them anytime in *My reviewers*.
- The reviewer only uses what's in your notes. Anything your notes mention but don't explain is listed
  under *Not Covered in Your Notes* so you know what to look up.
- Free limits: a few reviewers per minute and a daily cap. If you hit it, wait and try again.
"""
    )
