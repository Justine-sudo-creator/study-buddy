"""Model back-ends + the study-guide schema.

Quality strategy (Gemini):
  Pass 1  ANALYSE  — read text + images like a tutor: what the lecture is really about, what it
                     stresses, where students get stuck, how ideas connect, what each figure shows.
  Pass 2  WRITE    — turn that analysis + the notes into a structured StudyGuide (concept cards,
                     worked examples, misconceptions, leveled practice, cheat sheet, flashcards).
The PDF layout is produced by code from the structured guide, so every reviewer has the same
clean design regardless of the model's formatting habits.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from .extract import NoteImage

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Free-tier Flash models, best first. Unavailable/overloaded ones are skipped automatically.
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3-flash-preview",
                 "gemini-flash-latest", "gemini-flash-lite-latest"]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ImageInsight(BaseModel):
    label: str = Field(description="Exact image label, e.g. 'Slide 8 image 1'.")
    shows: str = Field(description="Transcribe/explain what the image shows (text, code, formula, diagram).")
    teaches: str = Field(description="What a student should learn from it. 'decorative' if it carries no content.")


class NotesAnalysis(BaseModel):
    topic: str
    course_context: str = Field(description="Which course/unit this seems to be and what level.")
    central_question: str = Field(description="The one question this lecture is really answering.")
    key_ideas: list[str] = Field(description="Most important ideas, most important first, in the notes' terms.")
    emphasised: list[str] = Field(description="What the notes stress, repeat, box, bold or warn about.")
    tricky_points: list[str] = Field(description="Where a student is likely to get confused, and why.")
    relationships: list[str] = Field(description="How the ideas depend on / contrast with each other.")
    examples_in_notes: list[str] = Field(description="Every concrete example, value, code or proof in the notes.")
    image_insights: list[ImageInsight] = Field(default_factory=list)
    likely_exam_questions: list[str]
    gaps: list[str] = Field(description="Things mentioned but not explained in the notes.")


class Concept(BaseModel):
    name: str
    explanation: str = Field(description="Clear explanation in plain words using the notes' terms and examples.")
    why_it_matters: str = Field(description="Why this matters in this lecture / what it lets you do.")
    example: str = Field(description="A concrete example taken from the notes (values, code, case). '' if none.")
    watch_out: str = Field(description="The specific mistake students make with this. '' if the notes give none.")
    connects_to: list[str] = Field(default_factory=list, description="Other concept names it relates to.")
    source: str = Field(description="Anchor, e.g. 'Slides 4-6'.")


class WorkedExample(BaseModel):
    title: str
    problem: str
    steps: list[str] = Field(description="Each step explains WHAT is done and WHY.")
    answer: str
    takeaway: str = Field(description="The transferable lesson from this example.")
    source: str


class Formula(BaseModel):
    name: str
    formula: str = Field(description="Readable plain text with Unicode (Σ, ≤, ², √, →). No LaTeX.")
    meaning: str = Field(description="What it says in words, and what each symbol means.")
    when_to_use: str
    source: str


class Misconception(BaseModel):
    wrong: str = Field(description="What students wrongly believe.")
    right: str = Field(description="What is actually true, per the notes.")
    why: str = Field(description="Why the wrong idea is tempting / how to tell the difference.")


class PracticeQuestion(BaseModel):
    level: Literal["Remember", "Apply", "Analyze"]
    question: str
    answer: str
    explanation: str = Field(description="Why this is the answer; reasoning a student can learn from.")
    source: str


class FigureNote(BaseModel):
    label: str = Field(description="Exact image label as given, e.g. 'Slide 8 image 1'.")
    insight: str = Field(description="What this figure teaches, 1-3 sentences.")
    question: str = Field(description="A quick question about the figure for the student to answer before revealing the insight.", default="")


class Flashcard(BaseModel):
    question: str = Field(description="Front of the card. Plain text, no '#', no Markdown.")
    answer: str = Field(description="Back of the card, <= 25 words, plain text, no '#'. ")
    source: str = Field(description="Anchor in the notes, e.g. 'Slide 7' or 'Slide 8 image 1'.")


class KeyTerm(BaseModel):
    term: str = Field(description="The key word or phrase.")
    definition: str = Field(description="What it means in the context of these notes, plain words.")
    source: str = Field(description="Anchor, e.g. 'Slide 5'.")


class Gap(BaseModel):
    term: str = Field(description="What's missing or unexplained.")
    action: str = Field(description="What to do about it — e.g. 'Check chapter 4', 'Ask your teacher', 'Watch a 5-min recap'.")


class StudyGuide(BaseModel):
    title: str
    subject: str = Field(description="Course/subject, e.g. 'Data Structures & Algorithms'.")
    big_picture: str = Field(description="3-6 sentences: what this lecture is about, the central question, "
                                         "and how the pieces fit together. Written for a student.")
    learning_goals: list[str] = Field(description="'You should be able to ...' items, concrete and testable.")
    concepts: list[Concept]
    key_terms: list[KeyTerm] = Field(default_factory=list, description="Key vocabulary with definitions from the notes.")
    connections: list[str] = Field(description="How the concepts relate; each one a full insight sentence.")
    worked_examples: list[WorkedExample] = Field(default_factory=list)
    formulas: list[Formula] = Field(default_factory=list)
    figures: list[FigureNote] = Field(default_factory=list)
    misconceptions: list[Misconception] = Field(default_factory=list)
    practice: list[PracticeQuestion]
    cheat_sheet: list[str] = Field(description="One-line facts to review right before the exam.")
    gaps: list[Gap] = Field(default_factory=list, description="Mentioned but not explained in the notes, with what to do.")
    flashcards: list[Flashcard]


class EasyConcept(BaseModel):
    name: str
    in_plain_words: str = Field(description="1-3 short sentences a 12-year-old could follow. No jargon; if a "
                                            "technical word is needed, explain it right away.")
    think_of_it_like: str = Field(description="An everyday comparison (food, games, school, chores, sports).")
    from_your_notes: str = Field(description="A tiny concrete example taken from the notes. '' if none.")
    remember: str = Field(description="One short line to remember for the exam.")
    try_it: str = Field(default="", description="A mini problem/scenario where this concept is used immediately. "
                                     "1-2 sentences showing what to do, then the answer. "
                                     "'' if no natural scenario fits.")


class MemoryTrick(BaseModel):
    trick: str = Field(description="A mnemonic, rhyme, acronym or vivid image.")
    helps_remember: str = Field(description="What fact from the notes it helps you remember.")


class QuickCheck(BaseModel):
    question: str
    answer: str = Field(description="Short answer plus one simple 'because ...'.")


class EasyWorkedExample(BaseModel):
    title: str
    steps: list[str] = Field(description="3-4 short steps, plain words, each saying what to do and why.")
    answer: str
    takeaway: str = Field(description="The one-line lesson from this example.")
    source: str


class EasyGuide(BaseModel):
    title: str
    one_sentence: str = Field(description="The whole lecture in one simple sentence.")
    story: str = Field(description="The big picture as a short, friendly story or walkthrough, 3-6 short sentences.")
    concepts: list[EasyConcept]
    connections: list[str] = Field(default_factory=list,
                                           description="How the key ideas relate to each other.")
    key_terms: list[KeyTerm] = Field(default_factory=list,
                                            description="3-5 key words/phrases with plain definitions.")
    worked_examples: list[EasyWorkedExample] = Field(default_factory=list,
                                                       description="One simple worked example showing how to apply a concept.")
    must_know: list[str] = Field(description="The facts most likely to be on the exam, each one short line.")
    dont_mix_up: list[str] = Field(default_factory=list,
                                   description="Pairs students confuse: 'X is not Y: X ..., but Y ...'.")
    memory_tricks: list[MemoryTrick] = Field(default_factory=list)
    quick_check: list[QuickCheck] = Field(description="5-8 easy self-check questions.")
    top_three: list[str] = Field(description="If you only remember three things, remember these.")
    flashcards: list[Flashcard]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
EASY_PROMPT = """You write an EASY-TO-DIGEST version of a student's lecture: for someone who is still
confused, or cramming the night before an exam with no time to absorb everything.

VOICE
- Explain it like you would to a smart 12-year-old: short sentences, everyday words, friendly tone.
- Every technical word the exam needs is kept, but explained the moment it appears.
- One idea at a time. No walls of text. Each field should be quick to read.

CONTENT
- Cover every important idea in the notes, in lecture order, but only the core of each.
- `think_of_it_like`: a vivid everyday comparison that matches how the idea actually works.
  Analogies are the ONLY thing you may invent; every fact must still come from the notes.
- `from_your_notes`: reuse the notes' own tiny examples (values, code, cases) in simple words.
- `must_know`: what a teacher would most likely test, phrased as short facts.
- `dont_mix_up`: confusing pairs from the notes and the one-line difference.
- `memory_tricks`: 2-5 memorable tricks for facts that are easy to forget.
|- `worked_examples`: 1 simple worked example showing how to apply a key concept,
|  3-4 short steps with plain-language reasoning, an answer and a one-line takeaway.
|- `connections`: how the key ideas relate to each other, in plain words.
|- `key_terms`: 3-5 key words/phrases with plain one-sentence definitions each.
|- `quick_check`: 5-8 easy questions to confirm understanding, with short answers + "because".
- `top_three`: the 3 things that matter most.
- flashcards: 15-25 SIMPLE cards in plain words, answer <= 15 words, never the '#' character, no Markdown.

SCAFFOLDING
- Each concept MUST include `try_it`: a mini problem/scenario that uses the concept immediately
  after the explanation. 1-2 sentences: what to do, then the answer. Show how to apply the
  concept, not just what it is.
- The worked example should be a COMBINED problem that uses 2-3 concepts together, showing
  how they connect in one realistic scenario.

FORMAT: plain text; you may use **bold** for key words. Formulas in readable Unicode (Σ, ², √, →), never LaTeX.
Do not add facts that are not in the notes.
"""

ANALYSE_PROMPT = """You are an excellent tutor preparing to teach from ONE student's lecture notes.
Read the notes and every image carefully (images are often the most important part: formulas,
diagrams, recursion trees, code screenshots, tables). Analyse them; do not summarise them.

- Work out the central question the lecture answers and the logic that connects its parts.
- Notice what the notes emphasise (repeated, bold, boxed, warned about, worked through).
- Find where students will get stuck and why.
- List every concrete example, value, code fragment, proof step in the notes.
- For each image, transcribe what it shows exactly and say what it teaches.
- Use only what is in the notes and images. No outside facts. Note gaps instead of filling them.
"""

WRITE_PROMPT = """You write a study reviewer for ONE student from their lecture notes and a tutor's
analysis of them. The goal is insight and understanding, not a generic summary.

QUALITY BAR
- Insightful: explain WHY things work and HOW ideas connect, not just WHAT they are.
- Specific: every explanation, example and question uses the notes' own terms, examples, values,
  code and figures. If a sentence could appear in any textbook without reading these notes,
  rewrite it with the notes' specifics or delete it.
- Teach like a great tutor: plain words first, then precision. Short paragraphs.
- Worked examples: re-use the notes' own examples; each step says what is done AND why.
- Misconceptions: real confusions this material invites, with how to tell the difference.
- Coverage: one concept entry for every idea the notes teach (usually 6-12), in lecture order.
- Practice: 10-15 questions, mixing Remember / Apply / Analyze; at least half Apply or Analyze,
  built on the notes' examples (trace this, what happens if, compare, find the flaw).
  Every answer gets an explanation that teaches.
- Cite anchors like "Slide 12" or "Slide 8 image 1" in `source` fields.

GROUNDING
- Use only facts from the notes, the images and the analysis. Rewording and connecting ideas is
  encouraged; new facts, outside examples, names or dates are not. Note gaps with a suggested
  action for each one (check a chapter, ask the teacher, watch a recap).

FORMAT
- Plain text with light Markdown (**bold**, `code`) only. Formulas in readable Unicode text
  (Σ, ≤, ², √, log₂ n, →) — never LaTeX or $.
|- figures: include images that carry study content (not logos), using exact labels.
|  For each figure, add a question the student should answer before reading the insight.
- Formulas: only if this is a math/physics/science subject. For humanities,
  philosophy, history, or arts, omit the Formulas section entirely — those
  subjects rarely have them, and a formulas section hurts more than it helps.
|- key_terms: 5-10 key words/phrases from the notes with a one-sentence definition each.
|- flashcards: 20-40 cards, one atomic idea each, answer <= 25 words, never the '#' character,
  no Markdown. Prefer why / what-happens-if / compare / apply-to-the-notes'-example cards over
  bare "What is X?" cards.
"""


def _notes_block(filename: str, text: str, images: list[NoteImage], image_mode: str,
                 skipped: list[str] | None = None) -> str:
    skipped = skipped or []
    parts = [f"Source file: {filename}", f"Date: {date.today().isoformat()}", "",
             "NOTES (text extracted from the file):", "<<<NOTES", text or "(no text layer)", "NOTES>>>", ""]
    if images and image_mode == "attached":
        parts.append("IMAGES are attached after this text, each preceded by its label: "
                     + ", ".join(i.label for i in images) + ".")
    if skipped:
        parts.append("Pictures that could NOT be read (do not guess their content; list them in gaps): "
                     + "; ".join(skipped))
    if images and image_mode == "described":
        parts.append("IMAGE DESCRIPTIONS (from a vision model):")
        parts += [f"[{im.label}] {im.description}" for im in images if im.description]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
def _gemini_client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


def gemini_available_models(api_key: str) -> list[str]:
    try:
        names = {(m.name or "").removeprefix("models/") for m in _gemini_client(api_key).models.list()}
        return [m for m in GEMINI_MODELS if m in names] or GEMINI_MODELS
    except Exception:  # noqa: BLE001
        return GEMINI_MODELS


def _image_parts(images: list[NoteImage]):
    from google.genai import types

    parts = []
    for im in images:
        parts.append(types.Part.from_text(text=f"[{im.label}]"))
        parts.append(types.Part.from_bytes(data=im.data, mime_type=im.mime))
    return parts


def _gemini_json(api_key: str, models: list[str], system: str, contents, schema, temperature: float,
                 on_status: Callable[[str], None]):
    """Try models in order; skip retired/overloaded/quota-limited ones."""
    from google.genai import errors, types

    client = _gemini_client(api_key)
    last = None
    for model in models:
        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                        response_mime_type="application/json",
                        response_schema=schema,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                )
                parsed = resp.parsed if resp.parsed is not None else schema.model_validate_json(resp.text)
                return parsed, model
            except errors.APIError as e:
                last = e
                code = getattr(e, "code", None)
                if code == 503 and attempt == 0:
                    on_status("The AI service is busy, retrying…")
                    time.sleep(4)
                    continue
                if code in (404, 429, 503, 500):
                    on_status("Switching to a backup model…")
                    break
                raise
            except Exception as e:  # noqa: BLE001 — bad JSON etc.; try next model
                last = e
                break
    raise RuntimeError(f"All Gemini models failed. Last error: {last}")


def gemini_analyse(api_key: str, models: list[str], filename: str, text: str, images: list[NoteImage],
                   on_status: Callable[[str], None], skipped: list[str] | None = None) -> tuple[NotesAnalysis, str]:
    from google.genai import types

    on_status(f"Reading your notes and {len(images)} picture(s)…")
    contents = [types.Part.from_text(text=_notes_block(filename, text, images, "attached", skipped))] + _image_parts(images)
    return _gemini_json(api_key, models, ANALYSE_PROMPT, contents, NotesAnalysis, 0.2, on_status)


def _write_input(notes_block: str, analysis: NotesAnalysis | None, images: list[NoteImage]) -> str:
    s = notes_block
    if analysis is not None:
        s += ("\n\nTUTOR'S ANALYSIS OF THESE NOTES (including what each image shows):\n"
              + analysis.model_dump_json(indent=1))
    if images:
        s += "\n\nAvailable image labels for `figures`: " + ", ".join(i.label for i in images)
    return s


def gemini_write_full(api_key: str, models: list[str], notes_block: str, analysis: NotesAnalysis | None,
                      images: list[NoteImage], temperature: float, on_status: Callable[[str], None],
                      style: str = "") -> tuple[StudyGuide, str]:
    from google.genai import types

    on_status("Writing your full reviewer…")
    return _gemini_json(api_key, models, WRITE_PROMPT + style,
                        [types.Part.from_text(text=_write_input(notes_block, analysis, images))],
                        StudyGuide, temperature, on_status)


def gemini_write_easy(api_key: str, models: list[str], notes_block: str, analysis: NotesAnalysis | None,
                      temperature: float, on_status: Callable[[str], None], style: str = "") -> tuple[EasyGuide, str]:
    from google.genai import types

    on_status("Writing your easy version…")
    return _gemini_json(api_key, models, EASY_PROMPT + style,
                        [types.Part.from_text(text=_write_input(notes_block, analysis, []))],
                        EasyGuide, max(temperature, 0.4), on_status)


def notes_block(filename: str, text: str, images: list[NoteImage], skipped: list[str] | None = None) -> str:
    return _notes_block(filename, text, images, "none", skipped)


def gemini_study_guide(api_key: str, models: list[str], filename: str, text: str, images: list[NoteImage],
                         temperature: float, on_status: Callable[[str], None],
                         skipped: list[str] | None = None,
                         style: str = "") -> tuple[StudyGuide, NotesAnalysis, str]:
    analysis, used = gemini_analyse(api_key, models, filename, text, images, on_status, skipped)
    models2 = [used] + [m for m in models if m != used]
    guide, used2 = gemini_write_full(api_key, models2, notes_block(filename, text, images, skipped), analysis,
                                         images, temperature, on_status, style)
    return guide, analysis, used2


# ---------------------------------------------------------------------------
# Ollama (offline)
# ---------------------------------------------------------------------------
def _ollama():
    import ollama

    return ollama.Client(host=OLLAMA_HOST)


def ollama_models() -> tuple[list[str], list[str]]:
    """(all models, models that can see images)."""
    try:
        client = _ollama()
        names = [m.model for m in client.list().models]
        vision = []
        for n in names:
            try:
                if "vision" in (client.show(n).capabilities or []):
                    vision.append(n)
            except Exception:  # noqa: BLE001
                pass
        return names, vision
    except Exception:  # noqa: BLE001
        return [], []


def ollama_prewarm(model: str, num_ctx: int = 2048) -> None:
    """Keep model loaded in RAM so live generations don't suffer cold-load lag. Use the same num_ctx as
    the real generations, otherwise Ollama reloads the model on the first request."""
    try:
        client = _ollama()
        client.chat(model=model, messages=[{"role": "user", "content": "ping"}], keep_alive="24h",
                    options={"num_ctx": num_ctx, "num_predict": 1})
    except Exception:
        pass


def ollama_describe_images(model: str, images: list[NoteImage], on_status: Callable[[str], None]) -> None:
    client = _ollama()
    for n, im in enumerate(images, 1):
        on_status(f"Looking at picture {n} of {len(images)}…")
        r = client.chat(
            model=model,
            messages=[{"role": "user", "images": [im.data],
                       "content": "Transcribe all text/code/formulas in this image from study notes and "
                                  "explain any diagram. Be factual and brief."}],
            options={"temperature": 0.0, "num_ctx": 2048},
            keep_alive="24h",
        )
        im.description = r["message"]["content"].strip()


# Long notes are written section by section with small schemas (a 1.5B model on CPU is slow at long
# structured outputs and loses track of large inputs), then merged and given one short overview pass.
LOCAL_CHUNK_CHARS = 6_000
LOCAL_SECTION_TOKENS = 1_200
LOCAL_OVERVIEW_TOKENS = 500


class SectionGuide(BaseModel):
    # Field order is generation order: the must-haves come first so the token cap only trims extras.
    concepts: list[Concept]
    flashcards: list[Flashcard]
    practice: list[PracticeQuestion]
    key_terms: list[KeyTerm]
    cheat_sheet: list[str]
    misconceptions: list[Misconception]
    worked_examples: list[WorkedExample]
    formulas: list[Formula]
    gaps: list[Gap]


class GuideOverview(BaseModel):
    title: str
    subject: str = Field(description="Course/subject, e.g. 'Data Structures & Algorithms'.")
    big_picture: str = Field(description="3-5 sentences: what the lecture is about and how the pieces fit together.")
    learning_goals: list[str] = Field(description="'You should be able to ...' items.")
    connections: list[str] = Field(description="How the concepts relate; each one a full sentence.")


class EasySection(BaseModel):
    concepts: list[EasyConcept]
    flashcards: list[Flashcard]
    must_know: list[str]
    quick_check: list[QuickCheck]
    key_terms: list[KeyTerm]
    dont_mix_up: list[str]
    memory_tricks: list[MemoryTrick]
    worked_examples: list[EasyWorkedExample]


class EasyOverview(BaseModel):
    title: str
    one_sentence: str = Field(description="The whole lecture in one simple sentence.")
    story: str = Field(description="The big picture as a short, friendly walkthrough, 3-5 short sentences.")
    connections: list[str] = Field(description="How the key ideas relate, in plain words.")
    top_three: list[str] = Field(description="If you only remember three things, remember these.")


SECTION_CAPS = dict(concepts=3, flashcards=5, practice=2, key_terms=3, cheat_sheet=3, misconceptions=2,
                    worked_examples=1, formulas=2, gaps=2)
GUIDE_OVERVIEW_CAPS = dict(learning_goals=5, connections=5)
EASY_SECTION_CAPS = dict(concepts=3, flashcards=5, must_know=3, quick_check=2, key_terms=3, dont_mix_up=2,
                         memory_tricks=2, worked_examples=1)
EASY_OVERVIEW_CAPS = dict(connections=4, top_three=3)

SECTION_PROMPT = """You write ONE SECTION of a study reviewer from one part of a student's lecture notes.
Use only facts from this part of the notes; reuse their own terms, examples, values and code.
Explain WHY and HOW in plain words. Keep every field short.
- concepts: up to 3 ideas this part teaches, in order. Each: explanation, why it matters, an example
  from the notes ('' if none), the mistake students make ('' if none), source anchor.
- key_terms: up to 3 terms with one-sentence definitions.
- worked_examples: at most 1, only if the notes work through a problem or example; otherwise [].
- formulas: only for math/science content; otherwise [].
- misconceptions: up to 2 real confusions this part invites; otherwise [].
- practice: up to 2 questions (Remember / Apply / Analyze) with answer and a short explanation.
- cheat_sheet: up to 3 one-line facts.
- gaps: things mentioned but not explained, with what to do; otherwise [].
- flashcards: up to 5, one idea each, answer <= 25 words, plain text, never the '#' character.
Cite anchors like "Page 3" or "Slide 5" in source fields. Plain text; formulas in Unicode, never LaTeX.
"""

GUIDE_OVERVIEW_PROMPT = """You are given the list of concepts a study reviewer covers and the start of the
student's notes. Write the reviewer's title, subject, a short big picture (3-5 sentences), up to 5
"You should be able to ..." learning goals and up to 5 sentences on how the concepts connect.
Use only what is given. Plain text.
"""

EASY_SECTION_PROMPT = """You write ONE SECTION of an EASY-TO-DIGEST reviewer from one part of a student's
lecture notes, for someone who is confused or cramming. Short sentences, everyday words, friendly tone;
explain every technical word the moment it appears. Use only facts from this part of the notes
(analogies are the only thing you may invent). Keep every field short.
- concepts: up to 3 ideas, in order. Each: plain words, an everyday comparison, a tiny example from the
  notes ('' if none), one line to remember, and a 1-2 sentence try-it scenario with its answer.
- key_terms: up to 3 terms with plain definitions.
- worked_examples: at most 1 simple example with 3-4 short steps; otherwise [].
- must_know: up to 3 short facts most likely on the exam.
- dont_mix_up: up to 2 confusing pairs, 'X is not Y: X ..., but Y ...'; otherwise [].
- memory_tricks: up to 2 mnemonics or vivid images.
- quick_check: up to 2 easy questions with a short answer plus 'because ...'.
- flashcards: up to 5 simple cards, answer <= 15 words, plain text, never the '#' character.
"""

EASY_OVERVIEW_PROMPT = """You are given the list of ideas an easy study reviewer covers and the start of the
student's notes. Write a title, the whole lecture in one simple sentence, a short friendly walkthrough
(3-5 short sentences), up to 4 plain-words connections between the ideas, and the top 3 things to
remember. Use only what is given. Short sentences, everyday words, plain text.
"""


def split_notes(text: str, size: int = LOCAL_CHUNK_CHARS) -> list[str]:
    """Split notes into chunks of at most ~size chars at page/slide/paragraph boundaries. A chunk that
    starts mid-page is told which page it continues so its source anchors stay right."""
    text = (text or "").strip()
    if len(text) <= size:
        return [text]
    blocks: list[str] = []
    for block in re.split(r"\n\s*\n|\n(?=--- (?:Page|Slide) \d+ ---|===== FILE: )", text):
        block = block.strip()
        while len(block) > size:
            cut = block.rfind("\n", 0, size)
            if cut < size // 2:
                cut = block.rfind(" ", 0, size)
            if cut < size // 2:
                cut = size
            blocks.append(block[:cut].strip())
            block = block[cut:].strip()
        if block:
            blocks.append(block)
    chunks: list[str] = []
    cur = ""
    for block in blocks:
        if cur and len(cur) + 2 + len(block) > size:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n\n{block}" if cur else block
    if cur:
        chunks.append(cur)
    anchor = re.compile(r"--- (?:Page|Slide) \d+ ---")
    for i in range(1, len(chunks)):
        prev = anchor.findall(chunks[i - 1])
        if prev and not chunks[i].startswith("---"):
            chunks[i] = f"(continued from {prev[-1].strip('- ')})\n{chunks[i]}"
    return chunks


def _capped_schema(cls: type[BaseModel], caps: dict[str, int]) -> dict:
    schema = cls.model_json_schema()
    for field, n in caps.items():
        schema["properties"][field]["maxItems"] = n
    return schema


def _close_json(raw: str) -> dict:
    """Parse JSON that may have been cut off mid-way (output token cap): keep everything up to the last
    completed object/array and close the open brackets."""
    s = raw[raw.find("{"):] if "{" in raw else ""
    stack: list[str] = []
    in_str = esc = False
    last_ok: tuple[int, list[str]] | None = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
            if not stack:
                return json.loads(s[: i + 1])
            last_ok = (i, list(stack))
    if last_ok is None:
        return {}
    i, still_open = last_ok
    return json.loads(s[: i + 1] + "".join(reversed(still_open)))


def _validate_lenient(cls: type[BaseModel], data: dict):
    """Fill missing top-level fields and drop list items that are incomplete or malformed."""
    data = dict(data) if isinstance(data, dict) else {}
    for name, info in cls.model_fields.items():
        if name not in data and info.is_required():
            data[name] = "" if info.annotation is str else []
    for _ in range(100):
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            bad: dict[str, set[int]] = {}
            for err in e.errors():
                loc = err["loc"]
                field = cls.model_fields.get(str(loc[0])) if loc else None
                if len(loc) >= 2 and isinstance(loc[1], int) and isinstance(data.get(loc[0]), list):
                    bad.setdefault(loc[0], set()).add(loc[1])
                elif field is not None:
                    data[loc[0]] = "" if field.annotation is str else []
                else:
                    raise
            for name, idx in bad.items():
                data[name] = [v for i, v in enumerate(data[name]) if i not in idx]
    return cls.model_validate(data)


def _parse_model(cls: type[BaseModel], raw: str):
    try:
        return cls.model_validate_json(raw)
    except Exception:  # noqa: BLE001 — small models leave fields out or get cut off by the token cap
        try:
            data = _close_json(raw)
        except Exception:  # noqa: BLE001
            data = {}
        return _validate_lenient(cls, data)


def _ollama_json(model: str, system: str, user: str, cls: type[BaseModel], caps: dict[str, int],
                 temperature: float, num_ctx: int, num_predict: int):
    r = _ollama().chat(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        format=_capped_schema(cls, caps),
        options={"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict},
        keep_alive="24h",
    )
    return _parse_model(cls, r["message"]["content"])


def _dedupe(items: list, key: Callable[[object], str]) -> list:
    seen: set[str] = set()
    out = []
    for it in items:
        k = re.sub(r"\W+", "", key(it).lower())
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out


def _overview_input(filename: str, ideas: list[str], text: str) -> str:
    listed = "\n".join(f"- {i}" for i in ideas)[:3_500]
    return (f"Source file: {filename}\n\nIDEAS COVERED (in lecture order):\n{listed}\n\n"
            f"START OF THE NOTES:\n<<<NOTES\n{text[:1_500]}\nNOTES>>>\n\nReturn the JSON.")


def _write_sections(model: str, filename: str, text: str, images: list[NoteImage], system: str,
                    cls: type[BaseModel], caps: dict[str, int], temperature: float, num_ctx: int,
                    on_status: Callable[[str], None], what: str) -> list:
    chunks = split_notes(text)
    mode = "described" if any(i.description for i in images) else "none"
    sections = []
    for n, chunk in enumerate(chunks, 1):
        on_status(f"Writing {what} on this computer — section {n} of {len(chunks)}…")
        part = "" if len(chunks) == 1 else f"These notes are part {n} of {len(chunks)} of the file.\n"
        user = _notes_block(filename, chunk, images if n == 1 else [], mode) + f"\n{part}\nReturn the section JSON."
        sections.append(_ollama_json(model, system, user, cls, caps, temperature, num_ctx, LOCAL_SECTION_TOKENS))
    return sections


def ollama_easy_guide(model: str, filename: str, text: str, temperature: float, num_ctx: int,
                      on_status: Callable[[str], None]) -> EasyGuide:
    temperature = max(temperature, 0.4)
    secs = _write_sections(model, filename, text, [], EASY_SECTION_PROMPT, EasySection, EASY_SECTION_CAPS,
                           temperature, num_ctx, on_status, "your easy version")
    concepts = _dedupe([c for s in secs for c in s.concepts], lambda c: c.name)
    on_status("Pulling the sections together…")
    ov = _ollama_json(model, EASY_OVERVIEW_PROMPT,
                      _overview_input(filename, [f"{c.name}: {c.in_plain_words[:160]}" for c in concepts], text),
                      EasyOverview, EASY_OVERVIEW_CAPS, temperature, num_ctx, LOCAL_OVERVIEW_TOKENS)
    return EasyGuide(
        title=ov.title or filename, one_sentence=ov.one_sentence, story=ov.story, concepts=concepts,
        connections=ov.connections,
        key_terms=_dedupe([k for s in secs for k in s.key_terms], lambda k: k.term),
        worked_examples=[w for s in secs for w in s.worked_examples][:2],
        must_know=_dedupe([m for s in secs for m in s.must_know], str),
        dont_mix_up=_dedupe([d for s in secs for d in s.dont_mix_up], str),
        memory_tricks=_dedupe([m for s in secs for m in s.memory_tricks], lambda m: m.trick),
        quick_check=_dedupe([q for s in secs for q in s.quick_check], lambda q: q.question),
        top_three=ov.top_three,
        flashcards=[f for s in secs for f in s.flashcards],
    )


def ollama_study_guide(model: str, filename: str, text: str, images: list[NoteImage], temperature: float,
                       num_ctx: int, on_status: Callable[[str], None]) -> StudyGuide:
    secs = _write_sections(model, filename, text, images, SECTION_PROMPT, SectionGuide, SECTION_CAPS,
                           temperature, num_ctx, on_status, "your reviewer")
    concepts = _dedupe([c for s in secs for c in s.concepts], lambda c: c.name)
    on_status("Pulling the sections together…")
    ov = _ollama_json(model, GUIDE_OVERVIEW_PROMPT,
                      _overview_input(filename, [f"{c.name}: {c.explanation[:160]}" for c in concepts], text),
                      GuideOverview, GUIDE_OVERVIEW_CAPS, temperature, num_ctx, LOCAL_OVERVIEW_TOKENS)
    return StudyGuide(
        title=ov.title or filename, subject=ov.subject, big_picture=ov.big_picture,
        learning_goals=ov.learning_goals, concepts=concepts,
        key_terms=_dedupe([k for s in secs for k in s.key_terms], lambda k: k.term),
        connections=ov.connections,
        worked_examples=[w for s in secs for w in s.worked_examples],
        formulas=_dedupe([f for s in secs for f in s.formulas], lambda f: f.name),
        misconceptions=[m for s in secs for m in s.misconceptions],
        practice=[p for s in secs for p in s.practice],
        cheat_sheet=_dedupe([c for s in secs for c in s.cheat_sheet], str),
        gaps=_dedupe([g for s in secs for g in s.gaps], lambda g: g.term),
        flashcards=[f for s in secs for f in s.flashcards],
    )
