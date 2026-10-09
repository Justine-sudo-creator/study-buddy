"""Offline active-recall tutor: page-anchored passages, short local-model calls for questions and grading,
and a simple review queue that brings missed questions back."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

from pydantic import BaseModel

from study.engines import _ollama, _parse_model

QUIZ_NUM_CTX = 4096
QUESTION_TOKENS = 220
GRADE_TOKENS = 70
PASSAGE_CHARS = 1_500
MIN_PASSAGE_CHARS = 200
BUFFER_TARGET = 6
REVIEW_GAP = 3

_ANCHOR = re.compile(r"^--- ((?:Page|Slide) \d+) ---$|^===== FILE: (.+?) =====$", re.M)


@dataclass
class Passage:
    id: int
    where: str
    text: str


@dataclass
class Question:
    id: int
    question: str
    answer: str
    passage: Passage


class _QuizQuestion(BaseModel):
    question: str
    answer: str


class _QuizBatch(BaseModel):
    questions: list[_QuizQuestion]


QUESTION_PROMPT = """You write quiz questions that check whether a student understood a passage from their own
notes. Write 3 questions. Ask about causes, differences, processes and meanings, not trivia like dates or
names unless they matter. Each answer must come straight from the passage and be under 15 words. Never
ask about something the passage does not say."""

GRADE_PROMPT = """You grade a student's answer to a quiz question using ONLY the notes given. Accept answers
in the student's own words if they mean the same thing as the expected answer. Reply in exactly this format:
VERDICT: correct | partly correct | incorrect
HINT: one short sentence (under 25 words) to the student, using "you", saying what was right or missing
according to the notes."""


def passages(text: str, size: int = PASSAGE_CHARS) -> list[Passage]:
    """Cut notes into short passages that each remember which page/slide (and file) they came from."""
    out: list[Passage] = []
    file = where = ""
    pos = 0
    pieces: list[tuple[str, str]] = []
    for m in _ANCHOR.finditer(text):
        pieces.append((file, where, text[pos:m.start()]))
        if m.group(2):
            file, where = m.group(2), ""
        else:
            where = m.group(1)
        pos = m.end()
    pieces.append((file, where, text[pos:]))
    for file, where, body in pieces:
        label = f"{file}: {where}" if file and where else (where or file)
        cur = ""
        for para in re.split(r"\n\s*\n", body):
            para = para.strip()
            while len(para) > size:
                cut = para.rfind(". ", 0, size)
                cut = cut + 1 if cut > size // 2 else size
                if cur:
                    out.append(Passage(0, label, cur))
                    cur = ""
                out.append(Passage(0, label, para[:cut].strip()))
                para = para[cut:].strip()
            if cur and len(cur) + len(para) + 2 > size:
                out.append(Passage(0, label, cur))
                cur = ""
            cur = f"{cur}\n\n{para}" if cur else para
        if cur:
            out.append(Passage(0, label, cur))
    merged: list[Passage] = []
    for p in out:
        if merged and merged[-1].where == p.where and len(merged[-1].text) < MIN_PASSAGE_CHARS:
            merged[-1].text = f"{merged[-1].text}\n\n{p.text}"
        elif p.text.strip():
            merged.append(p)
    kept = [p for p in merged if len(p.text) >= MIN_PASSAGE_CHARS] or merged
    for i, p in enumerate(kept):
        p.id = i
    return kept


def _schema() -> dict:
    schema = _QuizBatch.model_json_schema()
    schema["properties"]["questions"]["maxItems"] = 3
    props = schema["$defs"]["_QuizQuestion"]["properties"]
    props["question"]["maxLength"] = 160
    props["answer"]["maxLength"] = 100
    return schema


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def write_questions(model: str, passage: Passage,
                    cancelled: Callable[[], bool] = lambda: False) -> list[tuple[str, str]] | None:
    """Questions for one passage, or None if `cancelled()` became true mid-way (the request is dropped so
    the model is free at once)."""
    stream = _ollama().chat(
        model=model,
        stream=True,
        messages=[{"role": "system", "content": QUESTION_PROMPT},
                  {"role": "user", "content": f"PASSAGE ({passage.where or 'notes'}):\n{passage.text}"}],
        format=_schema(),
        options={"temperature": 0.3, "num_ctx": QUIZ_NUM_CTX, "num_predict": QUESTION_TOKENS},
        keep_alive="24h",
    )
    raw = ""
    for chunk in stream:
        if cancelled():
            stream.close()
            return None
        raw += chunk["message"]["content"]
    batch = _parse_model(_QuizBatch, raw)
    out = []
    for q in batch.questions:
        qq, aa = q.question.strip(), q.answer.strip()
        new_words = _words(aa) - _words(qq)
        answer_words = _words(aa)
        restates = len(answer_words) > 4 and len(new_words) / len(answer_words) < 0.35
        if len(qq) > 10 and qq.endswith("?") and new_words and not restates and len(aa.split()) <= 25:
            out.append((qq, aa))
    return out


_STOP = set("a an the of to in on and or is are was were be by for with as at it its that this from into "
            "can will which their they them than then also".split())
_GIVE_UP = {"", "idk", "i don t know", "dont know", "don t know", "no idea", "pass", "skip", "not sure"}


def _words(s: str) -> set[str]:
    return {w for w in _norm(s).split() if w not in _STOP}


def quick_verdict(student: str, expected: str) -> str | None:
    """Decide obvious cases without the model: blank/give-up answers and near-copies of the answer."""
    s = _norm(student)
    if s in _GIVE_UP or len(s) < 2:
        return "incorrect"
    exp = _words(expected)
    got = _words(student)
    if exp and len(exp & got) / len(exp) >= 0.8 and len(got) <= 2 * len(exp) + 3:
        return "correct"
    return None


def grade_stream(model: str, q: Question, student: str) -> Iterator[str]:
    stream = _ollama().chat(
        model=model,
        stream=True,
        messages=[{"role": "system", "content": GRADE_PROMPT},
                  {"role": "user", "content": f"NOTES ({q.passage.where or 'notes'}):\n{q.passage.text}\n\n"
                                              f"QUESTION: {q.question}\nEXPECTED ANSWER: {q.answer}\n"
                                              f"STUDENT ANSWER: {student}"}],
        options={"temperature": 0.1, "num_ctx": QUIZ_NUM_CTX, "num_predict": GRADE_TOKENS},
        keep_alive="24h",
    )
    for chunk in stream:
        yield chunk["message"]["content"]


def parse_grade(raw: str) -> tuple[str, str]:
    m = re.search(r"VERDICT:\s*(partly correct|partially correct|incorrect|correct|partly|wrong)", raw, re.I)
    v = (m.group(1).lower() if m else "")
    verdict = ("partly" if v.startswith("part") else "incorrect" if v in ("incorrect", "wrong")
               else "correct" if v == "correct" else "partly")
    h = re.search(r"HINT:\s*(.+)", raw, re.S | re.I)
    hint = h.group(1).strip() if h else re.sub(r"VERDICT:.*", "", raw).strip()
    hint = re.split(r"\n\s*\n", hint)[0].strip()
    hint = " ".join(re.split(r"(?<=[.!?])\s+", hint)[:2])
    return verdict, hint


@dataclass
class Quiz:
    """Questions plus the order to ask them in. A background thread writes questions a few at a time,
    and drops/pauses its request whenever a grade is waiting, so grading always gets the model first."""

    model: str
    passages: list[Passage]
    questions: list[Question] = field(default_factory=list)
    queue: list[int] = field(default_factory=list)
    done: set[int] = field(default_factory=set)
    attempts: dict[int, list[str]] = field(default_factory=dict)
    next_passage: int = 0
    error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)
    model_lock: threading.Lock = field(default_factory=threading.Lock)
    grading: threading.Event = field(default_factory=threading.Event)
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    writing: bool = False
    first_ready_s: float | None = None
    started: float = field(default_factory=time.time)
    _seen: set[str] = field(default_factory=set)

    @property
    def finished_writing(self) -> bool:
        return self.next_passage >= len(self.passages)

    def start(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set() and not self.finished_writing:
            if self.grading.is_set() or len(self.queue) >= BUFFER_TARGET:
                time.sleep(0.3)
                continue
            p = self.passages[self.next_passage]
            self.writing = True
            try:
                with self.model_lock:
                    pairs = write_questions(self.model, p, lambda: self.grading.is_set() or self.stop.is_set())
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
                self.writing = False
                return
            self.writing = False
            if pairs is None:
                continue
            self.next_passage += 1
            self._add(p, pairs)

    def _add(self, p: Passage, pairs: list[tuple[str, str]]) -> None:
        with self.lock:
            for qq, aa in pairs:
                key = _norm(qq)
                if key in self._seen:
                    continue
                self._seen.add(key)
                q = Question(len(self.questions), qq, aa, p)
                self.questions.append(q)
                self.queue.append(q.id)
            if self.questions and self.first_ready_s is None:
                self.first_ready_s = time.time() - self.started

    def current(self) -> Question | None:
        with self.lock:
            return self.questions[self.queue[0]] if self.queue else None

    def record(self, qid: int, verdict: str) -> None:
        """Correct answers leave the queue; others come back after a few other questions."""
        with self.lock:
            self.attempts.setdefault(qid, []).append(verdict)
            if self.queue and self.queue[0] == qid:
                self.queue.pop(0)
            if verdict == "correct":
                self.done.add(qid)
            else:
                self.queue.insert(min(REVIEW_GAP, len(self.queue)), qid)

    def grade(self, q: Question, student: str, on_text: Callable[[str], None]) -> tuple[str, str]:
        self.grading.set()
        try:
            with self.model_lock:
                raw = ""
                for piece in grade_stream(self.model, q, student):
                    raw += piece
                    on_text(raw)
        finally:
            self.grading.clear()
        return parse_grade(raw)

    def stats(self) -> dict:
        with self.lock:
            first_try = [v[0] for v in self.attempts.values()]
            return dict(
                ready=len(self.questions),
                answered=len(self.attempts),
                mastered=len(self.done),
                first_try_correct=sum(v == "correct" for v in first_try),
                to_review=sum(1 for qid in self.queue if qid in self.attempts),
            )

    def missed(self) -> list[Question]:
        with self.lock:
            return [self.questions[i] for i, v in self.attempts.items() if any(x != "correct" for x in v)]
