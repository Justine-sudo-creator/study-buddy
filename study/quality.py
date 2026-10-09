"""Quality gate for flashcards.

1. Local, free checks (always on): format, '#', length, duplicates, placeholder labels.
2. Optional Jev (TypeSafe AI) check: Jev is a "System One" decision model — it can't write
   text, but it returns calibrated yes/no probabilities fast. We ask it, per card:
     grounded  — is the answer supported by the notes?
     specific  — does the card test something specific to these notes (not generic trivia)?
   Cards below the thresholds are dropped (and shown in the app as rejected).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import requests

from .engines import Flashcard

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_STATE_CHARS = 60_000      # ~15k tokens; Jev allows 32k tokens for state + longest question
_PLACEHOLDER_Q = {"question", "q", "term", "front", "card"}


@dataclass
class CardVerdict:
    card: Flashcard
    ok: bool
    reason: str = ""
    grounded: float | None = None
    specific: float | None = None


def _clean(s: str) -> str:
    s = s.replace("#", " number ").replace("$", "").replace("**", "").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def local_check(cards: list[Flashcard]) -> list[CardVerdict]:
    seen: set[str] = set()
    out = []
    for c in cards:
        q, a = _clean(c.question), _clean(c.answer)
        c = Flashcard(question=q, answer=a, source=c.source.strip())
        key = re.sub(r"\W+", "", q.lower())
        if not q or not a:
            out.append(CardVerdict(c, False, "Empty card"))
        elif q.rstrip(":").lower() in _PLACEHOLDER_Q or re.fullmatch(r"(question|q)\s*\d*", q.lower()):
            out.append(CardVerdict(c, False, "Not a real question"))
        elif key in seen:
            out.append(CardVerdict(c, False, "Repeats another card"))
        elif len(a.split()) > 40:
            out.append(CardVerdict(c, False, "Answer too long to memorise"))
        else:
            seen.add(key)
            out.append(CardVerdict(c, True))
    return out


def jev_check(api_key: str, notes_text: str, figures_text: str, verdicts: list[CardVerdict],
              min_grounded: float, min_specific: float, timeout: int = 60) -> str:
    """Mutates verdicts in place. Returns a short status string for the UI."""
    live = [v for v in verdicts if v.ok]
    if not live:
        return "Jev: no cards to check"
    state = {"notes": notes_text[:JEV_STATE_CHARS]}
    if figures_text:
        state["figures"] = figures_text[:8000]
    questions = {}
    for i, v in enumerate(live):
        card = {"question": v.card.question, "answer": v.card.answer}
        questions[f"g{i}"] = {
            "type": "noul",
            "instructions": {
                "card": card,
                "question": "Is the `card` answer correct and supported by the `notes` (or `figures`)?",
            },
        }
        questions[f"s{i}"] = {
            "type": "noul",
            "instructions": {
                "card": card,
                "question": ("Does the `card` test a specific term, example, value, step or argument that "
                             "appears in the `notes`, rather than generic knowledge any course would ask?"),
            },
        }
    r = requests.post(
        JEV_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"state": state, "model": JEV_MODEL, "questions": questions},
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Jev HTTP {r.status_code}: {r.text[:300]}")
    answers = r.json().get("answers", {})
    dropped = 0
    for i, v in enumerate(live):
        g = answers.get(f"g{i}", {}).get("noul")
        s = answers.get(f"s{i}", {}).get("noul")
        v.grounded, v.specific = g, s
        if g is not None and g < min_grounded:
            v.ok, v.reason = False, "Not clearly supported by your notes"
            dropped += 1
        elif s is not None and s < min_specific:
            v.ok, v.reason = False, "Too general — not specific to your notes"
            dropped += 1
    return f"Checked {len(live)} cards, removed {dropped} weak ones"


def to_quizlet(verdicts: list[CardVerdict]) -> str:
    return "\n".join(f"{v.card.question} # {v.card.answer}" for v in verdicts if v.ok)


def jev_key() -> str:
    return os.environ.get("TYPESAFE_API_KEY", "")
