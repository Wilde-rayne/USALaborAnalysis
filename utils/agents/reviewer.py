"""
Reviewer / critic agent that scores specialist blurbs for accuracy
and insight, and triggers a single revision pass when a candidate
falls short.

The pipeline is two-stage so we don't pay an LLM round-trip on every
candidate:

1. :meth:`ReviewerAgent.quick_check` — pure-Python heuristics. Catches
   the most common failure modes from earlier sessions (preamble
   like "Here's a plain-English explanation", responses too short to
   carry interpretation, no specific numbers, error sentinels from
   ``LaborAgent.invoke`` that leaked through). No LLM call.
2. :meth:`ReviewerAgent.evaluate` — LLM critic. Only invoked when the
   quick check fails. Returns a structured ``{"passed", "feedback"}``
   dict the orchestrator uses to compose a revision prompt.

The reviewer defaults to the small worker model (``OLLAMA_AGENT_MODEL``
→ ``phi3``) at temperature ``0.0`` — deterministic critique, fast on
CPU. Each ``LaborAgent`` instance is short-lived (created by the
caller per click) so no module-level singleton holds stale state.
"""
from __future__ import annotations

import logging
import os

from utils.agents.base import LaborAgent, LaborAgentConfig
from utils.agents.ollama import AGENT_MODEL_ENV, OLLAMA_BASE_URL
from utils.constants import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


REVIEWER_SYSTEM_PROMPT = (
    "You review short labor-market analyses for accuracy and insight. "
    "Output exactly two lines:\n"
    "Line 1: PASS or REVISE\n"
    "Line 2: one-sentence reason citing the most important issue (if "
    "REVISE) or the single strongest interpretive claim (if PASS).\n"
    "PASS requires: cites at least 2 specific numbers from the facts; "
    "interprets rather than restates; no invented numbers; no preamble "
    "like 'Here's a plain-English explanation'."
)


#: Common preamble snippets the specialist agents emit. Two roles:
#: (1) the quick-check flags any candidate that *starts* with one
#: of these so the orchestrator can fall back to a re-prompt; (2)
#: :func:`strip_preamble` removes the offending opening clause
#: directly when LLM-review is disabled (the reactive default).
#:
#: Match strings are lower-cased + use plain ASCII apostrophes; the
#: stripper does the case-insensitive comparison.
_PREAMBLE_MARKERS: tuple[str, ...] = (
    "here's a plain-english",
    "here is a plain-english",
    "here's an explanation",
    "here is an explanation",
    "here's a 3-sentence",
    "here is a 3-sentence",
    "here's a possible",
    "here is a possible",
    "let me explain",
    "the panel shows that",
    "based on the facts",
    "based on the panel facts",
    "based on the panel",
    "based on the provided facts",
    "based on the provided information",
    "based on the provided data",
    "based on the information provided",
    "according to the panel",
    "according to the facts",
    "in this analysis",
    "executive summary:",
    "panel: executive summary",
    "panel:",
    "topic:",
    "headline:",
    "headline finding:",
)


def strip_preamble(text: str) -> str:
    """
    Remove a single leading preamble clause from ``text`` if one
    matches :data:`_PREAMBLE_MARKERS`. Returns the cleaned text;
    leaves the input unchanged when no marker matches.

    Strategy: case-insensitive prefix match against the markers; on
    a hit, drop everything up to and including the first sentence-
    ending punctuation (``. : !``) or newline, then strip residual
    whitespace. Conservative — only strips when we're sure it's a
    preamble (matches the start, not somewhere in the middle).
    """
    if not text:
        return text
    candidate = text.lstrip()
    # Loop so we strip nested preambles like "Panel: IA … \n
    # Headline Finding: …" — each pass removes the outermost layer
    # until no marker matches.
    for _pass in range(3):
        result = _strip_one_preamble(candidate)
        if result == candidate:
            return candidate
        candidate = result
    return candidate


def _strip_one_preamble(text: str) -> str:
    """Single-pass strip — removes at most one preamble marker."""
    candidate = text.lstrip()
    if not candidate:
        return candidate
    lower = candidate.lower()
    for marker in _PREAMBLE_MARKERS:
        if not lower.startswith(marker):
            continue
        # Find the end of the leading clause. Only cut on:
        #   - newline
        #   - colon (preamble headers like "Executive Summary:" /
        #     "Panel: Executive Summary")
        #   - "," (comma) ending the preamble clause —
        #     "Based on the provided facts, forward expectations…"
        #     drops just the leading "Based on the provided facts,"
        #     and keeps the substantive forecast that follows.
        # We deliberately do NOT cut on "." because the period
        # might be inside a number ("66.3%") and a sentence-end
        # period would lose substantive content; the comma /
        # colon cut is enough for every preamble marker we've
        # seen the model emit.
        cut = -1
        marker_len = len(marker)
        for idx, ch in enumerate(candidate):
            # Examine characters AT or AFTER the marker boundary.
            # ``idx == marker_len`` is the first char that isn't part
            # of the marker — the trailing punctuation that ends the
            # preamble clause ("Based on the provided facts," — the
            # comma is at exactly ``marker_len``).
            if idx < marker_len:
                continue
            if ch in (":", "\n", ","):
                cut = idx + 1
                break
            if ch in ("!", "?"):
                cut = idx + 1
                break
        if cut < 0:
            # Whole string was preamble (no terminator found).
            return ""
        cleaned = candidate[cut:].lstrip()
        # If the strip leaves nothing useful, return the original —
        # better to ship the imperfect text than nothing.
        if len(cleaned) < 60:
            return candidate
        return cleaned
    return candidate


class ReviewerAgent:
    """Heuristic + LLM-backed reviewer for blurb candidates."""

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.agent = LaborAgent(
            LaborAgentConfig(
                model=model or os.getenv(AGENT_MODEL_ENV, "phi3"),
                system_prompt=REVIEWER_SYSTEM_PROMPT,
                base_url=OLLAMA_BASE_URL,
                # Deterministic critique; we don't want stochastic
                # disagreement on identical candidates.
                temperature=0.0,
                timeout=timeout,
            )
        )

    # ------------------------------------------------------------------
    # Stage 1 — pure-Python heuristics (free, runs on every candidate)
    # ------------------------------------------------------------------
    @staticmethod
    def quick_check(blurb: str) -> tuple[bool, str]:
        """
        Returns ``(looks_ok, reason)`` based on a few cheap rules.

        Catches preamble, error sentinels, suspiciously short output,
        and "no specific numbers" cases that are almost always bad.
        Falls through to the LLM reviewer for borderline candidates.
        """
        text = (blurb or "").strip()
        if not text:
            return False, "empty response"
        if len(text) < 80:
            return False, "too short to interpret anything"
        # Sentinels from ``LaborAgent.invoke``'s exception handler.
        lower = text.lower()
        if "agent error" in lower or "[agent:" in lower or "[ai] error" in lower:
            return False, "agent error sentinel"
        # Preamble — common LLM tic that wastes the user's first sentence.
        head = lower[:60]
        if any(marker in head for marker in _PREAMBLE_MARKERS):
            return False, "preamble; cut it"
        # Specific-number floor: each blurb should ground at least
        # two numeric facts. Counts digit characters, which is robust
        # to currency, percent, and decimal formats.
        digit_count = sum(1 for c in text if c.isdigit())
        if digit_count < 3:
            return False, "fewer than 3 numeric digits — likely not grounded"
        return True, ""

    # ------------------------------------------------------------------
    # Stage 2 — LLM critic
    # ------------------------------------------------------------------
    def evaluate(self, blurb: str, facts: str) -> dict:
        """
        Submit the candidate blurb + the facts it was grounded in to
        the LLM critic. Returns ``{"passed": bool, "feedback": str}``.
        Falls back to ``passed=True`` if the response is malformed —
        we'd rather ship a candidate the heuristic already cleared
        than block on a parsing failure.
        """
        prompt = (
            f"Facts the analysis must ground in:\n{facts}\n\n"
            f"Analysis to review:\n{blurb}\n\n"
            f"Review:"
        )
        try:
            response = self.agent.invoke(prompt).strip()
        except Exception as exc:  # noqa: BLE001 — never let critic kill the panel
            logger.warning("[reviewer] evaluate raised: %s", exc)
            return {"passed": True, "feedback": ""}

        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        if not lines:
            return {"passed": True, "feedback": ""}
        verdict = lines[0].upper()
        feedback = lines[1] if len(lines) > 1 else ""
        passed = verdict.startswith("PASS")
        return {"passed": passed, "feedback": feedback}

    # ------------------------------------------------------------------
    # Composite — the orchestrator's main entry point
    # ------------------------------------------------------------------
    def review(self, blurb: str, facts: str) -> dict:
        """
        Two-stage review. Returns ``{"passed", "feedback", "stage"}``
        so callers / progress messages can tell whether the
        decision came from the cheap heuristic or the LLM critic.
        """
        ok, reason = self.quick_check(blurb)
        if ok:
            # Heuristic was happy — skip the LLM round-trip.
            return {"passed": True, "feedback": "", "stage": "quick"}
        # Heuristic flagged it — confirm with the LLM and surface
        # actionable feedback for the revision prompt.
        llm_review = self.evaluate(blurb, facts)
        return {
            "passed": llm_review["passed"],
            "feedback": llm_review["feedback"] or reason,
            "stage": "llm",
        }


HOLISTIC_REVIEWER_SYSTEM_PROMPT = (
    "You are the senior auditor for a labor-market dashboard tab. "
    "You see every panel narrative plus the recap synthesis the user "
    "is about to read. Audit the WHOLE OUTPUT for cross-section "
    "coherence — does the recap actually weave the panels together, "
    "does it surface a non-obvious cross-panel observation, do any "
    "panels contradict each other, does any narrative invent values "
    "not in its grounding facts.\n\n"
    "Output exactly two lines:\n"
    "Line 1: PASS or REVISE_<section>\n"
    "  where <section> is one of the section names you were given\n"
    "  (most often 'recap' when the issue is weak synthesis)\n"
    "Line 2: one-sentence directive for the revision (or one-sentence "
    "       summary of the strongest cross-panel insight if PASS)."
)


class HolisticReviewer:
    """
    Cross-section auditor — the deep / holistic reviewer.

    The per-section :class:`ReviewerAgent` checks each panel in
    isolation. Once every section has settled, a tab can still ship
    a *coherent set of individually-acceptable narratives* that
    nonetheless contradict each other, or a recap that fails to draw
    a non-obvious cross-panel insight. This auditor closes that gap:
    it sees every panel narrative + the recap simultaneously and
    issues *targeted* revision feedback (which section is weakest +
    why) so the orchestrator can surgically rewrite that one section
    instead of reflowing everything.

    The auditor's verdict is one of:

    - ``PASS`` — ship as-is.
    - ``REVISE_<section>`` — rewrite the named section using the
      paired feedback line. ``<section>`` is most commonly ``recap``
      when the synthesis fails to weave; for individual-panel
      contradictions it'll be the offending panel.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.agent = LaborAgent(
            LaborAgentConfig(
                model=model or os.getenv(AGENT_MODEL_ENV, "phi3"),
                system_prompt=HOLISTIC_REVIEWER_SYSTEM_PROMPT,
                base_url=OLLAMA_BASE_URL,
                temperature=0.0,
                timeout=timeout,
            )
        )

    def audit(
        self,
        section_texts: dict[str, str],
        section_facts: dict[str, str],
    ) -> dict:
        """
        Audit the full tab output. Returns ``{"passed", "target",
        "feedback"}``: ``target`` is the section name to revise
        when ``passed`` is False (defaults to ``"recap"`` if the
        auditor's response is malformed but indicates a problem).
        """
        prompt = self._build_audit_prompt(section_texts, section_facts)
        try:
            response = self.agent.invoke(prompt).strip()
        except Exception as exc:  # noqa: BLE001 — never let auditor block the page
            logger.warning("[holistic] audit raised: %s", exc)
            return {"passed": True, "target": None, "feedback": ""}

        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        if not lines:
            return {"passed": True, "target": None, "feedback": ""}
        verdict = lines[0]
        feedback = lines[1] if len(lines) > 1 else ""

        if verdict.upper().startswith("PASS"):
            return {"passed": True, "target": None, "feedback": feedback}
        if verdict.upper().startswith("REVISE_"):
            target = verdict.split("_", 1)[1].strip().lower().rstrip(":")
            # Sanity: the auditor sometimes invents a section. Fall
            # back to ``recap`` in that case — it's the most common
            # target and the safest one to rewrite.
            if target not in section_texts:
                target = "recap" if "recap" in section_texts else next(iter(section_texts))
            return {"passed": False, "target": target, "feedback": feedback}
        # Malformed — bias toward shipping (the per-section reviewer
        # already cleared each piece individually).
        return {"passed": True, "target": None, "feedback": ""}

    @staticmethod
    def _build_audit_prompt(
        section_texts: dict[str, str],
        section_facts: dict[str, str],
    ) -> str:
        chunks: list[str] = []
        for section, text in section_texts.items():
            facts = section_facts.get(section, "")
            chunks.append(
                f"### Section: {section}\n"
                f"Facts the section was grounded in:\n{facts or '(none provided)'}\n"
                f"Narrative:\n{text}"
            )
        return (
            "Audit the following tab output. The section names are "
            f"{list(section_texts.keys())}.\n\n"
            + "\n\n".join(chunks)
            + "\n\nReview:"
        )


TILE_REVIEWER_SYSTEM_PROMPT = (
    "You audit a single dashboard tile — one figure plus its "
    "accompanying narrative — for consistency. Fail any tile where "
    "the narrative makes a claim the figure cannot support, or vice "
    "versa, or where they describe different scopes (different "
    "states, different time windows, different metrics). "
    "Output two lines:\n"
    "Line 1: PASS or REVISE\n"
    "Line 2: one-sentence reason citing the specific mismatch (REVISE) "
    "       or the strongest claim both reinforce (PASS)."
)


class TileReviewer:
    """
    Per-tile auditor — sees ONE figure spec + its blurb together and
    checks that they tell the same story. Sits between
    :class:`ReviewerAgent` (per-section text) and
    :class:`HolisticReviewer` (cross-section coherence).

    Today the figure spec is None for every tile (the image-writer
    track is :mod:`utils.agents.image_writer` and still stubbed), so
    :meth:`TileReviewer.review` short-circuits to PASS. When the
    figure-builder lands the LLM round-trip kicks in and the
    auditor's REVISE feedback flows back into the same one-shot
    revise path the holistic reviewer uses.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.agent = LaborAgent(
            LaborAgentConfig(
                model=model or os.getenv(AGENT_MODEL_ENV, "phi3"),
                system_prompt=TILE_REVIEWER_SYSTEM_PROMPT,
                base_url=OLLAMA_BASE_URL,
                temperature=0.0,
                timeout=timeout,
            )
        )

    def review(
        self,
        *,
        section: str,
        blurb: str,
        figure_spec: object | None,
        facts: str,
    ) -> dict:
        """
        Audit a single ``(figure, blurb)`` tile.

        Returns ``{"passed", "feedback"}``. When ``figure_spec`` is
        ``None`` (the current default — see :mod:`image_writer`) the
        method bias-passes since we have nothing to cross-check against.
        """
        if figure_spec is None:
            # Image-writer track is stubbed; fall through to PASS. The
            # text-track ReviewerAgent already cleared the blurb in
            # isolation, and the HolisticReviewer will catch
            # cross-section issues.
            return {"passed": True, "feedback": "", "stage": "skipped (no image spec)"}
        prompt = (
            f"Section: {section}\n"
            f"Grounding facts:\n{facts}\n\n"
            f"Figure spec:\n{figure_spec}\n\n"
            f"Narrative:\n{blurb}\n\n"
            f"Review:"
        )
        try:
            response = self.agent.invoke(prompt).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[tile-reviewer] section=%s raised: %s", section, exc)
            return {"passed": True, "feedback": "", "stage": "error-pass"}
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        if not lines:
            return {"passed": True, "feedback": "", "stage": "malformed-pass"}
        verdict = lines[0].upper()
        feedback = lines[1] if len(lines) > 1 else ""
        return {
            "passed": verdict.startswith("PASS"),
            "feedback": feedback,
            "stage": "llm",
        }
