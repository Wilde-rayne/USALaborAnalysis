"""
Image-writer agent — stubbed skeleton for the parallel
plot-generation track in the nested harness.

The user's architecture sketch is::

    base → total context → page context
                              ├─→ review → writer → review   (text track — implemented)
                              └─→ image writer / creator     (this module)

The image writer is the parallel branch that produces the Plotly /
seaborn / matplotlib code (or a structured spec) for the panel
figures. Today the chart-construction code lives inline in each tab
(``tabs/lfp_tab.py``, etc.) so this stub is a placeholder for the
agent that will eventually own that responsibility.

Two implementation paths are open:

1. **Code-generating writer** — the agent emits Python (Plotly /
   matplotlib) that a sandboxed executor turns into a figure. Most
   flexible; security-sensitive (need to constrain imports + exec
   environment).
2. **Spec-generating writer** — the agent emits a JSON spec keyed
   to a known set of plot templates (line+forecast, bar+CI, etc.).
   Safer; less flexible. Almost certainly the right answer here
   given the dashboard's bounded chart vocabulary.

This module reserves the namespace + the interface so the next slice
has a clean hook. ``ImageWriterAgent.write_spec`` is the entry point
the orchestrator will call alongside the text-track specialists in a
future revision.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from utils.agents.base import LaborAgent, LaborAgentConfig
from utils.agents.ollama import AGENT_MODEL_ENV, OLLAMA_BASE_URL
from utils.constants import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FigureSpec:
    """
    Structured plot description the image-writer emits. A small
    builder in :mod:`tabs._charts` (next slice) turns this into a
    real ``plotly.graph_objs.Figure``.

    Fields are intentionally narrow — the bounded vocabulary keeps
    the LLM's output tractable to validate.
    """
    kind: str                      # "line_with_forecast" / "bar_with_ci" / "kde" / ...
    title: str
    x_field: str
    y_field: str
    series: list[dict] = field(default_factory=list)   # per-series cosmetics
    annotations: list[dict] = field(default_factory=list)
    color_scheme: str = "brand"


IMAGE_WRITER_SYSTEM_PROMPT = (
    "You design a labor-market figure for a state workforce planner. "
    "Given a panel view_state and a fixed set of chart kinds, output "
    "a STRICT JSON object matching the FigureSpec schema:\n"
    "  {kind, title, x_field, y_field, series:[...], annotations:[...], "
    "color_scheme}\n"
    "Use ONLY these kinds: line_with_forecast, bar_with_ci, kde, "
    "stacked_bar, ridge.\n"
    "Output JSON only — no preamble, no markdown fences."
)


class ImageWriterAgent:
    """
    Skeleton for the spec-generating image writer.

    Not wired into the orchestrator yet — the existing tabs render
    their own figures inline. Once the figure-builder utility lands,
    swap the inline ``go.Figure(...)`` blocks for
    :meth:`ImageWriterAgent.write_spec` calls and a small
    ``build_figure(spec)`` builder.
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
                system_prompt=IMAGE_WRITER_SYSTEM_PROMPT,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
                timeout=timeout,
            )
        )

    def write_spec(self, view_state: dict) -> FigureSpec | None:
        """
        Emit a :class:`FigureSpec` for the given panel view_state.

        Currently returns ``None`` — the inline chart code in each
        tab still drives rendering. The caller is expected to fall
        back to the inline path on ``None`` until this method is
        fully implemented.
        """
        # NOTE: stubbed. Wiring happens in the figure-builder slice;
        # the LLM round-trip + JSON parsing + validation lands there.
        logger.debug(
            "[image-writer] write_spec called for kind=%s (stubbed)",
            view_state.get("_kind"),
        )
        return None
