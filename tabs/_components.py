"""
Shared UI components used across tabs.

Kept intentionally small — the two things the tabs legitimately need
to reuse are (a) the page-level error boundary decorator for callbacks
and (b) a small set of status/badge components that read consistently
across the app. Everything here is a pure Dash element factory; no
side effects.
"""
from __future__ import annotations

import functools
import logging
import traceback
from typing import Any, Callable

from dash import dcc, html
from dash.exceptions import PreventUpdate

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Error boundary for Dash callbacks
# --------------------------------------------------------------------------
def error_boundary(
    fallback_id: str | None = None,
    *,
    extra_outputs: int = 0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for Dash callback functions: catches uncaught exceptions
    and returns a friendly alert instead of the raw Dash debug page.

    ``PreventUpdate`` and legitimate ``no_update`` paths are passed
    through unchanged; only unexpected exceptions are wrapped.
    ``fallback_id`` is baked into the rendered markup as a
    ``data-error-of`` attribute so clientside code / tests can locate
    the failed boundary.

    Multi-output callbacks set ``extra_outputs`` to the number of
    *additional* outputs beyond the primary alert. The boundary
    returns ``(alert, None, None, …)`` so Dash's tuple-shape check
    still passes after a failure.
    """

    def _wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def inner(*args: Any, **kwargs: Any):
            try:
                return func(*args, **kwargs)
            except PreventUpdate:
                raise
            except Exception as exc:  # noqa: BLE001 — user-facing boundary
                logger.warning(
                    f"[error-boundary] {func.__name__} raised: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                alert = error_alert(
                    message=f"{type(exc).__name__}: {exc}",
                    fallback_id=fallback_id,
                )
                if extra_outputs <= 0:
                    return alert
                return (alert, *(None for _ in range(extra_outputs)))

        return inner

    return _wrap


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------
def status_pill(label: str, tone: str = "default", *, title: str | None = None) -> html.Span:
    """
    Tiny capsule badge used in the status strip.

    ``tone`` picks the colour — one of ``"default"``, ``"ok"``,
    ``"warn"``, ``"danger"``.
    """
    class_map = {
        "default": "pi-pill",
        "ok":      "pi-pill pi-pill--ok",
        "warn":    "pi-pill pi-pill--warn",
        "danger":  "pi-pill pi-pill--danger",
    }
    cls = class_map.get(tone, "pi-pill")
    extra: dict[str, Any] = {}
    if title:
        extra["title"] = title
    return html.Span(
        [html.Span(className="pi-dot"), label],
        className=cls,
        **extra,
    )


def error_alert(message: str, *, fallback_id: str | None = None) -> html.Div:
    """
    Friendly error banner — replaces the raw Dash/Flask traceback page
    when a callback blows up. The original exception is logged; the
    user sees a short, non-scary message.
    """
    attrs: dict[str, Any] = {
        "role": "alert",
        "aria-live": "assertive",
    }
    if fallback_id:
        attrs["data-error-of"] = fallback_id
    return html.Div(
        [
            html.Strong("Something went wrong. "),
            html.Span(message, className="pi-subtle"),
            html.Div(
                "The error has been logged. Try clicking Run again, "
                "or refresh if it persists.",
                className="pi-muted mt-2 small",
            ),
        ],
        className="alert alert-danger mt-2",
        **attrs,
    )


def loading_skeleton(*, lines: int = 3, width: str = "100%") -> html.Div:
    """
    Shimmer placeholder for not-yet-populated panels. Good for Dash's
    ``dcc.Loading`` ``children=`` arg on first render.

    The width pattern (100% / 90% / 70%) repeats so callers can ask
    for any positive ``lines`` count; previously ``lines > 3``
    silently truncated to three rows because ``(100, 90, 70)[:lines]``
    is bounded by the tuple's length.
    """
    if lines < 1:
        raise ValueError(f"loading_skeleton expects lines >= 1, got {lines}")
    base = (100, 90, 70)
    widths = (base * ((lines // len(base)) + 1))[:lines]
    return html.Div(
        [
            html.Div(
                className="pi-skeleton",
                style={
                    "height": "0.85rem",
                    "marginBottom": "0.5rem",
                    "width": f"{max(40, width_pct)}%",
                },
            )
            for width_pct in widths
        ],
        style={"width": width, "padding": "0.5rem 0"},
    )


# --------------------------------------------------------------------------
# Interleaved figure → caption → AI explanation pattern
# --------------------------------------------------------------------------
#: Pattern-matching ID type used by the deferred AI explanation callback.
#: Tabs use ``{"type": PANEL_BLURB_TYPE, "tab": "lfp", "section": "<id>"}``;
#: the per-tab callback fires once per matched section and renders the
#: AI prose into the placeholder.
PANEL_BLURB_TYPE = "pi-panel-blurb"


def figure_panel(
    *,
    title: str,
    figure: Any,
    caption: str | None,
    blurb_id: dict,
    placeholder: str = "Generating narrative analysis…",
) -> html.Div:
    """
    Interleaved tile: figure / table → caption → AI explanation.

    The ``figure`` argument can be any Dash element — a ``dcc.Graph``,
    a Bootstrap table built elsewhere, or an ``html.Img`` for a
    seaborn-rendered PNG. ``blurb_id`` is the pattern-matching dict
    (e.g. ``{"type": PANEL_BLURB_TYPE, "tab": "lfp", "section":
    "forecast"}``) the deferred AI callback writes to. A ``caption``
    of ``None`` skips the caption row entirely.

    Accessibility: the AI placeholder carries ``role="status"`` and
    ``aria-live="polite"`` so screen readers announce when the
    asynchronous narrative arrives without stealing focus.
    """
    children: list[Any] = [
        html.H6(title, className="pi-panel-title"),
        html.Div(figure, className="pi-panel-figure"),
    ]
    if caption:
        children.append(
            html.Div(caption, className="pi-panel-caption pi-muted small")
        )
    children.append(
        html.Div(
            html.Em(placeholder, className="pi-muted small"),
            id=blurb_id,
            className="pi-panel-blurb",
            role="status",
            **{"aria-live": "polite", "aria-busy": "true"},
        )
    )
    return html.Div(children, className="pi-panel-tile")


def tab_recap(
    *,
    title: str = "Recap & deeper detail",
    blurb_id: dict,
    placeholder: str = "Synthesising the recap…",
) -> html.Div:
    """
    End-of-tab synthesis section. Same deferred-fill pattern as
    :func:`figure_panel`, but visually distinct (heavier title, no
    figure) so the user reads it as a wrap-up rather than another
    chart.
    """
    return html.Div(
        [
            html.H5(title, className="pi-recap-title"),
            html.Div(
                html.Em(placeholder, className="pi-muted small"),
                id=blurb_id,
                className="pi-recap-blurb",
                role="status",
                **{"aria-live": "polite", "aria-busy": "true"},
            ),
        ],
        className="pi-recap-section",
    )


def progress_strip(progress_id: str) -> html.Div:
    """
    Top-of-tab progress region. Driven by the polling callback that
    streams panel completions in. Carries ``role="status"`` +
    ``aria-live="polite"`` so screen readers announce each transition
    ("generating panel 1 of 4" → "generated panel 1 of 4: forecast" →
    …) without interrupting the user's reading focus.
    """
    return html.Div(
        html.Span("", id=progress_id, className="pi-progress-text"),
        className="pi-progress-strip",
        role="status",
        **{"aria-live": "polite", "aria-atomic": "true"},
    )


def render_blurb(text: str, *, disclaimer: bool = True) -> html.Div:
    """
    Standard wrapper for AI-generated prose: Markdown body + optional
    italic disclaimer. Centralised so every callback that fills a
    panel-blurb placeholder uses the same shape.
    """
    children: list[Any] = [dcc.Markdown(text, className="pi-blurb-body")]
    if disclaimer:
        children.append(
            dcc.Markdown(
                "_AI-generated narrative. May contain inaccuracies — "
                "ground decisions in the cited methodology._",
                className="pi-blurb-disclaimer pi-muted small",
            )
        )
    return html.Div(children, className="pi-blurb")
