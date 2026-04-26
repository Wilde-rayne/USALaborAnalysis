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


def empty_state(
    title: str,
    body: str,
    *,
    action: html.Button | None = None,
    icon: str = "📊",
) -> html.Div:
    """
    Consistent "nothing to show yet" block — used as a fallback when a
    tab's data or cache isn't ready.
    """
    children: list[Any] = [
        html.Div(icon, style={"fontSize": "2.5rem", "marginBottom": "0.5rem"}),
        html.H5(title, className="mb-2"),
        html.P(body, className="pi-muted mb-3"),
    ]
    if action is not None:
        children.append(action)
    return html.Div(
        children,
        className="pi-section text-center",
        role="status",
        **{"aria-live": "polite"},
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


def section(title: str, *children: Any, id: str | None = None) -> html.Div:
    """Consistent ``pi-section`` wrapper with an optional H5 heading."""
    head = [html.H5(title)] if title else []
    props: dict[str, Any] = {"className": "pi-section"}
    if id:
        props["id"] = id
    return html.Div([*head, *children], **props)


def loading_skeleton(*, lines: int = 3, width: str = "100%") -> html.Div:
    """
    Shimmer placeholder for not-yet-populated panels. Good for Dash's
    ``dcc.Loading`` ``children=`` arg on first render.
    """
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
            for width_pct in (100, 90, 70)[:lines]
        ],
        style={"width": width, "padding": "0.5rem 0"},
    )
