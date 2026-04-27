"""
Floating right-side chat drawer shared across every tab.

Replaces the four per-tab chat strips that used to live at the
bottom of each tab. Those were redundant (same LLM, same
generate_insight call) and crowded the data view below the fold.

Mounted once in ``app.py`` outside the tab content container, so
it's always available. A single callback wires the submit button
to ``generate_insight`` with the currently-active tab as context,
so the RAG retrieval stays tab-aware.

Components exported
-------------------
- :func:`render_drawer` — the markup block. Drop into the top-level
  layout once.
- :func:`register_callbacks(app)` — wires toggle + submit.
"""
from __future__ import annotations

import logging

from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from utils.llm_utils import AI_FAILURE_MESSAGE, generate_insight
from utils.agents.sentence_rag import default_rag_builder

logger = logging.getLogger(__name__)

#: Friendly labels for the context pill inside the drawer header.
TAB_LABELS: dict[str, str] = {
    "eda":   "EDA / Overview",
    "lfp":   "LFP Forecast",
    "super": "Supersector Forecast",
    "about": "About",
}

#: Hard cap on a single chat message — both client-side (textarea
#: ``maxLength``) and server-side (callback validation). 5 000 chars
#: comfortably exceeds any legitimate question and stops a CSV paste
#: from queueing a multi-minute Ollama call (OWASP API A04:2023).
MAX_CHAT_MESSAGE_CHARS = 5_000

#: How much of an over-length message to echo back so the user can
#: see what got rejected without filling the chat history.
_TRUNCATE_PREVIEW_CHARS = 200


def render_drawer() -> html.Div:
    """Return the drawer + FAB button; drop once into app.layout."""
    return html.Div(
        [
            html.Button(
                [
                    html.Span("💬", className="pi-chat-fab-icon", **{"aria-hidden": "true"}),
                    html.Span("Chat"),
                ],
                id="chat-drawer-toggle",
                className="pi-chat-fab",
                **{"aria-label": "Open Prairie Assistant chat"},
                n_clicks=0,
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5("Prairie Assistant"),
                                    html.Small(
                                        "Context: EDA / Overview",
                                        id="chat-drawer-context",
                                        className="pi-chat-context",
                                    ),
                                ]
                            ),
                            html.Button(
                                "×",
                                id="chat-drawer-close",
                                className="pi-chat-drawer-close",
                                **{"aria-label": "Close chat"},
                                n_clicks=0,
                            ),
                        ],
                        className="pi-chat-drawer-header",
                    ),
                    html.Div(
                        html.Div(
                            "Ask about the data, forecasts, or methodology. "
                            "The assistant answers from the current tab's "
                            "context — switch tabs to refocus.",
                            className="pi-chat-empty",
                        ),
                        id="chat-drawer-body",
                        className="pi-chat-drawer-body",
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Chat message",
                                htmlFor="chat-drawer-input",
                                className="visually-hidden",
                            ),
                            dcc.Textarea(
                                id="chat-drawer-input",
                                placeholder="What would you like to know?",
                                maxLength=MAX_CHAT_MESSAGE_CHARS,
                            ),
                            html.Button(
                                "Send",
                                id="chat-drawer-submit",
                                className="btn btn-primary",
                                n_clicks=0,
                            ),
                        ],
                        className="pi-chat-drawer-footer",
                    ),
                ],
                id="chat-drawer",
                className="pi-chat-drawer pi-chat-drawer-closed",
                **{"aria-hidden": "true"},
            ),
            # Client-visible state of the drawer.
            dcc.Store(id="chat-drawer-open", data=False),
            # Conversation log: list of (role, text).
            dcc.Store(id="chat-drawer-history", data=[]),
        ]
    )


def _render_history(history: list[dict]) -> list:
    """Turn the running message log into bubble divs."""
    if not history:
        return [
            html.Div(
                "Ask about the data, forecasts, or methodology. "
                "The assistant answers from the current tab's context — "
                "switch tabs to refocus.",
                className="pi-chat-empty",
            )
        ]
    bubbles = []
    for msg in history:
        role = msg.get("role", "assistant")
        text = msg.get("text", "") or ""
        klass = (
            "pi-chat-bubble pi-chat-bubble-user"
            if role == "user"
            else "pi-chat-bubble pi-chat-bubble-assistant"
        )
        bubbles.append(dcc.Markdown(text, className=klass))
    return bubbles


def register_callbacks(app) -> None:
    @app.callback(
        Output("chat-drawer", "className"),
        Output("chat-drawer", "aria-hidden"),
        Output("chat-drawer-open", "data"),
        Input("chat-drawer-toggle", "n_clicks"),
        Input("chat-drawer-close", "n_clicks"),
        State("chat-drawer-open", "data"),
    )
    def toggle_drawer(toggle_clicks, close_clicks, is_open):
        # Close button always closes; toggle inverts.
        from dash import ctx  # noqa: PLC0415

        trigger = ctx.triggered_id
        if trigger == "chat-drawer-close":
            new_state = False
        elif trigger == "chat-drawer-toggle":
            new_state = not bool(is_open)
        else:
            raise PreventUpdate
        class_name = "pi-chat-drawer" if new_state else "pi-chat-drawer pi-chat-drawer-closed"
        return class_name, "false" if new_state else "true", new_state

    @app.callback(
        Output("chat-drawer-context", "children"),
        Input("tabs", "active_tab"),
    )
    def update_context_label(active_tab):
        label = TAB_LABELS.get(active_tab, "Dashboard")
        return f"Context: {label}"

    @app.callback(
        Output("chat-drawer-history", "data"),
        Output("chat-drawer-body", "children"),
        Output("chat-drawer-input", "value"),
        Input("chat-drawer-submit", "n_clicks"),
        State("chat-drawer-input", "value"),
        State("tabs", "active_tab"),
        State("chat-drawer-history", "data"),
        State("pi-active-view", "data"),
    )
    def submit_chat(n_clicks, query, active_tab, history, active_view):
        if not n_clicks or not query or not query.strip():
            raise PreventUpdate
        cleaned = query.strip()
        history = list(history or [])
        # Refuse before reaching the LLM rather than truncating silently
        # — the client-side ``maxLength`` is advisory; this is the gate.
        if len(cleaned) > MAX_CHAT_MESSAGE_CHARS:
            history.append(
                {"role": "user", "text": cleaned[:_TRUNCATE_PREVIEW_CHARS] + "…"}
            )
            history.append({
                "role": "assistant",
                "text": (
                    f"Your message is {len(cleaned):,} characters long; the "
                    f"limit is {MAX_CHAT_MESSAGE_CHARS:,}. Please shorten it."
                ),
            })
            return history, _render_history(history), ""
        history.append({"role": "user", "text": cleaned})
        # Glue the user's prompt to whatever the active tab last
        # rendered — the chat answer is then grounded in the figures
        # the user is literally looking at, not just RAG keywords.
        view_prefix = _render_active_view_context(active_view)
        prompt = (
            f"{view_prefix}\n\nUser question: {cleaned}"
            if view_prefix
            else cleaned
        )
        try:
            answer = generate_insight(prompt, active_tab=active_tab)
        except Exception as exc:  # noqa: BLE001 — full trace stays server-side
            logger.warning(
                "[chat-drawer] generate_insight failed: %s: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            # Wrap the shared failure message in Markdown italics so the
            # bubble visually distinguishes a fallback from a real reply.
            answer = f"_{AI_FAILURE_MESSAGE}_"
        history.append({"role": "assistant", "text": answer})
        return history, _render_history(history), ""


def _render_active_view_context(active_view: dict | None) -> str:
    """
    Turn the global ``pi-active-view`` payload into ontology-aware
    sentences the chat drawer can prepend to the user's prompt.

    Every tab that wants its rendered output to ground chat answers
    writes ``{"tab": "<id>", "panels": [view_state, ...], "recap":
    view_state}`` into the Store; we run those view_states through
    the same :class:`SentenceRAGBuilder` the per-panel blurbs use, so
    the LLM sees plain English ("Iowa labor force participation rate
    is currently 64.5%, down 1.2 pp over five years") rather than
    raw key/value dumps.
    """
    if not isinstance(active_view, dict):
        return ""
    panels = active_view.get("panels") or []
    recap = active_view.get("recap")
    builder = default_rag_builder()
    sentences: list[str] = []
    for panel in panels:
        sentences.extend(builder.render_view_sentences(panel))
    if recap:
        sentences.extend(builder.render_view_sentences(recap))
    if not sentences:
        return ""
    body = "\n".join(f"- {s}" for s in sentences)
    return (
        "Active dashboard view (use these facts to ground your answer; "
        "do not invent values):\n"
        f"{body}"
    )
