"""
Smoke tests for the Dash tab modules.

Verifies each tab: (a) imports cleanly, (b) render_layout returns a
valid Dash component, (c) register_callbacks wires the expected number
of callbacks against a mock Dash app. Skip the file if Dash isn't
installed in the local env — the lightweight CI path doesn't need it.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("dash")

from dash import html  # noqa: E402

from tabs import about_tab, eda_tab, lfp_tab, super_tab  # noqa: E402
from tabs._components import (  # noqa: E402
    error_alert,
    error_boundary,
    loading_skeleton,
    status_pill,
)


class FakeApp:
    """Minimal stand-in for the Dash app — captures registered callbacks."""

    def __init__(self) -> None:
        self.registered: list[tuple[Any, ...]] = []

    def callback(self, *args: Any, **kwargs: Any):
        def _decorator(func):
            self.registered.append((func.__name__, args, kwargs))
            return func

        return _decorator


# --------------------------------------------------------------------------
# Each tab renders and registers callbacks
# --------------------------------------------------------------------------
class TestRenderLayout:
    @pytest.mark.parametrize(
        "tab_mod",
        [eda_tab, lfp_tab, super_tab, about_tab],
        ids=["eda", "lfp", "super", "about"],
    )
    def test_render_returns_component(self, tab_mod) -> None:
        layout = tab_mod.render_layout()
        # Every tab returns a Dash component tree rooted at html.Div.
        assert layout is not None
        assert hasattr(layout, "children")
        # Don't over-assert on shape; just confirm there IS content.
        assert layout.children


class TestRegisterCallbacks:
    """
    Chat was centralised into ``tabs/_chat_drawer.py`` (see I3). Each
    forecast tab also registers a ``poll_*_blurbs`` polling callback
    that streams the LLM panel narratives in as they complete — these
    tests assert the expected callback names are present without
    over-constraining the exact set so future polling helpers don't
    immediately break the smoke suite.
    """

    def test_eda_registers_expected_callbacks(self) -> None:
        app = FakeApp()
        eda_tab.register_callbacks(app)
        names = {r[0] for r in app.registered}
        # update_metadata + update_eda + poll_eda_blurbs. Chat lives in
        # the global drawer.
        assert {"update_metadata", "update_eda"}.issubset(names)

    def test_lfp_registers_forecast_callback(self) -> None:
        app = FakeApp()
        lfp_tab.register_callbacks(app)
        names = {r[0] for r in app.registered}
        assert "update_lfp" in names

    def test_super_registers_forecast_callback(self) -> None:
        app = FakeApp()
        super_tab.register_callbacks(app)
        names = {r[0] for r in app.registered}
        assert "update_super" in names

    def test_about_registers_no_callbacks(self) -> None:
        app = FakeApp()
        about_tab.register_callbacks(app)
        # About is pure content now; no callbacks.
        assert app.registered == []


# --------------------------------------------------------------------------
# tabs/_components public surface
# --------------------------------------------------------------------------
class TestStatusPill:
    def test_default_tone(self) -> None:
        p = status_pill("hello")
        assert "pi-pill" in p.className
        assert p.children[1] == "hello"

    def test_ok_tone_adds_modifier(self) -> None:
        p = status_pill("ready", tone="ok")
        assert "pi-pill--ok" in p.className

    def test_unknown_tone_falls_back_to_default(self) -> None:
        p = status_pill("x", tone="mystery")
        assert p.className == "pi-pill"

    def test_title_passes_through(self) -> None:
        p = status_pill("x", title="hover tip")
        assert p.title == "hover tip"


class TestErrorAlert:
    def test_has_alert_role(self) -> None:
        alert = error_alert("boom")
        assert alert.role == "alert"

    def test_fallback_id_recorded_in_dom_attrs(self) -> None:
        alert = error_alert("boom", fallback_id="xyz")
        # The underlying Dash html.Div stores raw HTML attrs; assert by
        # reading the constructor kwargs off the instance.
        # (Dash html.Div accepts arbitrary kwargs that land on _other_props.)
        # Non-standard attrs ride on the Div's raw dict.
        dom_attrs = getattr(alert, "_children", None)
        assert alert is not None  # basic smoke — attribute actually landed elsewhere
        # The data-error-of attr round-trips via html.Div's kwargs.
        assert getattr(alert, "data-error-of", None) == "xyz"


class TestErrorBoundary:
    def test_passes_preventupdate_through(self) -> None:
        from dash.exceptions import PreventUpdate

        @error_boundary()
        def bad(_n):
            raise PreventUpdate

        with pytest.raises(PreventUpdate):
            bad(1)

    def test_catches_other_exceptions(self) -> None:
        @error_boundary(fallback_id="id-x")
        def boom(_n):
            raise ValueError("nope")

        result = boom(1)
        # Returns an error_alert Div — not a raised exception.
        assert result is not None
        assert "alert" in result.className

    def test_returns_value_when_no_exception(self) -> None:
        @error_boundary()
        def ok(_n):
            return html.Div("hi")

        assert ok(1).children == "hi"


class TestSkeleton:
    def test_skeleton_renders_n_lines(self) -> None:
        sk = loading_skeleton(lines=2)
        # Outer wrapper Div has list of skeleton Divs.
        assert len(sk.children) == 2

    def test_skeleton_supports_more_than_three_lines(self) -> None:
        # Reviews item #8: the legacy implementation silently truncated
        # past three lines because it sliced a fixed 3-tuple.
        sk = loading_skeleton(lines=5)
        assert len(sk.children) == 5
