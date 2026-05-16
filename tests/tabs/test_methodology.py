"""Tests for the methodology + references panel."""
from __future__ import annotations

import pytest

pytest.importorskip("dash")
from dash import html  # noqa: E402

from tabs._methodology import (
    DATA_SOURCE_KEYS,
    DIAGNOSTIC_NOTES,
    FORECAST_NOTES,
    SOFTWARE_ATTRIBUTION_KEYS,
    MethodologyNote,
    _inline_cites_suffix,
    _render_notes,
    _render_references,
    chart_source_annotation,
    methodology_panel,
)


class TestNotes:
    def test_every_note_cites_a_known_key(self) -> None:
        from utils.citations import CITATIONS

        for note in FORECAST_NOTES + DIAGNOSTIC_NOTES:
            for key in note.cites:
                assert key in CITATIONS, (note.text[:40], key)

    def test_every_data_source_key_is_known(self) -> None:
        from utils.citations import CITATIONS

        for key in DATA_SOURCE_KEYS:
            assert key in CITATIONS

    def test_every_software_attribution_key_is_known(self) -> None:
        """Software / model attribution keys (Track C item 9) must all
        resolve against the citation registry."""
        from utils.citations import CITATIONS

        for key in SOFTWARE_ATTRIBUTION_KEYS:
            assert key in CITATIONS, key

    def test_software_attribution_includes_built_with_llama_license(self) -> None:
        """The Llama 3.2 Community License key is required by Meta's
        license for any service that surfaces Llama-family output."""
        assert "llama3_license_2024" in SOFTWARE_ATTRIBUTION_KEYS


class TestInlineCitesSuffix:
    def test_no_cites_returns_empty(self) -> None:
        assert _inline_cites_suffix([]) == ""

    def test_single_cite_gives_one_suffix(self) -> None:
        out = _inline_cites_suffix(["holt_1957"])
        assert out.startswith(" ")
        assert "Holt" in out and "1957" in out

    def test_multiple_cites_separated_by_semicolon(self) -> None:
        out = _inline_cites_suffix(["holt_1957", "winters_1960"])
        assert "Holt" in out and "Winters" in out
        assert ";" in out


class TestRenderNotes:
    def test_produces_html_ul_with_one_li_per_note(self) -> None:
        notes = [
            MethodologyNote("first fact.", cites=("holt_1957",)),
            MethodologyNote("second fact.", cites=("winters_1960",)),
        ]
        ul = _render_notes(notes)
        assert isinstance(ul, html.Ul)
        assert len(ul.children) == 2


class TestRenderReferences:
    def test_sorts_by_year_then_authors(self) -> None:
        """References should render in year-ascending order."""
        ol = _render_references(
            ["box_jenkins_1970", "holt_1957", "winters_1960"]
        )
        # Render children; first must be 1957, then 1960, then 1970.
        line_texts: list[str] = []
        for li in ol.children:
            # li.children is a mixed list — stitch strings together.
            txt = "".join(c for c in li.children if isinstance(c, str))
            line_texts.append(txt)
        assert "1957" in line_texts[0]
        assert "1960" in line_texts[1]
        assert "1970" in line_texts[2]

    def test_deduplicates_repeated_keys(self) -> None:
        ol = _render_references(["holt_1957", "holt_1957"])
        assert len(ol.children) == 1


class TestMethodologyPanel:
    def test_default_panel_renders_all_sections(self) -> None:
        panel = methodology_panel()
        # Top-level is Details with Summary + body Div.
        assert isinstance(panel, html.Details)
        # Children: [Summary, Div(body)].
        assert len(panel.children) == 2
        body = panel.children[1]
        assert isinstance(body, html.Div)
        # Section headers: Forecasting + Residual + Data sources +
        # Software & model attribution + Refs.
        header_texts = [
            c.children
            for c in body.children
            if isinstance(c, html.H6) or (hasattr(c, "children") and not isinstance(c, html.Ul))
        ]
        # Simpler check — count H6s.
        h6_count = sum(1 for c in body.children if isinstance(c, html.H6))
        assert h6_count == 5

    def test_custom_summary_text(self) -> None:
        panel = methodology_panel(summary_text="Why you should trust this")
        summary = panel.children[0]
        assert summary.children == "Why you should trust this"

    def test_open_by_default_flag(self) -> None:
        closed = methodology_panel()
        opened = methodology_panel(open_by_default=True)
        assert closed.open is False
        assert opened.open is True

    def test_can_override_notes(self) -> None:
        custom = [MethodologyNote("just one thing.", cites=("holt_1957",))]
        panel = methodology_panel(forecast_notes=custom)
        body = panel.children[1]
        # First H6 is "Forecasting & scoring", immediately followed by a Ul
        # with one li.
        ul = next(c for c in body.children if isinstance(c, html.Ul))
        assert len(ul.children) == 1


class TestChartSourceAnnotation:
    """Track C item 11 — every BLS/Census/BEA/FRED/FHFA chart gets a
    consolidated 'Source:' caption per each agency's citation policy."""

    def test_returns_plotly_annotation_dict(self) -> None:
        ann = chart_source_annotation()
        assert isinstance(ann, dict)
        # Required plotly fields for a paper-anchored caption.
        for required in ("text", "xref", "yref", "showarrow"):
            assert required in ann, required
        assert ann["xref"] == "paper"
        assert ann["yref"] == "paper"
        assert ann["showarrow"] is False

    def test_default_text_names_every_agency(self) -> None:
        text = chart_source_annotation()["text"]
        for agency in ("Bureau of Labor Statistics", "Census Bureau",
                       "Bureau of Economic Analysis", "FRED",
                       "Federal Housing Finance Agency"):
            assert agency in text, agency

    def test_text_override(self) -> None:
        ann = chart_source_annotation(text="Source: BLS only")
        assert ann["text"] == "Source: BLS only"
