"""Tests for the citation registry and its rendering helpers."""
from __future__ import annotations

import pytest

from utils.citations import (
    CITATIONS,
    Citation,
    _shorten_authors,
    citation,
    render_full,
    render_inline,
)


class TestRegistry:
    def test_has_core_forecasting_citations(self) -> None:
        """Model and stats-test seminal papers must all be present."""
        required = {
            "holt_1957",
            "winters_1960",
            "box_jenkins_1970",
            "hochreiter_schmidhuber_1997",
            "dickey_fuller_1979",
            "kpss_1992",
            "ljung_box_1978",
            "jarque_bera_1980",
            "diebold_mariano_1995",
            "harvey_leybourne_newbold_1997",
            "hyndman_athanasopoulos_2018",
            "tashman_2000",
        }
        assert required.issubset(CITATIONS.keys())

    def test_has_every_data_source_handbook(self) -> None:
        required = {
            "bls_ces_handbook",
            "bls_laus_handbook",
            "bls_jolts_handbook",
            "bls_qcew_handbook",
            "bls_cpi_handbook",
            "census_acs_handbook",
            "bea_regional_handbook",
            "fred_api",
            "fhfa_hpi_handbook",
        }
        assert required.issubset(CITATIONS.keys())

    def test_every_entry_has_non_empty_fields(self) -> None:
        for key, c in CITATIONS.items():
            assert c.key == key
            assert c.authors.strip(), key
            assert c.title.strip(), key
            assert c.venue.strip(), key
            assert 1950 <= c.year <= 2030, key
            if c.url is not None:
                assert c.url.startswith("http"), key


class TestCitationLookup:
    def test_citation_raises_on_miss(self) -> None:
        with pytest.raises(KeyError, match="unknown citation"):
            citation("not_a_real_key")

    def test_citation_returns_dataclass(self) -> None:
        c = citation("box_jenkins_1970")
        assert isinstance(c, Citation)
        assert c.year == 1970


class TestInlineRendering:
    def test_two_authors_joined_with_ampersand(self) -> None:
        assert render_inline("box_jenkins_1970") == "(Box & Jenkins, 1970)"
        assert render_inline("dickey_fuller_1979") == "(Dickey & Fuller, 1979)"

    def test_three_or_more_authors_becomes_et_al(self) -> None:
        # Harvey, Leybourne, Newbold: 3 authors joined with " and " only once.
        # "Harvey, D., Leybourne, S., and Newbold, P." → "Harvey et al."
        assert render_inline("harvey_leybourne_newbold_1997") == "(Harvey et al., 1997)"
        # KPSS: four authors, no "and" in the stored string (comma-separated)
        # → falls to the comma-path helper.
        result = render_inline("kpss_1992")
        assert result.startswith("(Kwiatkowski")

    def test_single_author_uses_surname(self) -> None:
        assert render_inline("holt_1957") == "(Holt, 1957)"


class TestFullRendering:
    def test_full_bibliography_shape(self) -> None:
        line = render_full("box_jenkins_1970")
        assert line.startswith("Box")
        assert "(1970)" in line
        assert "Time Series Analysis" in line
        assert "Holden-Day" in line

    def test_full_includes_url_when_available(self) -> None:
        line = render_full("diebold_mariano_1995")
        assert "doi.org" in line

    def test_full_without_url_still_renders(self) -> None:
        line = render_full("holt_1957")
        assert "Holt" in line
        assert "ONR" in line


class TestShortenAuthors:
    def test_single_surname(self) -> None:
        assert _shorten_authors("Holt, C. C.") == "Holt"

    def test_two_authors(self) -> None:
        assert _shorten_authors("Box, G. E. P., and Jenkins, G. M.") == "Box & Jenkins"

    def test_three_authors_collapse_to_et_al(self) -> None:
        assert (
            _shorten_authors("Harvey, D., Leybourne, S., and Newbold, P.")
            == "Harvey et al."
        )

    def test_comma_only_list_collapse_to_et_al(self) -> None:
        # Author list with NO "and" keyword, 3+ entries.
        assert (
            _shorten_authors("Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., Shin, Y.")
            == "Kwiatkowski et al."
        )
