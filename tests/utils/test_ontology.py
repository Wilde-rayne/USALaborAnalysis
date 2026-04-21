"""
Unit tests for utils.ontology.

Pure-Python, no runtime dependencies beyond the stdlib — these tests
run in every CI path.
"""
from __future__ import annotations

import pytest

from utils.ontology import (
    MEASURES,
    ONTOLOGY,
    SOURCES,
    STATES,
    SUPERSECTORS,
    Ontology,
    SeriesSpec,
)


class TestReferenceData:
    def test_state_count(self) -> None:
        # 50 states + DC + 5 territories = 56.
        assert len(STATES) == 56
        assert sum(1 for s in STATES if s.kind == "state") == 50
        assert sum(1 for s in STATES if s.kind == "district") == 1
        assert sum(1 for s in STATES if s.kind == "territory") == 5

    def test_no_duplicate_state_codes(self) -> None:
        codes = [s.code for s in STATES]
        assert len(codes) == len(set(codes))

    def test_no_duplicate_fips(self) -> None:
        fips = [s.fips for s in STATES]
        assert len(fips) == len(set(fips))

    def test_every_state_has_all_fields(self) -> None:
        for s in STATES:
            assert len(s.code) == 2 and s.code.isupper()
            assert len(s.fips) == 2 and s.fips.isdigit()
            assert s.name
            assert s.kind in {"state", "district", "territory"}
            assert s.region

    def test_every_supersector_key_is_unique(self) -> None:
        keys = [s.key for s in SUPERSECTORS]
        assert len(keys) == len(set(keys))

    def test_supersector_codes_are_two_digits(self) -> None:
        for s in SUPERSECTORS:
            assert len(s.code) == 2 and s.code.isdigit()

    def test_measures_cover_all_laus_suffixes_seen_in_data(self) -> None:
        """Every LAUS measure known to data_pipeline should exist in ontology."""
        seen_suffixes = {"003", "004", "005", "006"}
        ontology_suffixes = {m.laus_suffix for m in MEASURES if m.laus_suffix}
        assert seen_suffixes <= ontology_suffixes

    def test_sources_have_required_fields(self) -> None:
        for src in SOURCES:
            assert src.id and src.name and src.url
            assert isinstance(src.api_available, bool)


class TestLookup:
    def test_state_code_lookup(self) -> None:
        ia = ONTOLOGY.state("IA")
        assert ia.name == "Iowa"
        assert ia.fips == "19"
        assert ia.region == "Midwest"

    def test_state_lookup_is_case_insensitive(self) -> None:
        assert ONTOLOGY.state("ia").name == "Iowa"

    def test_state_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            ONTOLOGY.state("ZZ")

    def test_fips_lookup(self) -> None:
        s = ONTOLOGY.states_by_fips["06"]
        assert s.code == "CA"

    def test_states_in_region(self) -> None:
        midwest = ONTOLOGY.states_in("Midwest")
        codes = {s.code for s in midwest}
        assert {"IA", "IL", "IN", "OH", "WI"}.issubset(codes)
        assert "CA" not in codes

    def test_iter_territories(self) -> None:
        t = {s.code for s in ONTOLOGY.iter_states(kinds=("territory",))}
        assert t == {"PR", "VI", "GU", "AS", "MP"}

    def test_supersector_by_key(self) -> None:
        mfg = ONTOLOGY.supersector("Manufacturing")
        assert mfg.code == "30"
        assert "NAICS" in mfg.naics

    def test_measure_lookup(self) -> None:
        lf = ONTOLOGY.measure("Labor_Force")
        assert lf.laus_suffix == "006"
        assert lf.source == "LAUS"


class TestSeriesIdParsing:
    def test_parses_laus_labor_force(self) -> None:
        spec = ONTOLOGY.parse_series_id("LASST190000000000006")
        assert isinstance(spec, SeriesSpec)
        assert spec.state.code == "IA"
        assert spec.measure.key == "Labor_Force"
        assert spec.source == "bls_laus"
        assert spec.supersector is None

    def test_parses_laus_unemployment(self) -> None:
        spec = ONTOLOGY.parse_series_id("LASST390000000000004")
        assert spec.state.code == "OH"
        assert spec.measure.key == "Unemployment"

    def test_parses_ces_manufacturing(self) -> None:
        spec = ONTOLOGY.parse_series_id("SMS19000003000000001")
        assert spec.state.code == "IA"
        assert spec.source == "bls_ces"
        assert spec.supersector is not None
        assert spec.supersector.key == "Manufacturing"
        assert spec.measure.key == "Sector_Employment"

    def test_invalid_prefix_raises(self) -> None:
        with pytest.raises(ValueError):
            ONTOLOGY.parse_series_id("BOGUS12345")

    def test_unknown_fips_raises(self) -> None:
        with pytest.raises(ValueError):
            ONTOLOGY.parse_series_id("LASST990000000000006")


class TestDescribe:
    def test_laus_description_mentions_state_and_measure(self) -> None:
        text = ONTOLOGY.describe("LASST190000000000006")
        assert "Iowa" in text
        assert "labor force" in text.lower()

    def test_ces_description_mentions_supersector(self) -> None:
        text = ONTOLOGY.describe("SMS19000003000000001")
        assert "Iowa" in text
        assert "manufacturing" in text.lower()


class TestOntologyConstruction:
    def test_fresh_instance_matches_singleton(self) -> None:
        # If anything mutates the module-level singleton a test that
        # constructs a fresh Ontology should catch the drift.
        fresh = Ontology()
        assert set(fresh.states.keys()) == set(ONTOLOGY.states.keys())
        assert set(fresh.supersectors.keys()) == set(ONTOLOGY.supersectors.keys())


class TestStateCodes:
    def test_default_returns_all_56(self) -> None:
        codes = ONTOLOGY.state_codes()
        assert len(codes) == 56
        assert codes == sorted(codes)

    def test_states_only_kinds(self) -> None:
        codes = ONTOLOGY.state_codes(kinds=("state",))
        assert len(codes) == 50
        assert "DC" not in codes
        assert "PR" not in codes

    def test_states_plus_district(self) -> None:
        codes = ONTOLOGY.state_codes(kinds=("state", "district"))
        assert len(codes) == 51
        assert "DC" in codes

    def test_region_filter(self) -> None:
        midwest = ONTOLOGY.state_codes(regions=("Midwest",))
        assert len(midwest) == 12
        assert set(midwest) == {"IA", "IL", "IN", "KS", "MI", "MN",
                                "MO", "ND", "NE", "OH", "SD", "WI"}

    def test_kinds_and_regions_compose(self) -> None:
        # Territories in the Caribbean.
        caribbean_terr = ONTOLOGY.state_codes(
            kinds=("territory",), regions=("Caribbean",)
        )
        assert set(caribbean_terr) == {"PR", "VI"}


class TestCesSeriesIdGeneration:
    def test_generates_expected_format(self) -> None:
        sid = ONTOLOGY.ces_series_id("IA", "Manufacturing")
        # Round-trip through parse_series_id.
        spec = ONTOLOGY.parse_series_id(sid)
        assert spec.state.code == "IA"
        assert spec.supersector is not None
        assert spec.supersector.key == "Manufacturing"

    def test_seasonal_flag_prefix(self) -> None:
        sid_s = ONTOLOGY.ces_series_id("IA", "Manufacturing", seasonal="S")
        sid_u = ONTOLOGY.ces_series_id("IA", "Manufacturing", seasonal="U")
        assert sid_s.startswith("SMS")
        assert sid_u.startswith("SMU")

    def test_invalid_seasonal_raises(self) -> None:
        with pytest.raises(ValueError, match="seasonal must be"):
            ONTOLOGY.ces_series_id("IA", "Manufacturing", seasonal="X")

    def test_invalid_datatype_raises(self) -> None:
        with pytest.raises(ValueError, match="datatype must be 2 digits"):
            ONTOLOGY.ces_series_id("IA", "Manufacturing", datatype="001")

    def test_unknown_state_raises(self) -> None:
        with pytest.raises(KeyError):
            ONTOLOGY.ces_series_id("ZZ", "Manufacturing")


class TestLausSeriesIdGeneration:
    def test_labor_force_series_is_twenty_chars(self) -> None:
        sid = ONTOLOGY.laus_series_id("IA", "Labor_Force")
        assert sid == "LASST190000000000006"
        assert len(sid) == 20

    def test_round_trip_via_parser(self) -> None:
        sid = ONTOLOGY.laus_series_id("OH", "Unemployment")
        spec = ONTOLOGY.parse_series_id(sid)
        assert spec.state.code == "OH"
        assert spec.measure.key == "Unemployment"

    def test_measure_without_laus_suffix_raises(self) -> None:
        # LFPR is derived from LF / Population — no LAUS series for it.
        with pytest.raises(ValueError, match="no LAUS suffix"):
            ONTOLOGY.laus_series_id("IA", "LFPR")
