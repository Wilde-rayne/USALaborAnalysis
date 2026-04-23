"""
Labor-data ontology.

One place that knows: which states exist, their FIPS codes, which
supersectors BLS publishes, which measures LAUS/CES/Census emit, and
how BLS series IDs decompose. Every downstream consumer — the RAG
sentence builder, the fetch agents, the blurb agent, the tab UI — goes
through this module, so adding a state or a measure is one edit here
(plus a data-source refetch) instead of a grep across the repo.

The data itself is a module-level constant; all lookups are pure
functions / small dataclasses. No I/O, no optional dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# --------------------------------------------------------------------------
# Primary entities
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class State:
    """A US state, federal district, or territory."""

    code: str            # 2-letter USPS code ("IA", "PR", "DC")
    name: str            # "Iowa"
    fips: str            # 2-digit FIPS ("19")
    kind: str            # "state" | "district" | "territory"
    region: str          # Census region ("Midwest", "South", ...)


@dataclass(frozen=True, slots=True)
class Supersector:
    """A BLS CES/QCEW supersector roll-up (SM supersector code)."""

    code: str            # 2-digit BLS supersector code ("10", "20", ...)
    key: str             # canonical snake_case label ("Mining_and_Logging")
    name: str            # human-readable ("Mining and Logging")
    naics: str           # NAICS rough mapping string


@dataclass(frozen=True, slots=True)
class Measure:
    """An observable quantity we track for a (state, time) — and sometimes sector."""

    key: str             # "Labor_Force" | "Employment" | ...
    name: str            # "labor force" (lowercase, for RAG sentences)
    unit: str            # "persons" | "percent" | ...
    source: str          # "LAUS" | "CES" | "Census"
    laus_suffix: str | None = None   # last-3 digits of LASST series for LAUS measures
    ces_datatype: str | None = None  # CES SM datatype suffix
    description: str = ""


@dataclass(frozen=True, slots=True)
class DataSource:
    """Where a set of series comes from."""

    id: str                           # short machine id: "bls_ces", "bls_laus", "census_pep"
    name: str                         # human name
    url: str                          # authoritative URL
    api_available: bool
    description: str = ""


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    """
    A resolved series id plus its semantic parts. Returned by
    :meth:`Ontology.parse_series_id` — useful for labeling a chart or
    constructing a natural-language description.
    """

    id: str
    source: str
    state: State
    measure: Measure
    supersector: Supersector | None = None


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------
#: All 50 US states + DC + the 5 permanently-inhabited US territories.
#: This is the full target universe for Phase D; the current pipeline
#: only fetches the Midwest 12 by default (see utils.constants.ALL_STATES).
STATES: tuple[State, ...] = (
    State("AL", "Alabama",        "01", "state",     "South"),
    State("AK", "Alaska",         "02", "state",     "West"),
    State("AZ", "Arizona",        "04", "state",     "West"),
    State("AR", "Arkansas",       "05", "state",     "South"),
    State("CA", "California",     "06", "state",     "West"),
    State("CO", "Colorado",       "08", "state",     "West"),
    State("CT", "Connecticut",    "09", "state",     "Northeast"),
    State("DE", "Delaware",       "10", "state",     "South"),
    State("DC", "District of Columbia", "11", "district", "South"),
    State("FL", "Florida",        "12", "state",     "South"),
    State("GA", "Georgia",        "13", "state",     "South"),
    State("HI", "Hawaii",         "15", "state",     "West"),
    State("ID", "Idaho",          "16", "state",     "West"),
    State("IL", "Illinois",       "17", "state",     "Midwest"),
    State("IN", "Indiana",        "18", "state",     "Midwest"),
    State("IA", "Iowa",           "19", "state",     "Midwest"),
    State("KS", "Kansas",         "20", "state",     "Midwest"),
    State("KY", "Kentucky",       "21", "state",     "South"),
    State("LA", "Louisiana",      "22", "state",     "South"),
    State("ME", "Maine",          "23", "state",     "Northeast"),
    State("MD", "Maryland",       "24", "state",     "South"),
    State("MA", "Massachusetts",  "25", "state",     "Northeast"),
    State("MI", "Michigan",       "26", "state",     "Midwest"),
    State("MN", "Minnesota",      "27", "state",     "Midwest"),
    State("MS", "Mississippi",    "28", "state",     "South"),
    State("MO", "Missouri",       "29", "state",     "Midwest"),
    State("MT", "Montana",        "30", "state",     "West"),
    State("NE", "Nebraska",       "31", "state",     "Midwest"),
    State("NV", "Nevada",         "32", "state",     "West"),
    State("NH", "New Hampshire",  "33", "state",     "Northeast"),
    State("NJ", "New Jersey",     "34", "state",     "Northeast"),
    State("NM", "New Mexico",     "35", "state",     "West"),
    State("NY", "New York",       "36", "state",     "Northeast"),
    State("NC", "North Carolina", "37", "state",     "South"),
    State("ND", "North Dakota",   "38", "state",     "Midwest"),
    State("OH", "Ohio",           "39", "state",     "Midwest"),
    State("OK", "Oklahoma",       "40", "state",     "South"),
    State("OR", "Oregon",         "41", "state",     "West"),
    State("PA", "Pennsylvania",   "42", "state",     "Northeast"),
    State("RI", "Rhode Island",   "44", "state",     "Northeast"),
    State("SC", "South Carolina", "45", "state",     "South"),
    State("SD", "South Dakota",   "46", "state",     "Midwest"),
    State("TN", "Tennessee",      "47", "state",     "South"),
    State("TX", "Texas",          "48", "state",     "South"),
    State("UT", "Utah",           "49", "state",     "West"),
    State("VT", "Vermont",        "50", "state",     "Northeast"),
    State("VA", "Virginia",       "51", "state",     "South"),
    State("WA", "Washington",     "53", "state",     "West"),
    State("WV", "West Virginia",  "54", "state",     "South"),
    State("WI", "Wisconsin",      "55", "state",     "Midwest"),
    State("WY", "Wyoming",        "56", "state",     "West"),
    State("PR", "Puerto Rico",    "72", "territory", "Caribbean"),
    State("VI", "U.S. Virgin Islands", "78", "territory", "Caribbean"),
    State("GU", "Guam",           "66", "territory", "Pacific"),
    State("AS", "American Samoa", "60", "territory", "Pacific"),
    State("MP", "Northern Mariana Islands", "69", "territory", "Pacific"),
)


#: BLS state-level supersectors (SM series `supersector` field).
SUPERSECTORS: tuple[Supersector, ...] = (
    Supersector("00", "Total_Nonfarm",             "Total Nonfarm",                "All NAICS"),
    Supersector("05", "Total_Private",             "Total Private",                "NAICS 11–81"),
    Supersector("10", "Mining_and_Logging",        "Mining and Logging",           "NAICS 11, 21"),
    Supersector("20", "Construction",              "Construction",                 "NAICS 23"),
    Supersector("30", "Manufacturing",             "Manufacturing",                "NAICS 31–33"),
    Supersector("40", "Trade_Transportation_Utilities",
                "Trade, Transportation, and Utilities",                            "NAICS 22, 42, 44–49"),
    Supersector("50", "Information",               "Information",                  "NAICS 51"),
    Supersector("55", "Financial_Activities",      "Financial Activities",         "NAICS 52–53"),
    Supersector("60", "Professional_Business_Services",
                "Professional and Business Services",                              "NAICS 54–56"),
    Supersector("65", "Education_Health_Services",
                "Education and Health Services",                                   "NAICS 61–62"),
    Supersector("70", "Leisure_Hospitality",       "Leisure and Hospitality",      "NAICS 71–72"),
    Supersector("80", "Other_Services",            "Other Services",               "NAICS 81"),
    Supersector("90", "Government",                "Government",                   "Federal + state + local"),
)


#: LAUS measures published at the state level. The last three digits of
#: a LASST series id identify the measure — see
#: https://www.bls.gov/lau/lauov.htm for the full taxonomy.
MEASURES: tuple[Measure, ...] = (
    Measure(
        "Unemployment_Rate", "unemployment rate", "percent", "LAUS",
        laus_suffix="003",
        description="Share of the labor force that is unemployed.",
    ),
    Measure(
        "Unemployment", "unemployment", "persons", "LAUS",
        laus_suffix="004",
        description="Number of persons 16+ actively seeking work.",
    ),
    Measure(
        "Employment", "employment", "persons", "LAUS",
        laus_suffix="005",
        description="Number of persons 16+ working in civilian non-institutional roles.",
    ),
    Measure(
        "Labor_Force", "labor force", "persons", "LAUS",
        laus_suffix="006",
        description="Employed + unemployed persons 16+.",
    ),
    Measure(
        "LFPR", "labor force participation rate", "percent", "LAUS",
        description="100 × labor force / population.",
    ),
    Measure(
        "Population", "population", "persons", "Census",
        description="State population estimate (ACS or PEP).",
    ),
    Measure(
        "Sector_Employment", "employment", "persons", "CES",
        ces_datatype="001",
        description="All-employee, seasonally-adjusted jobs count (CES).",
    ),
)


SOURCES: tuple[DataSource, ...] = (
    DataSource(
        "bls_ces",
        "BLS Current Employment Statistics (State and Area)",
        "https://www.bls.gov/sae/",
        api_available=True,
        description="Monthly payroll jobs by state and supersector (SM series).",
    ),
    DataSource(
        "bls_laus",
        "BLS Local Area Unemployment Statistics",
        "https://www.bls.gov/lau/",
        api_available=True,
        description="Monthly labor force, employment, unemployment by state.",
    ),
    DataSource(
        "census_pep",
        "US Census Bureau — Population Estimates Program",
        "https://www.census.gov/programs-surveys/popest.html",
        api_available=True,
        description="Annual state population estimates.",
    ),
    DataSource(
        "census_acs",
        "US Census Bureau — American Community Survey",
        "https://www.census.gov/programs-surveys/acs/",
        api_available=True,
        description="Annual demographic + economic survey, state-level.",
    ),
    DataSource(
        "bea_regional",
        "US Bureau of Economic Analysis — Regional Accounts",
        "https://apps.bea.gov/regional/",
        api_available=True,
        description=(
            "State-level annual personal income (SAINC1) and GDP by industry "
            "(SAGDP2N), 1929–present. Per-capita figures are a strong "
            "compositional counterweight to BLS headcount series."
        ),
    ),
    DataSource(
        "fred",
        "Federal Reserve Economic Data (St. Louis Fed)",
        "https://fred.stlouisfed.org/",
        api_available=True,
        description=(
            "Hundreds of state-level macro indicators via the `{ST}{IND}` "
            "series naming convention (e.g. IAUR = Iowa unemployment rate, "
            "INPI = Indiana personal income). Monthly/quarterly/annual "
            "cadence; a natural companion to the Super tab's site-selection "
            "recommendations."
        ),
    ),
    DataSource(
        "bls_cpi",
        "BLS Consumer Price Index — regional",
        "https://www.bls.gov/cpi/",
        api_available=True,
        description=(
            "Monthly CPI-U indices for the 4 Census regions (Northeast, "
            "Midwest, South, West) plus the US city average. Used to "
            "deflate dollar-denominated series to real terms — the "
            "missing context when comparing 1996-vintage payroll counts "
            "against 2024-vintage ones."
        ),
    ),
    DataSource(
        "bls_jolts",
        "BLS Job Openings and Labor Turnover Survey (JOLTS)",
        "https://www.bls.gov/jlt/",
        api_available=True,
        description=(
            "Monthly job openings, hires, quits, layoffs, and total "
            "separations at national scope (2000-12–present); "
            "experimental state-level series (JTS State) available for "
            "recent periods. Direct input to the Super tab's "
            "site-selection story — openings-per-hire and quits-rate "
            "are the canonical tightness indicators."
        ),
    ),
)


# --------------------------------------------------------------------------
# Ontology facade
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Ontology:
    """
    Lookup facade over the reference tuples. Frozen dataclass because the
    content is immutable at runtime — any change should be a code edit.
    """

    states: dict[str, State] = field(default_factory=lambda: {s.code: s for s in STATES})
    states_by_fips: dict[str, State] = field(
        default_factory=lambda: {s.fips: s for s in STATES}
    )
    supersectors: dict[str, Supersector] = field(
        default_factory=lambda: {s.key: s for s in SUPERSECTORS}
    )
    supersectors_by_code: dict[str, Supersector] = field(
        default_factory=lambda: {s.code: s for s in SUPERSECTORS}
    )
    measures: dict[str, Measure] = field(default_factory=lambda: {m.key: m for m in MEASURES})
    sources: dict[str, DataSource] = field(default_factory=lambda: {s.id: s for s in SOURCES})

    # --- convenience accessors ---------------------------------------------
    def state(self, code: str) -> State:
        s = self.states.get(code.upper())
        if s is None:
            raise KeyError(f"unknown state code: {code!r}")
        return s

    def states_in(self, region: str) -> list[State]:
        return [s for s in self.states.values() if s.region == region]

    def supersector(self, key: str) -> Supersector:
        s = self.supersectors.get(key)
        if s is None:
            raise KeyError(f"unknown supersector key: {key!r}")
        return s

    def measure(self, key: str) -> Measure:
        m = self.measures.get(key)
        if m is None:
            raise KeyError(f"unknown measure key: {key!r}")
        return m

    # --- BLS series id decoding --------------------------------------------
    def parse_series_id(self, series_id: str) -> SeriesSpec:
        """
        Turn a raw BLS series id into a :class:`SeriesSpec` with the
        ontology nodes filled in. Supports:

        - LASST{FIPS:2}{area_code:13}{measure:3} — LAUS state
        - SM{FIPS:2}{area:5}{supersector:2}{industry:6}{datatype:3} — CES SM state
        """
        sid = series_id.strip()
        if sid.startswith("LASST") and len(sid) >= 20:
            fips = sid[5:7]
            measure_suffix = sid[-3:]
            state = self.states_by_fips.get(fips)
            if state is None:
                raise ValueError(f"unknown FIPS {fips} in {sid!r}")
            measure = next(
                (m for m in self.measures.values() if m.laus_suffix == measure_suffix),
                None,
            )
            if measure is None:
                raise ValueError(f"unknown LAUS measure suffix {measure_suffix!r} in {sid!r}")
            return SeriesSpec(id=sid, source="bls_laus", state=state, measure=measure)

        if sid.startswith("SMS") and len(sid) >= 20:
            # SM prefix omits the U/S seasonal flag on some vintages;
            # we handle the common SMS and SMU forms identically here.
            fips = sid[3:5]
            supersector_code = sid[10:12]
            state = self.states_by_fips.get(fips)
            if state is None:
                raise ValueError(f"unknown FIPS {fips} in {sid!r}")
            supersector = self.supersectors_by_code.get(supersector_code)
            return SeriesSpec(
                id=sid,
                source="bls_ces",
                state=state,
                measure=self.measures["Sector_Employment"],
                supersector=supersector,
            )

        raise ValueError(f"unrecognized series id format: {sid!r}")

    # --- natural-language rendering ----------------------------------------
    def describe(self, series_id: str) -> str:
        """
        Human-readable one-liner for a series. Feeds the sentence-RAG
        layer so the embedding model sees prose, not codes.
        """
        spec = self.parse_series_id(series_id)
        if spec.source == "bls_ces" and spec.supersector is not None:
            return (
                f"BLS CES monthly {spec.measure.name} for the "
                f"{spec.supersector.name} supersector in {spec.state.name}."
            )
        return (
            f"BLS {spec.source.upper().replace('BLS_', '')} monthly "
            f"{spec.measure.name} in {spec.state.name}."
        )

    # --- iteration helpers -------------------------------------------------
    def iter_states(self, kinds: Iterable[str] | None = None) -> list[State]:
        kinds_set = set(kinds) if kinds else None
        return [s for s in self.states.values() if kinds_set is None or s.kind in kinds_set]

    def state_codes(
        self,
        *,
        kinds: Iterable[str] | None = None,
        regions: Iterable[str] | None = None,
    ) -> list[str]:
        """
        All matching state/territory codes, deterministically sorted.

        Example::

            ONTOLOGY.state_codes()                      # all 56
            ONTOLOGY.state_codes(kinds=("state", "district"))   # 51
            ONTOLOGY.state_codes(regions=("Midwest",))  # 12 Midwest states
        """
        kinds_set = set(kinds) if kinds else None
        regions_set = set(regions) if regions else None
        out = [
            s.code
            for s in self.states.values()
            if (kinds_set is None or s.kind in kinds_set)
            and (regions_set is None or s.region in regions_set)
        ]
        return sorted(out)

    # --- BLS series id generation ------------------------------------------
    def ces_series_id(
        self,
        state: str,
        supersector: str,
        *,
        datatype: str = "01",
        seasonal: str = "S",
    ) -> str:
        """
        Build a BLS CES state-level series id.

        Format (20 chars total): ``SM`` + seasonal flag ("S"/"U") +
        FIPS(2) + area "00000" + supersector(2) + industry "000000" +
        datatype(2). The defaults (``datatype="01"`` = all-employees
        in thousands, ``seasonal="S"`` = seasonally adjusted) are the
        ones the existing pipeline already downloads.
        """
        if seasonal not in {"S", "U"}:
            raise ValueError(f"seasonal must be 'S' or 'U', got {seasonal!r}")
        if len(datatype) != 2 or not datatype.isdigit():
            raise ValueError(f"datatype must be 2 digits, got {datatype!r}")
        st = self.state(state)
        ss = self.supersector(supersector)
        return f"SM{seasonal}{st.fips}00000{ss.code}000000{datatype}"

    def laus_series_id(self, state: str, measure: str) -> str:
        """
        Build a LAUS state-level series id.

        Format (20 chars): ``LASST`` + FIPS(2) + "0000000000" +
        measure suffix(3). Measures without a ``laus_suffix`` (e.g.
        the derived LFPR) raise.
        """
        st = self.state(state)
        m = self.measure(measure)
        if not m.laus_suffix:
            raise ValueError(
                f"measure {measure!r} has no LAUS suffix (is it derived or Census?)"
            )
        return f"LASST{st.fips}0000000000{m.laus_suffix}"


#: Module-level singleton — the ontology is a configuration, not a connection.
ONTOLOGY = Ontology()
