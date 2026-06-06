"""Project-wide constants: state set, year range, Ollama settings, file paths."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Pull variables from a project-root .env for local dev.
# By default python-dotenv does not overwrite values already set in the
# environment, so docker-compose / CI secrets still win.
load_dotenv()


def get_env(name, default=None, cast=str):
    """Read ``name`` from the environment, casting to ``cast`` (default ``str``).

    On cast failure, returns ``cast(default)`` so callers always get a
    value of the requested type. Use for environment knobs that need
    numeric coercion (timeouts, port numbers, etc.).
    """
    val = os.getenv(name, default)
    try:
        return cast(val)
    except (ValueError, TypeError):
        return cast(default)


OLLAMA_CHAT_PATH = os.getenv("OLLAMA_CHAT_PATH", "/v1/chat/completions")
OLLAMA_API_PATH = OLLAMA_CHAT_PATH
OLLAMA_URL = get_env("OLLAMA_URL", "http://127.0.0.1:11434")
#: Default LLM used by the dashboard chat & blurb layer. Pinned to
#: Llama 3.2 — the Llama 3.2 Community License governs this service's
#: output (see ``docs/methodology/`` + the "Built with Llama" notice in
#: README and About). Override via the ``OLLAMA_MODEL`` env var if you
#: want to swap in a different chat model at runtime; do NOT downgrade
#: the default to Llama 2 — its license has a 700M-MAU consent clause
#: that 3.2 dropped.
OLLAMA_MODEL = get_env("OLLAMA_MODEL", "llama3.2:3b")
#: HTTP read-timeout for individual Ollama calls. Single-CPU Ollama
#: serializes inference, so a tab that fans out 4 panel blurbs queues
#: behind earlier in-flight calls. 600 s (10 min) is generous enough
#: for ~4-6 queued ~30-60 s calls; the previous 180 s default produced
#: cascade timeouts on the requirements / trend / recap blurbs once
#: the user touched a second tab before the first finished cooking.
DEFAULT_TIMEOUT = get_env("OLLAMA_TIMEOUT", 600, int)
# --- API credentials (see .env.example) ---
# BLS: optional — requests without a key work but are capped at 25 queries/day
# and limited to 10 years per series. Register free at
# https://data.bls.gov/registrationEngine/ for 500 queries/day + 20 years.
# Census: required for ACS/PEP endpoints. Free at
# https://api.census.gov/data/key_signup.html.
BLS_API_KEY = os.getenv("BLS_API_KEY") or None
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY") or None
API_KEY = BLS_API_KEY  # backward-compat alias used by fetch_ces/laus
#: Absolute path to the merged data file. Anchored at the repo root
#: (``utils/`` → ``parents[1]``) so importing from any process CWD
#: still resolves to the same file. Mirrored in ``utils.data_pipeline``;
#: callers should treat this as the source of truth.
OUTPUT_JSON = str(Path(__file__).resolve().parents[1] / "data" / "all_data.json")

LOCAL_EMBED_MODEL = "e5-small-v2"

# --- Data Pipeline Settings ---
# States covered by the dashboard. The default (12 Midwest states) keeps
# the cold-fetch cost bounded; override via environment to scale out:
#
#   STATE_SET=all_states        # 50 states + DC (no territories)
#   STATE_SET=all_jurisdictions # above + PR/VI/GU/AS/MP territories
#   STATE_SET=midwest           # legacy default
#   STATES=IA,IL,CA,TX          # explicit comma-separated codes
#
# Anything else falls back to `midwest` with a warning at import.
def _resolve_state_set() -> list[str]:
    import os  # noqa: PLC0415

    from utils.ontology import ONTOLOGY  # noqa: PLC0415

    explicit = os.getenv("STATES", "").strip()
    if explicit:
        codes = [c.strip().upper() for c in explicit.split(",") if c.strip()]
        # Silently drop unknown codes instead of crashing at import.
        valid = [c for c in codes if c in ONTOLOGY.states]
        if valid:
            return valid

    preset = os.getenv("STATE_SET", "midwest").strip().lower()
    if preset == "all_jurisdictions":
        return ONTOLOGY.state_codes()
    if preset == "all_states":
        return ONTOLOGY.state_codes(kinds=("state", "district"))
    if preset == "midwest":
        return ONTOLOGY.state_codes(regions=("Midwest",))
    # Unknown preset → fall back.
    return ONTOLOGY.state_codes(regions=("Midwest",))


ALL_STATES = _resolve_state_set()

START_YEAR = 1996
END_YEAR = 2024
YEARS = list(range(START_YEAR, END_YEAR + 1))

MONTH_MAP = {
    "M01": "January", "M02": "February", "M03": "March",
    "M04": "April",   "M05": "May",      "M06": "June",
    "M07": "July",    "M08": "August",   "M09": "September",
    "M10": "October", "M11": "November", "M12": "December",
    "A01": "Annual",
}

#: Supersector key list for the UI dropdown. Derived from the ontology
#: (which already mirrors the BLS CES JSON) so the UI cannot drift from
#: the fetcher / merger. The ontology lists 13 supersectors at the
#: state level: Total_Nonfarm, Total_Private, the nine NAICS roll-ups,
#: Leisure_Hospitality, Other_Services, and Government. All are
#: fetched by the CES pipeline and become reachable from the dropdown
#: as soon as the panel includes the matching ``{ST}_{sector}`` column.
def _resolve_supersectors() -> list[str]:
    from utils.ontology import SUPERSECTORS as _ONT_SS  # noqa: PLC0415

    return [s.key for s in _ONT_SS]


SUPERSECTORS = _resolve_supersectors()

README_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "README.md")
REPORT_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "REPORT.md")
MILESTONE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "MILESTONE.md")

TAB_CONTEXT_KEYWORDS = {
    "eda":   "exploratory data analysis overview",
    "lfp":   "labor force participation rate forecast",
    "super": "supersector employment forecast",
    "about": "project background and context",
}
