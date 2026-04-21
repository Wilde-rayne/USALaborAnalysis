import os
from dotenv import load_dotenv

# Pull variables from a project-root .env for local dev.
# By default python-dotenv does not overwrite values already set in the
# environment, so docker-compose / CI secrets still win.
load_dotenv()


def get_env(name, default=None, cast=str):
    val = os.getenv(name, default)
    try:
        return cast(val)
    except (ValueError, TypeError):
        return cast(default)

OLLAMA_CHAT_PATH  = os.getenv("OLLAMA_CHAT_PATH",  "/v1/chat/completions")
OLLAMA_EMBED_PATH = os.getenv("OLLAMA_EMBED_PATH", "/api/embeddings")
OLLAMA_API_PATH   = OLLAMA_CHAT_PATH
OLLAMA_URL       = get_env("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL     = get_env("OLLAMA_MODEL", "llama2:chat")
DEFAULT_TIMEOUT  = get_env("OLLAMA_TIMEOUT", 180, int)
# --- API credentials (see .env.example) ---
# BLS: optional — requests without a key work but are capped at 25 queries/day
# and limited to 10 years per series. Register free at
# https://data.bls.gov/registrationEngine/ for 500 queries/day + 20 years.
# Census: required for ACS/PEP endpoints. Free at
# https://api.census.gov/data/key_signup.html.
BLS_API_KEY    = os.getenv("BLS_API_KEY") or None
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY") or None
API_KEY        = BLS_API_KEY  # backward-compat alias used by fetch_ces/laus
OUTPUT_JSON = "data/all_data.json"

MODEL_NAME = OLLAMA_MODEL
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
END_YEAR   = 2024
YEARS = list(range(START_YEAR, END_YEAR + 1))

MONTH_MAP = {
    "M01": "January", "M02": "February", "M03": "March",
    "M04": "April",   "M05": "May",      "M06": "June",
    "M07": "July",    "M08": "August",   "M09": "September",
    "M10": "October", "M11": "November","M12": "December",
    "A01": "Annual"
}

SUPERSECTORS = [
    "Mining_and_Logging",
    "Construction",
    "Manufacturing",
    "Trade_Transportation_Utilities",
    "Information",
    "Financial_Activities",
    "Professional_Business_Services",
    "Education_Health_Services",
    "Government"
]

README_PATH    = os.path.join(os.path.dirname(__file__), os.pardir, "README.md")
REPORT_PATH    = os.path.join(os.path.dirname(__file__), os.pardir, "REPORT.md")
MILESTONE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 
"MILESTONE.md")

TAB_CONTEXT_KEYWORDS = {
    "eda":   "exploratory data analysis overview",
    "lfp":   "labor force participation rate forecast",
    "super": "supersector employment forecast",
    "about": "project background and context"
}
