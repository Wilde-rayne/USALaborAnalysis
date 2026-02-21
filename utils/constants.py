import os

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
API_KEY = get_env("BLS_API_KEY", "99da0b2263b74759a9f2160ba748b1e3")#"99da0b2263b74759a9f2160ba748b1e3""c1a89edcea134fbd8ef4d2e220b73e77"
CENSUS_API_KEY = get_env("CENSUS_API_KEY","43250f659bc9de08afa1b0b5a4835ee05774a7be")
OUTPUT_JSON = "data/all_data.json"

MODEL_NAME = OLLAMA_MODEL
LOCAL_EMBED_MODEL = "e5-small-v2"   

# --- Data Pipeline Settings ---
ALL_STATES = [
    "IA", "IL", "IN", "KS", "MI", "MN",
    "MO", "NE", "ND", "OH", "SD", "WI"
]

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
