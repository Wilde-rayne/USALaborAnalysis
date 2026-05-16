"""
fetch_ces_data.py

Download CES statewide “SMS” series for the selected states.

JSON expected (see data/ces_state_sms_codes.json):
{
  "Mining_and_Logging":          { "IA": "SMS19000001000000001", ... },
  "Construction":                { "IA": "SMS19000002000000001", ... },
  ...
}
"""
import json, math, os, sys
from typing import List
import requests
import logging
from .constants import API_KEY

logger = logging.getLogger(__name__)

API_URL   = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
HEADERS   = {"Content-Type": "application/json"}

CES_JSON  = "data/ces_state_sms_codes.json"
RAW_DIR   = "data/raw/ces"
BATCH_SZ  = 50

def _load_json() -> dict:
    if not os.path.exists(CES_JSON):
        logger.error(f"Missing {CES_JSON}")
        raise FileNotFoundError(f"Missing {CES_JSON}")
    with open(CES_JSON, encoding="utf-8") as f:
        return json.load(f)

def _series_ids(states: List[str]) -> List[str]:
    """
    Collect every SMS… code in the JSON that matches one of the wanted
    states, de‑duplicate, and ignore anything that does *not* start with
    “SMS” (just in case a stray POP_… or LASST… slipped in).
    """
    ids: list[str] = []
    data = _load_json()
    for supersec, st_map in data.items():
        for st in states:
            sid = st_map.get(st)
            if not sid:
                logger.warning(f"[CES] no code for {st} in supersector {supersec}")
                continue
            if sid.startswith("SMS"):
                ids.append(sid)
            else:
                logger.warning(f"[CES] {sid} is not an SMS code – skipped")
    return sorted(set(ids))

def fetch_ces_data(states: List[str], start: int, end: int) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    for fn in os.listdir(RAW_DIR):
        if fn.endswith(".txt"):
            os.remove(os.path.join(RAW_DIR, fn))

    ids = _series_ids(states)
    if not ids:
        logger.warning("[CES] No valid SMS series IDs to fetch – aborting.")
        return

    n_batches = math.ceil(len(ids) / BATCH_SZ)
    logger.info(f"[CES] Requesting {len(ids)} series in {n_batches} batch(es)…")

    for b, i in enumerate(range(0, len(ids), BATCH_SZ), start=1):
        payload = {
            "seriesid":  ids[i : i + BATCH_SZ],
            "startyear": str(start),
            "endyear":   str(end),
            "registrationkey": API_KEY,
        }
        try:
            r = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload), timeout=30)
        except Exception as e:
            logger.error(f"[CES] Batch {b}: request failed – {e}")
            continue
        if r.status_code != 200:
            logger.error(f"[CES] Batch {b}: HTTP {r.status_code} – {r.text}")
            continue

        series = r.json().get("Results", {}).get("series", [])
        if not series:
            logger.warning(f"[CES] Batch {b}: empty 'series' list (IDs bad?)")
            continue
        _save_series(series)

def _save_series(series_list: List[dict]) -> None:
    for s in series_list:
        sid   = s.get("seriesID", "")
        data  = s.get("data", [])
        if not data:
            logger.warning(f"[CES] {sid} returned no data.")
            continue
        path = os.path.join(RAW_DIR, f"{sid}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("series_id,year,period,value\n")
            for pt in reversed(data):  # oldest → newest
                f.write(f"{sid},{pt['year']},{pt['period']},{pt['value']}\n")
        logger.info(f"[CES] Saved {path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage:  python fetch_ces_data.py IA,IL,WI 1996 2024")
        sys.exit(1)
    states = sys.argv[1].upper().split(",")
    fetch_ces_data(states, int(sys.argv[2]), int(sys.argv[3]))
