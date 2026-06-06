"""
Generate ``data/ces_state_sms_codes.json`` and
``data/laus_state_codes.json`` from the ontology.

The two JSONs are the lookup tables the BLS fetchers consult when
building their request payloads. Historically they were hand-edited
for 12 Midwest states; when we extend coverage to all 50 + DC (or
to the territories), we'd rather derive the series ids from a single
source of truth — the ontology — than re-type 200+ rows.

Invoke with::

    python -m scripts.bootstrap_state_codes

Any existing JSON at the target paths is overwritten. Runs are
idempotent — identical inputs always produce identical output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from utils.ontology import ONTOLOGY

CES_OUT = Path("data") / "ces_state_sms_codes.json"
LAUS_OUT = Path("data") / "laus_state_codes.json"


#: Human-readable JSON key → ontology measure key. The JSON format is
#: a historical artefact consumed by ``utils.fetch_laus_data``; the
#: merger downstream normalizes by series-id suffix rather than by
#: these names, so we can stay backward-compatible with the original
#: top-level keys.
LAUS_MEASURE_MAP: dict[str, str] = {
    "Labor_Force_Level":  "Labor_Force",
    "Employment_Level":   "Employment",
    "Unemployment_Level": "Unemployment",
    "Unemployment_Rate":  "Unemployment_Rate",
}


def _sorted_state_codes(include_territories: bool) -> list[str]:
    kinds: Iterable[str] = (
        ("state", "district", "territory")
        if include_territories
        else ("state", "district")
    )
    return ONTOLOGY.state_codes(kinds=kinds)


def build_ces_codes(state_codes: Iterable[str]) -> dict[str, dict[str, str]]:
    """Build the CES code map.

    Parameters
    ----------
    state_codes : Iterable[str]
        USPS codes to expand.

    Returns
    -------
    dict[str, dict[str, str]]
        ``{supersector_key: {state_code: SM series id}}``.
    """
    return {
        key: {st: ONTOLOGY.ces_series_id(st, key) for st in state_codes}
        for key in ONTOLOGY.supersectors
    }


def build_laus_codes(state_codes: Iterable[str]) -> dict[str, dict[str, str]]:
    """Build the LAUS code map.

    Parameters
    ----------
    state_codes : Iterable[str]
        USPS codes to expand.

    Returns
    -------
    dict[str, dict[str, str]]
        ``{measure_json_key: {state_code: LASST series id}}``.
    """
    return {
        json_key: {st: ONTOLOGY.laus_series_id(st, measure_key) for st in state_codes}
        for json_key, measure_key in LAUS_MEASURE_MAP.items()
    }


def main() -> None:
    """Parse CLI args and regenerate the BLS state-code JSONs in-place."""
    parser = argparse.ArgumentParser(description="Regenerate BLS state code JSONs.")
    parser.add_argument(
        "--territories",
        action="store_true",
        help="Include PR/VI/GU/AS/MP territories (not all BLS series exist for them).",
    )
    parser.add_argument(
        "--ces-out",
        default=str(CES_OUT),
        help=f"CES output path (default: {CES_OUT})",
    )
    parser.add_argument(
        "--laus-out",
        default=str(LAUS_OUT),
        help=f"LAUS output path (default: {LAUS_OUT})",
    )
    args = parser.parse_args()

    codes = _sorted_state_codes(include_territories=args.territories)
    print(f"building CES + LAUS code JSONs for {len(codes)} jurisdictions")

    ces = build_ces_codes(codes)
    laus = build_laus_codes(codes)

    Path(args.ces_out).write_text(json.dumps(ces, indent=2, sort_keys=True), encoding="utf-8")
    Path(args.laus_out).write_text(json.dumps(laus, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"wrote {args.ces_out} ({len(ces)} supersectors × {len(codes)} states)"
    )
    print(
        f"wrote {args.laus_out} ({len(laus)} measures × {len(codes)} states)"
    )


if __name__ == "__main__":
    main()
