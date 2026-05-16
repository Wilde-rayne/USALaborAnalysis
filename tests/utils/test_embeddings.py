"""
Unit tests for pure helpers in ``utils.embeddings``.

Skips the full ``load_embeddings`` pipeline (needs a downloaded
SentenceTransformer model) — those are covered in the integration
suite. Here we lock in the chunking + preprocessing logic so a
refactor can't silently change how RAG input is shaped.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

# These tests exercise ``utils.embeddings`` which imports torch +
# sentence_transformers + transformers at module scope. Skip the whole
# file when any of those are missing (lightweight CI, lightweight local
# dev) rather than failing at collection.
pytest.importorskip("sentence_transformers")
pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def emb_module():
    """
    Import ``utils.embeddings`` once per test module. The import loads a
    Hugging Face tokenizer at module scope, which is cheap after the
    first call but not free — so keep it module-scoped.
    """
    return importlib.import_module("utils.embeddings")


class TestSplitTextIntoChunks:
    def test_short_text_returns_single_chunk(self, emb_module) -> None:
        out = emb_module._split_text_into_chunks("Iowa labor market is stable.")
        assert out == ["Iowa labor market is stable."]

    def test_empty_text_returns_empty(self, emb_module) -> None:
        assert emb_module._split_text_into_chunks("") == []

    def test_long_text_is_sliced_into_pieces(self, emb_module) -> None:
        # Build something comfortably past MAX_TOKENS so the loop activates.
        text = " ".join(["word"] * 400)
        out = emb_module._split_text_into_chunks(text, chunk_size=50)
        assert len(out) >= 2
        # No empty pieces slipped through.
        assert all(p.strip() for p in out)


class TestCollectChunks:
    def test_empty_when_no_files(self, emb_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no data JSON and no doc paths, we get zero chunks."""
        monkeypatch.setattr(emb_module, "OUTPUT_JSON", str(tmp_path / "missing.json"))
        monkeypatch.setattr(emb_module, "README_PATH", str(tmp_path / "missing-a.md"))
        monkeypatch.setattr(emb_module, "REPORT_PATH", str(tmp_path / "missing-b.md"))
        monkeypatch.setattr(emb_module, "MILESTONE_PATH", str(tmp_path / "missing-c.md"))
        assert emb_module._collect_chunks() == []

    def test_extracts_aggregated_from_json_records(
        self, emb_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two Jan+Feb rows for IA collapse into one sentence per metric."""
        data_path = tmp_path / "all_data.json"
        data_path.write_text(
            json.dumps(
                [
                    {
                        "state": "IA",
                        "year": 2020,
                        "period": "M01",
                        "Labor_Force": 1500000,
                        "Population": 3150000,
                        "LFPR": 47.6,
                    },
                    {
                        "state": "IA",
                        "year": 2020,
                        "period": "M02",
                        "Labor_Force": 1510000,
                        "Population": 3150000,
                        "LFPR": 47.9,
                    },
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(emb_module, "OUTPUT_JSON", str(data_path))
        monkeypatch.setattr(emb_module, "README_PATH", str(tmp_path / "none.md"))
        monkeypatch.setattr(emb_module, "REPORT_PATH", str(tmp_path / "none.md"))
        monkeypatch.setattr(emb_module, "MILESTONE_PATH", str(tmp_path / "none.md"))

        chunks = emb_module._collect_chunks()
        # SentenceRAGBuilder emits one "fact" sentence per (state, year,
        # metric-present-in-record). The sample record has Labor_Force,
        # Population, AND LFPR → 3 sentences. Ranking/trend layers stay
        # empty here (single state, single year).
        assert len(chunks) == 3
        assert any("labor force" in c.lower() for c in chunks)
        assert any("population" in c.lower() for c in chunks)
        # LFPR → ontology label is "labor force participation rate".
        lfpr_line = next(c for c in chunks if "participation rate" in c.lower())
        # Monthly mean = (47.6 + 47.9) / 2 = 47.75; format spec "{:.1f}%" → 47.8.
        assert "47.8" in lfpr_line

    def test_extracts_from_markdown_docs(
        self, emb_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# Hello\n\nSome intro text about the Midwest labor market.", encoding="utf-8")
        monkeypatch.setattr(emb_module, "OUTPUT_JSON", str(tmp_path / "none.json"))
        monkeypatch.setattr(emb_module, "README_PATH", str(readme))
        monkeypatch.setattr(emb_module, "REPORT_PATH", str(tmp_path / "none.md"))
        monkeypatch.setattr(emb_module, "MILESTONE_PATH", str(tmp_path / "none.md"))

        chunks = emb_module._collect_chunks()
        assert chunks
        assert any("Midwest" in c for c in chunks)
