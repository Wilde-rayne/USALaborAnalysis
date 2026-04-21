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


class TestPreprocessForEmbedding:
    def test_numeric_values_emit_sentence_per_entry(self, emb_module) -> None:
        text = "Labor Force: 1500000; Population: 3150000; state: IA"
        out = emb_module.preprocess_for_embedding(text, context_prefix="In Iowa,")
        # 'state: IA' is non-numeric and must be dropped.
        assert len(out) == 2
        assert out[0] == "In Iowa, Labor Force reported 1500000 thousand jobs."
        assert out[1] == "In Iowa, Population reported 3150000 thousand jobs."

    def test_empty_input_returns_empty_list(self, emb_module) -> None:
        assert emb_module.preprocess_for_embedding("", context_prefix="x") == []

    def test_non_numeric_only_returns_empty_list(self, emb_module) -> None:
        out = emb_module.preprocess_for_embedding("state: IA; period: M01")
        assert out == []

    def test_negative_and_decimal_values_pass_through(self, emb_module) -> None:
        out = emb_module.preprocess_for_embedding("growth: -1.5; rate: 0.25")
        assert len(out) == 2
        assert "-1.5" in out[0]
        assert "0.25" in out[1]


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
        # One sentence per metric in _RAG_METRICS that appears in the data:
        # Labor_Force, LFPR, Population — 3 sentences.
        assert len(chunks) == 3
        assert any("labor force" in c.lower() for c in chunks)
        assert any("lfpr" in c.lower() for c in chunks)
        assert any("population" in c.lower() for c in chunks)
        # LFPR mean = (47.6 + 47.9) / 2 = 47.75
        lfpr_line = next(c for c in chunks if "lfpr" in c.lower())
        assert "47.8" in lfpr_line or "47.75" in lfpr_line

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
