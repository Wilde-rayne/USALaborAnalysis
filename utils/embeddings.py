"""
RAG embedding pipeline for the chat assistant.

In-process batched encoding with ``sentence_transformers``. The prior
Spark ``local[*]`` implementation hung on first run under tight Docker
memory (llama2:chat + Spark + torch fighting for ~8 GB); dropping the
cluster removes the moving part entirely. For ~3k snippets the
end-to-end build is ~10-20 s on CPU.

Public API (unchanged for callers):
- ``load_embeddings()`` — build (or load from cache) the full corpus.
- ``retrieve_context(query, top_k)`` — top-k cosine retrieval.
- ``get_relevant_context(active_tab, top_k)`` — tab-keyword shortcut.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Iterable, List, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Pin torch's CPU intra-op threading to 1. On python:3.11-slim with
# tensorflow 2.21 + torch 2.11 + sentence_transformers 2.7 the worker
# process' native allocator gets wedged when torch kernels are invoked
# from a gthread worker *after* tensorflow has built an LSTM earlier in
# the same request. Symptom: ``free(): invalid pointer`` / SIGABRT.
# Single-thread torch is plenty for a 384-dim e5-small workload.
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    # set_num_interop_threads raises if already set by an earlier import.
    pass

from utils.constants import (
    LOCAL_EMBED_MODEL,
    MILESTONE_PATH,
    MONTH_MAP,
    OUTPUT_JSON,
    README_PATH,
    REPORT_PATH,
    TAB_CONTEXT_KEYWORDS,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Module state
# --------------------------------------------------------------------------
_chunks: List[str] = []
_embs: np.ndarray | None = None
_chunk_tokens: List[set[str]] | None = None
_st_model: SentenceTransformer | None = None

CACHE_EMBEDDINGS = "embeddings_cache.npz"
MAX_TOKENS = 150
BATCH_SIZE = 64

HF_MODELS = {
    "e5-small-v2": "intfloat/e5-small-v2",
    "roberta-base": "sentence-transformers/all-MiniLM-L6-v2",
}
EMBEDDING_DIMENSIONS = {
    "e5-small-v2": 384,
    "roberta-base": 384,
}
USE_MODEL = LOCAL_EMBED_MODEL
_model_name = HF_MODELS[USE_MODEL]
_embedding_dim = EMBEDDING_DIMENSIONS[USE_MODEL]
_tokenizer = AutoTokenizer.from_pretrained(_model_name)


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------
def _get_st_model() -> SentenceTransformer:
    """Lazy-init the SentenceTransformer; reuse for every call."""
    global _st_model
    if _st_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[EMB] loading SentenceTransformer({_model_name}) on {device}")
        _st_model = SentenceTransformer(_model_name, device=device)
    return _st_model


def _tokenize_for_overlap(text: str) -> set[str]:
    """
    Lowercase token-set used for query-time lexical overlap scoring.
    Drops subword markers so "##ment" matches "employment".
    """
    toks = _tokenizer.tokenize(text.lower())
    clean = {t.lstrip("#") for t in toks if len(t.lstrip("#")) > 1}
    # Keep alphanumeric-only to avoid matching punctuation-like tokens.
    return {t for t in clean if any(c.isalnum() for c in t)}


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _combined_input_hash(paths: Sequence[str]) -> str:
    """Deterministic key over existing inputs — drives cache invalidation."""
    h = hashlib.sha256()
    for p in paths:
        if os.path.exists(p):
            h.update(bytes.fromhex(_file_hash(p)))
    return h.hexdigest()


def _split_text_into_chunks(text: str, chunk_size: int | None = None) -> List[str]:
    """Token-aware splitter. Never emits empty strings."""
    limit = chunk_size or MAX_TOKENS
    toks = _tokenizer.encode(text, add_special_tokens=False)
    if len(toks) <= limit:
        s = text.strip()
        return [s] if s else []
    out = []
    for i in range(0, len(toks), limit):
        part = _tokenizer.decode(toks[i : i + limit], skip_special_tokens=True)
        if part.strip():
            out.append(part.strip())
    return out


def preprocess_for_embedding(text: str, context_prefix: str = "In") -> List[str]:
    """
    Turn a "key: value; key: value" record into natural-language sentences.
    Only keeps numeric values so the embedder sees "reported N thousand jobs"
    phrases rather than raw key-value blobs.
    """
    out = []
    for entry in text.split(";"):
        entry = entry.strip()
        if ":" not in entry:
            continue
        key, val = map(str.strip, entry.split(":", 1))
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", val):
            continue
        out.append(f"{context_prefix} {key} reported {val} thousand jobs.")
    return out


# Metrics worth embedding: the narrative layer for the chat. Everything
# else on the panel (per-state sector columns, etc.) blows the chunk
# count into the hundreds of thousands without adding discriminating
# signal for retrieval. Adjust this tuple if the assistant needs to
# answer questions about a new dimension.
_RAG_METRICS: tuple[str, ...] = (
    "Labor_Force",
    "Employment",
    "Unemployment",
    "LFPR",
    "Population",
)


def _chunks_from_records(records: Iterable[dict]) -> Iterable[str]:
    """
    Collapse the monthly panel into one sentence per (state, year, metric).

    The raw panel is ~4k rows × 200 wide columns; naive per-row encoding
    produces ~700k chunks which is both slow to embed and poor signal —
    most of those columns repeat information across states. Aggregating
    to an annual mean per metric yields ~12 states × 30 years × 5
    metrics ≈ 1.8k sentences that are readable by the chat model and
    cheap to retrieve.
    """
    from collections import defaultdict

    agg: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rec in records:
        state = rec.get("state")
        year = rec.get("year")
        if not state or year is None:
            continue
        for metric in _RAG_METRICS:
            val = rec.get(metric)
            if val is None:
                continue
            try:
                agg[(state, int(year))][metric].append(float(val))
            except (TypeError, ValueError):
                continue

    for (state, year), metrics in sorted(agg.items()):
        for metric in _RAG_METRICS:
            values = metrics.get(metric)
            if not values:
                continue
            avg = sum(values) / len(values)
            readable = metric.replace("_", " ").lower()
            sentence = f"In {year}, {state} {readable} averaged {avg:,.1f}."
            yield from _split_text_into_chunks(sentence)


def _chunks_from_doc(path: str) -> Iterable[str]:
    with open(path, encoding="utf-8") as f:
        yield from _split_text_into_chunks(f.read())


def _collect_chunks() -> List[str]:
    """Pull text chunks from merged JSON data + auxiliary markdown docs."""
    chunks: List[str] = []

    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, encoding="utf-8") as f:
            records = json.load(f)
        chunks.extend(_chunks_from_records(records))

    for path in (README_PATH, REPORT_PATH, MILESTONE_PATH):
        if os.path.exists(path):
            chunks.extend(_chunks_from_doc(path))

    return [c for c in chunks if c and c.strip()]


def _embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Encode ``texts`` into a (N, dim) float32 array with L2-normalized rows."""
    model = _get_st_model()
    with torch.inference_mode():
        raw = model.encode(
            list(texts),
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    arr = np.asarray(raw, dtype=np.float32)
    # SentenceTransformer normalizes already, but guard in case a backend
    # returns unnormalized rows.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def load_embeddings() -> tuple[List[str], np.ndarray]:
    """
    Build (or load from cache) the RAG chunk corpus + embeddings.

    Mutates module globals ``_chunks``, ``_embs``, ``_chunk_tokens``.
    The embeddings are retained on disk so future semantic-rerank
    callers can use them, but the default retrieval path
    (``retrieve_context``) is token-overlap based — see that function
    for the "why".
    """
    global _chunks, _embs, _chunk_tokens

    key = _combined_input_hash(
        (OUTPUT_JSON, README_PATH, REPORT_PATH, MILESTONE_PATH)
    )

    if os.path.exists(CACHE_EMBEDDINGS):
        cache = np.load(CACHE_EMBEDDINGS, allow_pickle=True)
        if str(cache.get("hash")) == key:
            logger.info(f"[EMB] cache hit: {CACHE_EMBEDDINGS}")
            _chunks = cache["chunks"].tolist()
            _embs = cache["embs"]
            _chunk_tokens = [_tokenize_for_overlap(c) for c in _chunks]
            return _chunks, _embs
        logger.info(f"[EMB] cache stale ({CACHE_EMBEDDINGS}); rebuilding")

    logger.info("[EMB] collecting chunks")
    chunks = _collect_chunks()
    if not chunks:
        logger.warning("[EMB] no chunks collected — data/docs missing?")
        _chunks = []
        _embs = np.zeros((0, _embedding_dim), dtype=np.float32)
        _chunk_tokens = []
        return _chunks, _embs

    logger.info(
        f"[EMB] embedding {len(chunks)} chunks (batch={BATCH_SIZE}, "
        f"model={_model_name})"
    )
    embs = _embed_texts(chunks)

    np.savez_compressed(
        CACHE_EMBEDDINGS,
        hash=key,
        chunks=np.array(chunks, dtype=object),
        embs=embs,
    )
    logger.info(f"[EMB] saved {embs.shape} → {CACHE_EMBEDDINGS}")
    _chunks, _embs = chunks, embs
    _chunk_tokens = [_tokenize_for_overlap(c) for c in _chunks]
    return _chunks, _embs


def retrieve_context(query: str, top_k: int = 3, return_scores: bool = False):
    """
    Pick top-k relevant chunks via tokenizer-based lexical overlap.

    NOTE: a previous version cosine-matched the query against the stored
    SentenceTransformer embeddings. That path crashed with
    ``free(): invalid pointer`` whenever a Gunicorn worker had already
    trained an LSTM (via tensorflow) earlier in the same request —
    torch-vs-tensorflow native allocator contention. The fallback here
    uses only the tokenizer (no torch at query time) and still gives
    reasonable results on a 1.8k-chunk corpus; we can swap back to
    semantic retrieval once the workers move to a process-based
    worker class or the TF/torch coexistence is fixed.
    """
    q = query.strip()
    if not q:
        raise ValueError("Cannot embed empty query.")

    if _chunk_tokens is None:
        load_embeddings()
    if not _chunks or not _chunk_tokens:
        return [] if return_scores else ""

    q_tokens = _tokenize_for_overlap(q)
    if not q_tokens:
        return [] if return_scores else ""

    scores: list[tuple[int, float]] = []
    for i, ctoks in enumerate(_chunk_tokens):
        if not ctoks:
            continue
        overlap = len(q_tokens & ctoks)
        if overlap == 0:
            continue
        # Jaccard-ish score normalized by sqrt(|chunk|) so longer chunks
        # don't dominate and very short chunks don't win by accident.
        score = overlap / (len(ctoks) ** 0.5)
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_k]

    if return_scores:
        return [(_chunks[i], float(s)) for i, s in top]
    return "\n\n".join(_chunks[i] for i, _ in top)


def get_relevant_context(active_tab: str, top_k: int = 3) -> str:
    if not active_tab:
        return ""
    key = active_tab.strip().lower()
    terms = TAB_CONTEXT_KEYWORDS.get(key, active_tab)
    try:
        return retrieve_context(terms, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 — RAG is best-effort in callbacks
        logger.warning(f"[EMB] retrieve_context failed: {exc}")
        return ""
