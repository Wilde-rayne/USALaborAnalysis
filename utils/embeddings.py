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

#: Persisted embedding matrix. Lives under HF_HOME (or TRANSFORMERS_CACHE
#: as a fallback) so the non-root container user (added in c4ae737) can
#: actually write it — relative paths resolve to ``/app/`` which is
#: root-owned, which silently broke the cache and caused every chat /
#: blurb call to re-embed 3.6 k chunks from scratch.
_CACHE_DIR = (
    os.environ.get("HF_HOME")
    or os.environ.get("TRANSFORMERS_CACHE")
    or os.path.expanduser("~/.cache")
)
os.makedirs(_CACHE_DIR, exist_ok=True)
#: NPZ holds *only* the float matrix + a SHA hash header — never the
#: chunk strings — so we can load it with ``allow_pickle=False``. The
#: chunk text lives in a sibling JSON file.
CACHE_EMBEDDINGS = os.path.join(_CACHE_DIR, "embeddings_cache.npz")
CACHE_CHUNKS = os.path.join(_CACHE_DIR, "embeddings_cache.json")
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
    """
    Lazy-init the SentenceTransformer; reuse for every call.

    Forces CPU explicitly because (a) the dashboard container has no
    GPU and (b) recent ``torch`` + ``sentence-transformers`` versions
    sometimes load the model with weights on the ``meta`` device
    when CUDA is *probed but unavailable*, then fail with
    ``Cannot copy out of meta tensor; no data!`` on the first
    ``encode()`` — silently breaking RAG retrieval. The post-load
    warm-up forward pass surfaces any remaining materialization
    issues at startup rather than mid-request.
    """
    global _st_model
    if _st_model is None:
        device = torch.device("cpu")
        logger.info(f"[EMB] loading SentenceTransformer({_model_name}) on {device}")
        _st_model = SentenceTransformer(_model_name, device=str(device))
        _st_model.eval()
        # Materialize any meta-device parameters by running a tiny
        # forward pass while we still hold the lazy-init exception
        # surface. If this fails we'd rather crash gunicorn now than
        # have RAG retrieval skip silently from then on.
        try:
            _st_model.encode(["warmup"], show_progress_bar=False, convert_to_numpy=True)
        except NotImplementedError as exc:
            # ``Cannot copy out of meta tensor`` lands here on torch>=2.6
            # with sentence-transformers<3 if the model object held
            # meta-only weights. Re-load with ``low_cpu_mem_usage=False``
            # via the underlying transformers args to force eager load.
            logger.warning(f"[EMB] meta-tensor warm-up failed: {exc}; reloading eager")
            _st_model = SentenceTransformer(
                _model_name,
                device="cpu",
                model_kwargs={"low_cpu_mem_usage": False},
            )
            _st_model.eval()
            _st_model.encode(["warmup"], show_progress_bar=False, convert_to_numpy=True)
        logger.info("[EMB] SentenceTransformer ready")
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


def _apply_e5_prefix(role: str, text: str) -> str:
    """
    Prepend the e5-required role prefix to ``text``.

    ``intfloat/e5-small-v2`` (and the rest of the e5 family) is trained
    with an asymmetric retrieval objective: corpus chunks carry a
    ``"passage: "`` prefix and queries carry a ``"query: "`` prefix.
    Skipping the prefix degrades retrieval by 5-15 % per the model card
    (https://huggingface.co/intfloat/e5-small-v2).

    No-ops for any non-e5 model (e.g. ``all-MiniLM-L6-v2``) so the
    helper is safe to call unconditionally.
    """
    if role not in ("query", "passage"):
        raise ValueError(f"role must be 'query' or 'passage', got {role!r}")
    if not LOCAL_EMBED_MODEL.startswith("e5"):
        return text
    return f"{role}: {text}"


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


#: Version marker mixed into the embedding cache key. Bump this any
#: time the corpus-encoding contract changes (e.g. prefix logic, chunk
#: normalization) so old caches don't load with a now-incompatible
#: model invocation. ``v2`` introduced the ``passage:``/``query:`` e5
#: prefix application — see ``_apply_e5_prefix``.
_EMBED_CACHE_VERSION = "v2-e5prefix"


def _combined_input_hash(paths: Sequence[str]) -> str:
    """
    Deterministic key over existing inputs — drives cache invalidation.

    The active embedding model name AND the corpus-encoding contract
    version are mixed in so that switching ``LOCAL_EMBED_MODEL`` (e.g.
    e5-small-v2 → all-MiniLM-L6-v2) or rotating the prefix logic
    cannot re-use stale vectors from a different setup that happen to
    have the same dimensionality.
    """
    h = hashlib.sha256()
    # Bind the cache to the model that produced it. Without this guard
    # a model swap leaves vectors that look fresh and load silently.
    h.update(b"model=")
    h.update(LOCAL_EMBED_MODEL.encode("utf-8"))
    h.update(b"\nversion=")
    h.update(_EMBED_CACHE_VERSION.encode("utf-8"))
    h.update(b"\n")
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


def _chunks_from_records(records: Iterable[dict]) -> Iterable[str]:
    """
    Turn the monthly panel into RAG-ready sentences via
    :class:`utils.agents.sentence_rag.SentenceRAGBuilder`.

    The builder emits three layers — deterministic facts, ontology-
    enriched per-state rankings, and per-(state, metric) trend
    summaries — all without calling the LLM on the default path. The
    optional ``polish=True`` flag routes each sentence through the
    worker agent (phi3 by default) and is off here because embedding
    rebuilds should be cheap.
    """
    from utils.agents.sentence_rag import SentenceRAGBuilder  # noqa: PLC0415

    builder = SentenceRAGBuilder()
    for sentence in builder.build_corpus(records):
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


def _embed_texts(texts: Sequence[str], role: str = "passage") -> np.ndarray:
    """
    Encode ``texts`` into a (N, dim) float32 array with L2-normalized rows.

    ``role`` is the e5 prefix role — ``"passage"`` for corpus chunks
    (default) or ``"query"`` for a search-time embed call. The prefix
    is no-op for non-e5 backends.
    """
    model = _get_st_model()
    prefixed = [_apply_e5_prefix(role, t) for t in texts]
    with torch.inference_mode():
        raw = model.encode(
            prefixed,
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


def _embed_query(text: str) -> np.ndarray:
    """
    Encode a single query string into a (dim,) float32 vector.

    Always applies the ``"query: "`` prefix (no-op for non-e5 models)
    and L2-normalizes the result so downstream cosine reduces to a dot
    product against the L2-normalized corpus matrix.
    """
    arr = _embed_texts([text], role="query")
    return arr[0]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def load_embeddings() -> tuple[List[str], np.ndarray]:
    """
    Build (or load from cache) the RAG chunk corpus + embeddings.

    Mutates module globals ``_chunks``, ``_embs``, ``_chunk_tokens``.
    Both the corpus matrix and the lexical-overlap token sets are kept;
    ``retrieve_context`` defaults to cosine similarity against ``_embs``
    and only falls back to the token-overlap path when
    ``USE_LEXICAL_RETRIEVAL`` is set or the embedding matrix is empty.
    """
    global _chunks, _embs, _chunk_tokens

    key = _combined_input_hash(
        (OUTPUT_JSON, README_PATH, REPORT_PATH, MILESTONE_PATH)
    )

    if os.path.exists(CACHE_EMBEDDINGS) and os.path.exists(CACHE_CHUNKS):
        # ``allow_pickle=False`` is mandatory: the cache directory is
        # under ``HF_HOME`` (often shared / world-writable on CI hosts),
        # and pickle deserialization would be a remote-code-execution
        # sink. The matrix is plain float32; chunk strings live in the
        # JSON sidecar.
        try:
            cache = np.load(CACHE_EMBEDDINGS, allow_pickle=False)
            # ``.item()`` extracts the Python str/bytes from a 0-d
            # numpy array regardless of whether the dtype is ``U`` or
            # ``S``; it raises if the array is multi-element, which is
            # the desired signal that the cache file is malformed.
            raw_hash = cache["hash"].item()
            cache_key = raw_hash.decode("utf-8") if isinstance(raw_hash, bytes) else str(raw_hash)
            if cache_key == key:
                with open(CACHE_CHUNKS, encoding="utf-8") as f:
                    sidecar = json.load(f)
                if sidecar.get("hash") == key:
                    logger.info(f"[EMB] cache hit: {CACHE_EMBEDDINGS}")
                    _chunks = list(sidecar["chunks"])
                    _embs = np.asarray(cache["embs"], dtype=np.float32)
                    _chunk_tokens = [_tokenize_for_overlap(c) for c in _chunks]
                    return _chunks, _embs
            logger.info(f"[EMB] cache stale ({CACHE_EMBEDDINGS}); rebuilding")
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            logger.warning(
                f"[EMB] cache load failed ({type(exc).__name__}: {exc}); rebuilding"
            )

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

    # Persist matrix + hash *without* pickling Python objects. Chunk
    # text rides alongside in a JSON sidecar.
    np.savez_compressed(
        CACHE_EMBEDDINGS,
        hash=np.asarray(key, dtype="U64"),
        embs=embs,
    )
    with open(CACHE_CHUNKS, "w", encoding="utf-8") as f:
        json.dump({"hash": key, "chunks": chunks}, f)
    logger.info(f"[EMB] saved {embs.shape} → {CACHE_EMBEDDINGS}")
    _chunks, _embs = chunks, embs
    _chunk_tokens = [_tokenize_for_overlap(c) for c in _chunks]
    return _chunks, _embs


def _use_lexical_retrieval() -> bool:
    """
    True if the caller has opted *in* to the lexical-Jaccard fallback.

    The old code path was disabled because torch-vs-tensorflow native
    allocator contention crashed Gunicorn workers that had also trained
    an LSTM earlier in the same request. With the rest of the pipeline
    moving off TensorFlow (LSTM is opt-in, not the default) cosine is
    safe again — but keep an env-gate escape hatch for the deployments
    that still mix TF + torch in one process.
    """
    return os.environ.get("USE_LEXICAL_RETRIEVAL", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _retrieve_lexical(
    query: str, top_k: int, return_scores: bool
):
    """Tokenizer-Jaccard fallback. Used when cosine is disabled or unavailable."""
    q_tokens = _tokenize_for_overlap(query)
    if not q_tokens:
        return [] if return_scores else ""

    scores: list[tuple[int, float]] = []
    for i, ctoks in enumerate(_chunk_tokens or []):
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


def retrieve_context(query: str, top_k: int = 3, return_scores: bool = False):
    """
    Pick top-k relevant chunks via cosine similarity over the cached
    SentenceTransformer embeddings.

    The query is prefixed with ``"query: "`` per the e5 model card and
    embedded once, then cosine-scored against the L2-normalized
    ``_embs`` matrix (cosine reduces to a dot product on normalized
    inputs). Falls back to tokenizer-Jaccard scoring when:

    - ``USE_LEXICAL_RETRIEVAL`` env var is truthy (manual escape hatch
      for deployments where torch + tensorflow share a Gunicorn worker
      and the native allocators fight — see history of this function);
    - ``_embs`` is missing or empty (e.g. ``load_embeddings`` returned
      no chunks);
    - the cosine path raises (defensive — RAG is best-effort).
    """
    q = query.strip()
    if not q:
        raise ValueError("Cannot embed empty query.")

    if _chunk_tokens is None or _embs is None:
        load_embeddings()
    if not _chunks:
        return [] if return_scores else ""

    use_lexical = (
        _use_lexical_retrieval()
        or _embs is None
        or _embs.size == 0
    )
    if use_lexical:
        return _retrieve_lexical(q, top_k=top_k, return_scores=return_scores)

    try:
        q_vec = _embed_query(q)
        # _embs rows are L2-normalized (see _embed_texts); q_vec is too.
        # Cosine = dot product on normalized vectors.
        sims = _embs @ q_vec
        # argsort descending; cap at top_k.
        n = min(top_k, sims.shape[0])
        if n <= 0:
            return [] if return_scores else ""
        # argpartition is O(n) for the top-k slice; full sort only the
        # k-element shortlist.
        idx_part = np.argpartition(-sims, n - 1)[:n]
        idx_sorted = idx_part[np.argsort(-sims[idx_part])]
        top = [(int(i), float(sims[i])) for i in idx_sorted]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"[EMB] cosine retrieval failed ({type(exc).__name__}: {exc}); "
            "falling back to lexical"
        )
        return _retrieve_lexical(q, top_k=top_k, return_scores=return_scores)

    if return_scores:
        return [(_chunks[i], s) for i, s in top]
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
