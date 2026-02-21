import os
import json
import re
import hashlib

import numpy as np
import torch
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from utils.constants import (
    OUTPUT_JSON,
    MONTH_MAP,
    README_PATH,
    REPORT_PATH,
    MILESTONE_PATH,
    TAB_CONTEXT_KEYWORDS,
    LOCAL_EMBED_MODEL
)


_chunks = []
_embs = None
CACHE_EMBEDDINGS = "embeddings_cache.npz"
MAX_TOKENS = 150
BATCH_SIZE = 256
USE_MODEL = LOCAL_EMBED_MODEL
HF_MODELS = {
    "e5-small-v2": "intfloat/e5-small-v2",
    "roberta-base": "sentence-transformers/all-MiniLM-L6-v2"
}
EMBEDDING_DIMENSIONS = {
    "e5-small-v2": 384,
    "roberta-base": 384
}
_model_name = HF_MODELS[USE_MODEL]
_embedding_dim = EMBEDDING_DIMENSIONS[USE_MODEL]
_tokenizer = AutoTokenizer.from_pretrained(_model_name)
_st_model = None


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_text_into_chunks(text, chunk_size=None):
    limit = chunk_size or MAX_TOKENS
    toks = _tokenizer.encode(text, add_special_tokens=False)
    if len(toks) <= limit:
        return [text.strip()]
    out = []
    for i in range(0, len(toks), limit):
        part = _tokenizer.decode(toks[i : i + limit], skip_special_tokens=True)
        if part.strip():
            out.append(part.strip())
    return out


def preprocess_for_embedding(text, context_prefix="In"):
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


def _embed_partition(it):
    global _st_model
    if _st_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _st_model = SentenceTransformer(_model_name, device=device)
    batch = []
    for txt in it:
        txt = txt.strip()
        if not txt:
            continue
        batch.append(txt)
        if len(batch) >= BATCH_SIZE:
            with torch.inference_mode():
                embs = _st_model.encode(batch, normalize_embeddings=True)
            for t, e in zip(batch, embs):
                if e.shape[0] == _embedding_dim:
                    yield t, e.astype(np.float32)
            batch.clear()
    if batch:
        with torch.inference_mode():
            embs = _st_model.encode(batch, normalize_embeddings=True)
        for t, e in zip(batch, embs):
            if e.shape[0] == _embedding_dim:
                yield t, e.astype(np.float32)


def load_embeddings_spark():
    h = hashlib.sha256()
    for p in (OUTPUT_JSON, README_PATH, REPORT_PATH, MILESTONE_PATH):
        if os.path.exists(p):
            h.update(bytes.fromhex(_file_hash(p)))
    combined_hash = h.hexdigest()

    if os.path.exists(CACHE_EMBEDDINGS):
        cache = np.load(CACHE_EMBEDDINGS, allow_pickle=True)
        if cache.get("hash") == combined_hash:
            return cache["chunks"].tolist(), cache["embs"]

    spark = SparkSession.builder \
        .appName("EmbeddingsJob") \
        .master(os.getenv("SPARK_MASTER_URL", "local[*]")) \
        .config("spark.pyspark.driver.python", os.sys.executable) \
        .config("spark.pyspark.python", os.sys.executable) \
        .getOrCreate()
    sc = spark.sparkContext

    raw = sc.wholeTextFiles(OUTPUT_JSON).values()
    json_rdd = raw.flatMap(lambda blob: json.loads(blob)) \
        .flatMap(lambda rec: [
            chunk
            for sent in preprocess_for_embedding(
                "; ".join(
                    f"{k.replace('_',' ')}: {rec[k]}"
                    for k in rec
                    if k not in ("year", "period") and rec[k] is not None
                ),
                context_prefix=(
                    f"In {MONTH_MAP.get(rec['period'], rec['period'])} {rec['year']},"
                )
            )
            for chunk in _split_text_into_chunks(sent)
        ])

    docs = sc.emptyRDD()
    for path in (README_PATH, REPORT_PATH, MILESTONE_PATH):
        if os.path.exists(path):
            docs = docs.union(
                sc.textFile(path)
                  .mapPartitions(lambda it: ["\n".join(it)])
                  .flatMap(_split_text_into_chunks)
            )

    rdd = json_rdd.union(docs).filter(lambda x: x and x.strip())
    rdd = rdd.cache()
    _ = rdd.count()  # materialize
    rdd = rdd.coalesce(min(4, sc.defaultParallelism))

    paired = rdd.mapPartitions(_embed_partition)
    texts, embs = [], []
    for t, e in paired.toLocalIterator():
        texts.append(t)
        embs.append(e)

    embs = np.vstack(embs)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    np.savez_compressed(
        CACHE_EMBEDDINGS,
        hash=combined_hash,
        chunks=np.array(texts, dtype=object),
        embs=embs
    )
    return texts, embs


def load_embeddings():
    global _chunks, _embs
    _chunks, _embs = load_embeddings_spark()


def retrieve_context(query, top_k=3, return_scores=False):
    q = query.strip()
    if not q:
        raise ValueError("Cannot embed empty query.")
    if _embs is None:
        load_embeddings()
    model = SentenceTransformer(_model_name, device="cpu")
    qv = model.encode(q, normalize_embeddings=True).astype(np.float32)
    qv /= np.linalg.norm(qv)
    scores = _embs @ qv
    idxs = np.argsort(-scores)[:top_k]
    if return_scores:
        return [(_chunks[i], float(scores[i])) for i in idxs]
    return "\n\n".join(_chunks[i] for i in idxs)


def get_relevant_context(active_tab, top_k=3):
    if not active_tab:
        return ""
    key = active_tab.strip().lower()
    terms = TAB_CONTEXT_KEYWORDS.get(key, active_tab)
    try:
        return retrieve_context(terms, top_k=top_k)
    except Exception:
        return ""
