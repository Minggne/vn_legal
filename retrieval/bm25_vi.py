"""Sparse BM25 cho Qdrant. Ingest và truy vấn dùng chung tokenizer/config.

Token được băm thành chiều sparse; Qdrant nhân IDF khi collection bật modifier.
Thay tokenizer hoặc tham số BM25 cần tạo lại sparse vector trong dữ liệu.
"""

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter

from qdrant_client import models


CONFIG_FILE = os.path.join("data", "bm25_config.json")
DEFAULTS = {"k1": 1.2, "b": 0.75, "avgdl": 200.0, "bigram": True}
STOP = frozenset("và của là các những một có được cho trong khi với theo để này đó thì bị như về từ".split())
_WORD = re.compile(r"\w+", re.UNICODE)
_cfg = None


# =========================================================
# CONFIGURATION
# =========================================================

def load_config(refresh=False):
    global _cfg
    if _cfg is None or refresh:
        _cfg = DEFAULTS.copy()
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, encoding="utf-8") as f:
                _cfg.update(json.load(f))
    return _cfg


def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    load_config(refresh=True)


# =========================================================
# TOKENIZATION AND VECTORS
# =========================================================

def tokens(text, bigram=None):
    """Tách âm tiết và tùy chọn thêm bigram tiếng Việt."""
    if bigram is None:
        bigram = load_config()["bigram"]
    words = _WORD.findall(unicodedata.normalize("NFC", text).lower())
    result = [word for word in words if word not in STOP]
    if bigram:
        result.extend(f"{a}_{b}" for a, b in zip(words, words[1:]) if a not in STOP and b not in STOP)
    return result


def _idx(token):
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big")


def doc_vector(text):
    """BM25 vector cho một tài liệu/chunk."""
    cfg = load_config()
    k1, b = cfg["k1"], cfg["b"]
    avgdl = cfg["avgdl"] if cfg["avgdl"] > 0 else 200.0
    counts = Counter(tokens(text))
    norm = k1 * (1 - b + b * sum(counts.values()) / avgdl)
    values = {}
    for token, tf in counts.items():
        index = _idx(token)
        values[index] = values.get(index, 0.0) + tf * (k1 + 1) / (tf + norm)
    return models.SparseVector(indices=list(values), values=list(map(float, values.values())))


def query_vector(text):
    """BM25 query vector với trọng số 1 cho mỗi token duy nhất."""
    indices = sorted({_idx(token) for token in tokens(text)})
    return models.SparseVector(indices=indices, values=[1.0] * len(indices))


# =========================================================
# CLI DIAGNOSTIC
# =========================================================

if __name__ == "__main__":
    cfg = load_config()
    print("Cấu hình:", cfg, "| file:", CONFIG_FILE, "(có)" if os.path.exists(CONFIG_FILE) else "(chưa có, dùng mặc định)")
    sample = "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước."
    doc, query = doc_vector(sample), query_vector("báo trước bao nhiêu ngày")
    print("\nToken mẫu:", tokens(sample)[:14], "...")
    print(f"Vector chunk: {len(doc.indices)} chiều, giá trị lớn nhất {max(doc.values):.2f}")
    print(f"Vector câu hỏi: {len(query.indices)} chiều (token: {tokens('báo trước bao nhiêu ngày')})")
