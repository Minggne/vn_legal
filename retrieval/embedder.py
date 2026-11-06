"""Embedding dense bằng BGE-M3 (1024 chiều). Phần từ khóa (sparse) do bm25_vi.py đảm nhiệm.

Chạy thử:  python embedder.py
"""
import gc
import time

import torch
from FlagEmbedding import BGEM3FlagModel

USE_FP16 = True        # đã kiểm tra trên GTX 1650: không bị NaN. Nếu thấy NaN thì đổi False.
BATCH_SIZE = 8         # đã đo: nhanh nhất ở trường hợp chunk dài
MAX_LENGTH = 1024

_model = None


def get_model():
    global _model
    if _model is None:
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=USE_FP16 and torch.cuda.is_available())
    return _model


def release():
    """Giải phóng model khỏi GPU (ví dụ trước khi gọi LLM chạy bằng Ollama để nhường VRAM)."""
    global _model
    _model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode(texts, batch_size=None, max_length=None):
    """Trả về danh sách vector dense (đã chuẩn hóa), mỗi vector là list 1024 số."""
    bs = batch_size or BATCH_SIZE
    ml = max_length or MAX_LENGTH
    while True:
        try:
            out = get_model().encode(texts, batch_size=bs, max_length=ml,
                                     return_dense=True, return_sparse=False,
                                     return_colbert_vecs=False)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"  [hết VRAM] giảm batch_size xuống {bs} và thử lại")
    return [v.tolist() for v in out["dense_vecs"]]


if __name__ == "__main__":
    import numpy as np

    gpu = torch.cuda.is_available()
    print("device:", torch.cuda.get_device_name(0) if gpu else "CPU", "| fp16:", USE_FP16 and gpu)
    t0 = time.time()
    get_model()
    print(f"nạp model: {time.time() - t0:.1f}s")
    d = np.array(encode(["Người lao động có quyền đơn phương chấm dứt hợp đồng lao động",
                         "Quyền chấm dứt hợp đồng của người lao động",
                         "Giá vàng hôm nay"]))
    print("dim =", d.shape[1], "| có NaN?", bool(np.isnan(d).any()))
    print("cos(câu1,câu2) =", round(float(d[0] @ d[1]), 3), "(phải cao) | cos(câu1,câu3) =",
          round(float(d[0] @ d[2]), 3), "(phải thấp hơn rõ)")