"""Reranker BAAI/bge-reranker-v2-m3 (cross-encoder): đọc CÙNG LÚC câu hỏi và chunk để chấm điểm liên quan.
Chính xác hơn so sánh vector, nhưng chậm: chi phí tỷ lệ với số chunk cần chấm (xem thời gian được in ra).

Chạy thử:  python reranker.py
"""
import gc
import time

import torch
from FlagEmbedding import FlagReranker

MODEL = "BAAI/bge-reranker-v2-m3"
USE_FP16 = True
_rr = None


def get_reranker():
    global _rr
    if _rr is None:
        _rr = FlagReranker(MODEL, use_fp16=USE_FP16 and torch.cuda.is_available())
    return _rr


def rerank(query, passages, batch_size=4, max_length=1024):
    """Trả về danh sách điểm 0..1 (đã chuẩn hóa sigmoid), cùng thứ tự với passages. Tự giảm batch nếu hết VRAM."""
    if not passages:
        return []
    pairs = [[query, p] for p in passages]
    bs = max(1, batch_size)
    while True:
        try:
            s = get_reranker().compute_score(pairs, batch_size=bs, max_length=max_length, normalize=True)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"  [hết VRAM] giảm batch_size reranker xuống {bs} và thử lại")
    if hasattr(s, "tolist"):
        s = s.tolist()
    if not isinstance(s, (list, tuple)):
        s = [s]
    return [float(x) for x in s]


def release():
    """Giải phóng reranker khỏi GPU (ví dụ trước khi gọi LLM chạy bằng Ollama để nhường VRAM)."""
    global _rr
    _rr = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    q = "người lao động nghỉ việc phải báo trước bao nhiêu ngày"
    ps = ["Điều 35. Quyền đơn phương chấm dứt hợp đồng lao động của người lao động. Người lao động có quyền đơn phương "
          "chấm dứt hợp đồng lao động nhưng phải báo trước cho người sử dụng lao động ít nhất 45 ngày.",
          "Điều 113. Nghỉ hằng năm. Người lao động làm việc đủ 12 tháng thì được nghỉ hằng năm 12 ngày làm việc.",
          "Điều 10. Phân loại đất. Đất nông nghiệp, đất phi nông nghiệp và đất chưa sử dụng."]
    t0 = time.time()
    get_reranker()
    print(f"nạp reranker: {time.time() - t0:.1f}s | thiết bị: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    t0 = time.time()
    sc = rerank(q, ps)
    print(f"chấm {len(ps)} đoạn: {time.time() - t0:.1f}s")
    for s, p in zip(sc, ps):
        print(f"  {s:.4f}  {p[:70]}")
    print("Thứ tự đúng nếu điểm giảm dần từ Điều 35 > Điều 113 > Điều 10:", sc[0] > sc[1] > sc[2])