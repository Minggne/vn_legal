"""Đo chất lượng truy xuất ở MỨC ĐIỀU qua 4 bước: dense, BM25, hybrid (RRF), hybrid + rerank.
Với mỗi câu hỏi in HẠNG và ĐIỂM của Điều đúng ở từng bước; cuối cùng in Recall@K, MRR cho từng bước.
Hạng tính theo Điều (các chunk cùng một Điều được gộp), vì đó là đơn vị đưa cho LLM.

Cách dùng:
    python eval_retrieval.py                    # có rerank (chậm: mỗi câu hỏi mất thời gian rerank)
    python eval_retrieval.py --no-rerank        # nhanh, bỏ qua bước rerank
    python eval_retrieval.py --rerank-pool 10   # rerank ít chunk hơn (nhanh hơn)
    python eval_retrieval.py --k 10 --rrf-k 60

eval_set.json: danh sách các mục {"q": câu hỏi, "so_hieu": số hiệu (nên có), "dieu": "35" hoặc ["35","36"]}.
"""
import argparse
import json
import sys

from retrieval.search import retrieve

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--file", default="data/eval_set.json")
ap.add_argument("--rrf-k", type=int, default=None, help="hằng số RRF (mặc định của Qdrant là k=2)")
ap.add_argument("--no-rerank", action="store_true")
ap.add_argument("--rerank-pool", type=int, default=15)
ap.add_argument("--rerank-batch", type=int, default=4)
ap.add_argument("--rerank-maxlen", type=int, default=1024)
ap.add_argument("--pool", type=int, default=15)
args = ap.parse_args()
K = args.k

try:
    data = json.load(open(args.file, encoding="utf-8"))
except FileNotFoundError:
    sys.exit(f"Không thấy {args.file}. Hãy tạo file này (xem hướng dẫn).")


def dung(pl, ex):
    ds = ex["dieu"] if isinstance(ex["dieu"], list) else [ex["dieu"]]
    if pl.get("dieu") not in [str(x) for x in ds]:
        return False
    return not ex.get("so_hieu") or pl.get("so_hieu") == ex["so_hieu"]


def hang_diem(cands, ex, khoa_diem):
    """(hạng theo Điều, điểm) của Điều đúng đầu tiên trong danh sách; (None, None) nếu không có."""
    seen, r = set(), 0
    for c in cands:
        key = (c["payload"]["src_id"], c["payload"]["dieu"])
        if key in seen:
            continue
        seen.add(key)
        r += 1
        if dung(c["payload"], ex):
            return r, c[khoa_diem]
    return None, None


CAC_BUOC = [("dense", "dense", "dense_score"), ("bm25", "bm25", "bm25_score"), ("hybrid", "hybrid", "hybrid_score")]
if not args.no_rerank:
    CAC_BUOC.append(("rerank", "reranked", "rerank_score"))
ten = [b[0] for b in CAC_BUOC]
found = {t: 0 for t in ten}
rr = {t: 0.0 for t in ten}
sum_hang = {t: 0 for t in ten}
n_hang = {t: 0 for t in ten}
so_sanh = {"tốt hơn": 0, "bằng": 0, "kém hơn": 0}
t_retr, t_rer = 0.0, 0.0


print(f"Hạng (theo Điều) và điểm của Điều đúng ở từng bước. K = {K}. '—' = không có trong danh sách của bước đó.")
print("điểm: dense = cosine | bm25 = điểm BM25 | hybrid = điểm RRF | rerank = 0..1\n")
for i, ex in enumerate(data, 1):
    res = retrieve(ex["q"], pool=args.pool, rerank_pool=args.rerank_pool, use_rerank=not args.no_rerank,
                   rrf_k=args.rrf_k, rerank_batch=args.rerank_batch, rerank_maxlen=args.rerank_maxlen)
    t_retr += res["t_retrieve"]
    t_rer += res["t_rerank"]
    kq = {}
    for t, khoa_ds, khoa_diem in CAC_BUOC:
        kq[t] = hang_diem(res[khoa_ds], ex, khoa_diem)
        h = kq[t][0]
        if h is not None:
            sum_hang[t] += h
            n_hang[t] += 1
        if h is not None and h <= K:
            found[t] += 1
            rr[t] += 1 / h
    phan = []
    for t in ten:
        h, d = kq[t]
        phan.append(f"{t} #{'—' if h is None else h}" + ("" if d is None else f" ({d:.3g})"))
    cot = " | ".join(phan)
    h_hy = kq["hybrid"][0]
    if not args.no_rerank:
        h_rr = kq["rerank"][0]
        a = h_hy if h_hy is not None else 10 ** 6
        b = h_rr if h_rr is not None else 10 ** 6
        so_sanh["tốt hơn" if b < a else "kém hơn" if b > a else "bằng"] += 1
    dau = "✔" if (kq[ten[-1]][0] is not None and kq[ten[-1]][0] <= K) else "✘"
    print(f"{dau} {i:>2}. {ex['q'][:70]}")
    print(f"      đáp án Điều {ex['dieu']}: {cot}")

n = len(data)
print(f"\n=== {n} câu hỏi | mức Điều | K = {K} | RRF k = {args.rrf_k if args.rrf_k else '2 (mặc định Qdrant)'} ===")
print(f"{'bước':<10}{'Recall@' + str(K):>11}{'MRR':>8}{'hạng TB*':>10}")
for t in ten:
    tb = f"{sum_hang[t] / n_hang[t]:.2f}" if n_hang[t] else "—"
    print(f"{t:<10}{found[t] / n:>11.2f}{rr[t] / n:>8.2f}{tb:>10}")
print("* hạng trung bình của Điều đúng, chỉ tính các câu mà bước đó tìm thấy Điều đúng.")
if not args.no_rerank:
    print(f"\nRerank so với hybrid (theo hạng Điều đúng): tốt hơn {so_sanh['tốt hơn']} câu, "
          f"bằng {so_sanh['bằng']} câu, kém hơn {so_sanh['kém hơn']} câu")
    print(f"Thời gian trung bình mỗi câu: truy xuất {t_retr / n:.2f}s | rerank {t_rer / n:.1f}s "
          f"({args.rerank_pool} chunk)")