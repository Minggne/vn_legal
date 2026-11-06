"""Tìm kiếm: dense (BGE-M3) + BM25 -> hybrid (RRF của Qdrant) -> rerank (bge-reranker-v2-m3, tùy chọn).
In điểm và hạng ở từng bước: TRƯỚC hybrid, SAU hybrid, SAU rerank.

Cách dùng:
    python search.py "câu hỏi"
    python search.py --no-rerank "câu hỏi"           # bỏ rerank (nhanh)
    python search.py --rerank-pool 10 "câu hỏi"      # chỉ rerank 10 chunk đầu của hybrid (nhanh hơn)
    python search.py --rrf-k 60 "câu hỏi"            # hằng số RRF khác (mặc định của Qdrant là k=2)
    python search.py --show 15 "câu hỏi"

Hàm dùng lại trong code: retrieve(), gop_dieu(), in_cac_buoc(), hybrid_search() (giữ lại cho tương thích).
"""
import argparse
import sys
import time
import unicodedata

from qdrant_client import models

import retrieval.bm25_vi as bm25_vi
from retrieval.embedder import encode
from ingestion.qdrant_db import get_client, COLLECTION, DENSE, SPARSE

client = get_client()
# Chỉ tìm trong văn bản còn hiệu lực hoặc còn hiệu lực một phần
VALID = models.Filter(must=[models.FieldCondition(
    key="tinh_trang", match=models.MatchAny(any=["con_hieu_luc", "mot_phan"]))])


def _enc(q):
    """Trả về (vector dense, vector sparse BM25) của câu hỏi."""
    q = unicodedata.normalize("NFC", q)
    return encode([q])[0], bm25_vi.query_vector(q)


def _dense(dv, limit):
    return client.query_points(COLLECTION, query=dv, using=DENSE, query_filter=VALID,
                               limit=limit, with_payload=True).points


def _sparse(sv, limit):
    if not sv.indices:
        return []
    return client.query_points(COLLECTION, query=sv, using=SPARSE, query_filter=VALID,
                               limit=limit, with_payload=True).points


def _fuse(dv, sv, pool, limit, rrf_k=None):
    """Hợp nhất dense + BM25 bằng RRF ngay trong Qdrant. Điểm mỗi nhánh = 1/(k + hạng - 1); k mặc định của Qdrant = 2."""
    prefetch = [models.Prefetch(query=dv, using=DENSE, limit=pool, filter=VALID)]
    if sv.indices:
        prefetch.append(models.Prefetch(query=sv, using=SPARSE, limit=pool, filter=VALID))
    fusion = (models.FusionQuery(fusion=models.Fusion.RRF) if rrf_k is None
              else models.RrfQuery(rrf=models.Rrf(k=rrf_k)))
    try:
        res = client.query_points(COLLECTION, prefetch=prefetch, query=fusion,
                                  query_filter=VALID, limit=limit, with_payload=True)
    except Exception as e:
        if rrf_k is None:
            raise
        raise RuntimeError(f"Qdrant không nhận tham số RRF k={rrf_k} (server có thể quá cũ, cần bản mới hơn). "
                           f"Bỏ --rrf-k để dùng mặc định. Chi tiết: {type(e).__name__}: {str(e)[:150]}") from e
    return res.points


def hybrid_search(q, top=6, pool=30, explain=True, rrf_k=None):
    """(Giữ lại cho tương thích.) Hybrid không rerank. explain=True: kèm hạng của từng nhánh (d_rank, s_rank)."""
    dv, sv = _enc(q)
    pts = _fuse(dv, sv, pool, top, rrf_k)
    d_ids = [p.id for p in _dense(dv, pool)] if explain else []
    s_ids = [p.id for p in _sparse(sv, pool)] if explain else []
    return [dict(id=p.id, score=p.score, payload=p.payload,
                 d_rank=d_ids.index(p.id) + 1 if p.id in d_ids else None,
                 s_rank=s_ids.index(p.id) + 1 if p.id in s_ids else None)
            for p in pts]


def _moi(p):
    return dict(id=p.id, payload=p.payload, dense_rank=None, dense_score=None, bm25_rank=None, bm25_score=None,
                hybrid_rank=None, hybrid_score=None, rerank_rank=None, rerank_score=None)


def retrieve(q, pool=30, hybrid_limit=20, rerank_pool=20, use_rerank=True, rrf_k=None,
             rerank_batch=4, rerank_maxlen=1024):
    """Chạy đủ các bước và trả về điểm + hạng của từng chunk ở MỖI bước.

    Kết quả (dict):
      dense, bm25   : danh sách chunk của từng nhánh (top `pool`), TRƯỚC hybrid
      hybrid        : danh sách chunk SAU hybrid (RRF), `hybrid_limit` chunk đầu
      reranked      : `rerank_pool` chunk đầu của hybrid, SAU rerank (rỗng nếu không rerank)
      final         : thứ tự cuối cùng (rerank nếu có, sau đó phần còn lại theo hybrid)
    Mỗi chunk là dict có: payload, dense_rank/score, bm25_rank/score, hybrid_rank/score, rerank_rank/score (None nếu không có).
    """
    import retrieval.embedder as embedder
    t = time.perf_counter()
    embedder.get_model()                                   # nạp model trước để thời gian bên dưới là thời gian chạy thật
    rr = None
    if use_rerank:
        import retrieval.reranker as rr
        rr.get_reranker()
    t_load = time.perf_counter() - t

    if use_rerank:
        hybrid_limit = max(hybrid_limit, rerank_pool)
    t = time.perf_counter()
    dv, sv = _enc(q)
    d_pts, b_pts = _dense(dv, pool), _sparse(sv, pool)
    h_pts = _fuse(dv, sv, pool, hybrid_limit, rrf_k)
    t_retrieve = time.perf_counter() - t

    cands = {}
    for r, p in enumerate(d_pts, 1):
        c = cands.setdefault(p.id, _moi(p))
        c["dense_rank"], c["dense_score"] = r, p.score
    for r, p in enumerate(b_pts, 1):
        c = cands.setdefault(p.id, _moi(p))
        c["bm25_rank"], c["bm25_score"] = r, p.score
    for r, p in enumerate(h_pts, 1):
        c = cands.setdefault(p.id, _moi(p))
        c["hybrid_rank"], c["hybrid_score"] = r, p.score
    dense = [cands[p.id] for p in d_pts]
    bm25 = [cands[p.id] for p in b_pts]
    hybrid = [cands[p.id] for p in h_pts]

    reranked, t_rerank = [], 0.0
    if use_rerank and hybrid:
        pool_c = hybrid[:rerank_pool]
        t = time.perf_counter()
        scores = rr.rerank(q, [c["payload"]["duong_dan"] + "\n" + c["payload"]["noi_dung"] for c in pool_c],
                           batch_size=rerank_batch, max_length=rerank_maxlen)
        t_rerank = time.perf_counter() - t
        for c, s in zip(pool_c, scores):
            c["rerank_score"] = s
        reranked = sorted(pool_c, key=lambda c: -c["rerank_score"])
        for r, c in enumerate(reranked, 1):
            c["rerank_rank"] = r
    final = reranked + [c for c in hybrid if c["rerank_rank"] is None] if reranked else hybrid
    return dict(q=q, dense=dense, bm25=bm25, hybrid=hybrid, reranked=reranked, final=final, rrf_k=rrf_k,
                t_load=t_load, t_retrieve=t_retrieve, t_rerank=t_rerank)


def gop_dieu(cands):
    """Gộp các chunk cùng một Điều (giữ thứ tự; chunk đầu tiên là chunk tốt nhất của Điều đó)."""
    seen, out = set(), []
    for c in cands:
        key = (c["payload"]["src_id"], c["payload"]["dieu"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def nhan_ngan(c, w=70):
    """Nhãn ngắn của chunk: '[số hiệu] Điều 35. Tên Điều · Khoản 2' (cắt phần tên Điều nếu quá dài)."""
    p = c["payload"]
    nhan = p.get("nhan") or "Điều"
    muc = next((x for x in p["duong_dan"].split(" > ") if x.startswith(f"{nhan} {p['dieu']}")), f"{nhan} {p['dieu']}")
    kh = f" · Khoản {p['khoan']}" if p.get("khoan") else ""
    dau = f"[{p['so_hieu']}] "
    ten = muc if len(dau) + len(muc) + len(kh) <= w else muc[:max(w - len(dau) - len(kh) - 1, 10)] + "…"
    return f"{dau}{ten}{kh}"


def _h(x):
    return "—" if x is None else str(x)


def in_cac_buoc(res, show=10, top_branch=5):
    print(f"Câu hỏi: {res['q']}")
    t = f"Thời gian: truy xuất {res['t_retrieve']:.2f}s"
    if res["reranked"]:
        t += f" | rerank {res['t_rerank']:.1f}s ({len(res['reranked'])} chunk)"
    if res["t_load"] > 0.5:
        t += f" | nạp model {res['t_load']:.1f}s (chỉ lần đầu)"
    print(t)

    print(f"\n[1] TRƯỚC HYBRID: kết quả riêng của từng nhánh (top {top_branch}, tính theo chunk)")
    print("  Dense (độ tương đồng cosine):")
    for r, c in enumerate(res["dense"][:top_branch], 1):
        print(f"    #{r:<3}{c['dense_score']:>9.4f}   {nhan_ngan(c)}")
    print("  BM25 (điểm BM25, đã nhân IDF; không có giới hạn trên):")
    for r, c in enumerate(res["bm25"][:top_branch], 1):
        print(f"    #{r:<3}{c['bm25_score']:>9.3f}   {nhan_ngan(c)}")

    k = res["rrf_k"]
    print(f"\n[2] SAU HYBRID (RRF, k={k if k else '2 = mặc định của Qdrant'}): top {show}")
    print(f"  {'hạng':>4} {'điểm RRF':>9} {'dense#':>7} {'bm25#':>6}   chunk")
    for c in res["hybrid"][:show]:
        print(f"  {c['hybrid_rank']:>4} {c['hybrid_score']:>9.4f} {_h(c['dense_rank']):>7} {_h(c['bm25_rank']):>6}   {nhan_ngan(c)}")

    if res["reranked"]:
        print(f"\n[3] SAU RERANK (điểm 0..1; chấm lại {len(res['reranked'])} chunk đầu của hybrid): top {show}")
        print(f"  {'hạng':>4} {'điểm':>7} {'hạng hybrid':>12} {'đổi':>5}   chunk")
        for c in res["reranked"][:show]:
            d = c["hybrid_rank"] - c["rerank_rank"]
            dd = "=" if d == 0 else (f"▲{d}" if d > 0 else f"▼{-d}")
            print(f"  {c['rerank_rank']:>4} {c['rerank_score']:>7.4f} {c['hybrid_rank']:>12} {dd:>5}   {nhan_ngan(c)}")
    else:
        print("\n[3] SAU RERANK: (không rerank)")


def in_dieu_cuoi(res, n=5):
    """In các Điều cuối cùng (gộp theo Điều, lấy chunk điểm cao nhất của mỗi Điều)."""
    print(f"\n[4] ĐIỀU CUỐI CÙNG (gộp theo Điều, top {n})")
    print(f"  {'#':>2} {'điểm':>8} {'loại điểm':>10}   Điều")
    for i, c in enumerate(gop_dieu(res["final"])[:n], 1):
        if c["rerank_score"] is not None:
            diem, loai = c["rerank_score"], "rerank"
        else:
            diem, loai = c["hybrid_score"], "hybrid"
        print(f"  {i:>2} {diem:>8.4f} {loai:>10}   {nhan_ngan(c)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cau_hoi", nargs="+")
    ap.add_argument("--no-rerank", action="store_true", help="bỏ rerank")
    ap.add_argument("--rerank-pool", type=int, default=20, help="số chunk đầu của hybrid đem đi rerank")
    ap.add_argument("--rerank-batch", type=int, default=4)
    ap.add_argument("--rerank-maxlen", type=int, default=1024)
    ap.add_argument("--pool", type=int, default=30, help="số chunk lấy từ mỗi nhánh trước khi hợp nhất")
    ap.add_argument("--show", type=int, default=10, help="số dòng in cho bước hybrid và rerank")
    ap.add_argument("--rrf-k", type=int, default=None)
    ap.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)      # tương thích bản cũ (luôn in chi tiết)
    a = ap.parse_args()
    res = retrieve(" ".join(a.cau_hoi), pool=a.pool, rerank_pool=a.rerank_pool, use_rerank=not a.no_rerank,
                   rrf_k=a.rrf_k, rerank_batch=a.rerank_batch, rerank_maxlen=a.rerank_maxlen)
    in_cac_buoc(res, show=a.show)
    in_dieu_cuoi(res)