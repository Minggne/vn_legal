"""Xem thống kê collection hoặc các point của một Điều trong Qdrant."""

import sys
from collections import Counter

from qdrant_client import models

from ingestion.qdrant_db import COLLECTION, SPARSE, get_client


# =========================================================
# DETAIL VIEW
# =========================================================

def show_article(client, number, article):
    flt = models.Filter(must=[
        models.FieldCondition(key="so_hieu", match=models.MatchValue(value=number)),
        models.FieldCondition(key="dieu", match=models.MatchValue(value=article)),
    ])
    points, _ = client.scroll(COLLECTION, scroll_filter=flt, limit=20, with_payload=True)
    print(f"\n{len(points)} point cho {number} Điều {article}:")
    if not points:
        print("  (không có). Kiểm tra lại số hiệu hoặc parser.")
    for point in points:
        payload = point.payload
        print(f"\n- {payload['duong_dan']}")
        print("  ", payload["noi_dung"][:300].replace("\n", " "))
        print("   tình trạng:", payload["tinh_trang"], "|", payload["tinh_trang_goc"])


# =========================================================
# COLLECTION STATISTICS
# =========================================================

def show_stats(client):
    status, documents, levels, kinds, tiers = (Counter() for _ in range(5))
    missing_sparse = max_length = 0
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION, limit=500, offset=offset,
            with_payload=["so_hieu", "tinh_trang", "cap", "noi_dung", "loai", "tier", "src_id"],
            with_vectors=[SPARSE],
        )
        for point in points:
            payload = point.payload
            status[payload["tinh_trang"]] += 1
            documents[payload["src_id"]] += 1
            levels[payload["cap"]] += 1
            kinds[payload.get("loai")] += 1
            tiers[payload.get("tier")] += 1
            max_length = max(max_length, len(payload.get("noi_dung") or ""))
            missing_sparse += not point.vector or SPARSE not in point.vector
        if offset is None:
            break

    print("Theo tình trạng :", dict(status))
    print("Theo loại VB    :", dict(kinds))
    print("Theo tầng (tier):", dict(tiers))
    print("Theo cấp chunk  :", dict(levels))
    print("Số văn bản      :", len(documents))
    print("Văn bản nhiều point nhất (src_id):", documents.most_common(5))
    print("Chunk dài nhất  :", max_length, "ký tự")
    print("Point thiếu vector bm25:", missing_sparse)


# =========================================================
# CLI
# =========================================================

def main():
    client = get_client()
    if not client.collection_exists(COLLECTION):
        sys.exit(f"Chưa có collection '{COLLECTION}'. Chạy: python qdrant_db.py")
    total = client.count(COLLECTION, exact=True).count
    print(f"Collection '{COLLECTION}' | tổng points: {total}")
    if total == 0:
        sys.exit("Collection đang trống. Hãy chạy ingest_hf.py trước.")
    if len(sys.argv) >= 3:
        show_article(client, sys.argv[1], sys.argv[2])
    else:
        show_stats(client)


if __name__ == "__main__":
    main()
