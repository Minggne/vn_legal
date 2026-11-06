"""Nạp metadata Hugging Face vào Qdrant: parse, dense BGE-M3, sparse BM25, upsert theo batch."""

import argparse
import csv
import os
import sys
import time
import uuid
from datetime import datetime

import pandas as pd
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException

import retrieval.bm25_vi as bm25_vi
from retrieval.embedder import encode
from ingestion.hf_data import iter_content
from ingestion.html_text import html_to_text
from ingestion.law_parser import build_chunks, parse_van_ban
from ingestion.qdrant_db import COLLECTION, DIM, DENSE, SPARSE, ensure_collection, get_client


MAX_HTML = 3_000_000
EMBED_SLICE = 32
UPSERT_BATCH = 64
LOG_FILE = os.path.join("data", "ingest_log.csv")
LOG_FIELDS = ["time", "src_id", "so_hieu", "status", "dieu", "points", "seconds"]


# =========================================================
# CONFIGURATION AND HELPERS
# =========================================================

def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("limit", nargs="?", type=int, default=5, help="Số văn bản mới tối đa")
    parser.add_argument("--tier", help="Chỉ nạp tầng A, B, C...")
    parser.add_argument("--loai", nargs="+", help="Chỉ nạp các loại văn bản đã chọn")
    parser.add_argument("--ids", nargs="*", help="Chỉ nạp các ID trong dataset")
    parser.add_argument("--force", action="store_true", help="Nạp lại cả văn bản đã có")
    parser.add_argument("--max-minutes", type=float, default=0, help="Tự dừng sau số phút; 0 là không giới hạn")
    return parser


def ngay(value):
    try:
        return datetime.strptime(str(value).strip(), "%d/%m/%Y").date().isoformat()
    except (TypeError, ValueError):
        return None


def trang_thai(value):
    value = (value or "").strip().lower()
    if "một phần" in value:
        return "mot_phan"
    if value.startswith("còn hiệu lực"):
        return "con_hieu_luc"
    if value.startswith("hết hiệu lực"):
        return "het_hieu_luc"
    return "khac"


def src_filter(src_id):
    return models.Filter(must=[models.FieldCondition(
        key="src_id", match=models.MatchValue(value=src_id),
    )])


def da_nap_day_du(client, src_id):
    """So số point với n_chunks để nhận diện lần ingest bị ngắt giữa chừng."""
    count = client.count(COLLECTION, count_filter=src_filter(src_id), exact=True).count
    if count == 0:
        return False
    points, _ = client.scroll(
        COLLECTION, scroll_filter=src_filter(src_id), limit=1,
        with_payload=["n_chunks"], with_vectors=False,
    )
    expected = points[0].payload.get("n_chunks") if points else None
    return expected is None or count >= expected


def ghi_log(**row):
    is_new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({"time": datetime.now().isoformat(timespec="seconds"), **row})


# =========================================================
# METADATA AND VECTORS
# =========================================================

def load_metadata(args):
    meta = pd.read_parquet("data/selected_meta.parquet").astype(object)
    meta = meta.where(pd.notna(meta), None)
    rows = {str(row["id"]): row for row in meta.to_dict("records")}
    if args.tier:
        if "tier" not in meta.columns:
            sys.exit("selected_meta.parquet không có cột 'tier'. Hãy chạy select_all.py.")
        rows = {key: row for key, row in rows.items() if row.get("tier") == args.tier}
    if args.loai:
        kinds = {kind.strip().lower() for kind in args.loai}
        rows = {key: row for key, row in rows.items()
                if (row["loai_van_ban"] or "").strip().lower() in kinds}
    if args.ids:
        missing = set(args.ids) - set(rows)
        if missing:
            print("⚠ id không nằm trong phạm vi đã chọn:", missing)
        selected = set(args.ids)
        rows = {key: row for key, row in rows.items() if key in selected}
    if not rows:
        sys.exit("Không có văn bản nào trong phạm vi (kiểm tra --tier / --ids / selected_meta.parquet).")
    return rows


def encode_chunks(chunks):
    texts = [chunk["tien_to"] for chunk in chunks]
    vectors = []
    for start in range(0, len(texts), EMBED_SLICE):
        batch = texts[start:start + EMBED_SLICE]
        encoded = encode(batch)
        if not isinstance(encoded, list) or len(encoded) != len(batch):
            raise RuntimeError(f"encode() trả về {len(encoded)} vector cho {len(batch)} đoạn văn bản")
        vectors.extend(encoded)
        print(f"    embedding {min(start + EMBED_SLICE, len(texts))}/{len(texts)}", end="\r")
    print()
    if len(vectors) != len(chunks):
        raise RuntimeError(f"số vector ({len(vectors)}) khác số chunk ({len(chunks)})")
    return vectors


def make_points(src_id, meta, title, number, kind, chunks, leaf, dense):
    points = []
    for index, (chunk, dense_vector) in enumerate(zip(leaf, dense)):
        vector = {DENSE: dense_vector}
        sparse_vector = bm25_vi.doc_vector(chunk["tien_to"])
        if sparse_vector.indices:
            vector[SPARSE] = sparse_vector
        parent = chunks[chunk["parent"]] if chunk["parent"] is not None else chunk
        payload = {
            "src_id": src_id, "so_hieu": number, "ten": title,
            "tier": meta.get("tier"), "n_chunks": len(leaf),
            "loai": meta["loai_van_ban"], "co_quan": meta["co_quan_ban_hanh"],
            "nganh": meta["nganh"], "linh_vuc": meta["linh_vuc"],
            "ngay_ban_hanh": ngay(meta["ngay_ban_hanh"]),
            "ngay_hieu_luc": ngay(meta["ngay_co_hieu_luc"]),
            "ngay_het_hieu_luc": ngay(meta["ngay_het_hieu_luc"]),
            "tinh_trang": trang_thai(meta["tinh_trang_hieu_luc"]),
            "tinh_trang_goc": (meta["tinh_trang_hieu_luc"] or "").strip(),
            "cap": chunk["cap"], "nhan": chunk["nhan"], "kieu": kind,
            "dieu": chunk["dieu"], "khoan": chunk["khoan"],
            "duong_dan": chunk["duong_dan"], "noi_dung": chunk["noi_dung"],
            "dieu_noi_dung": parent["noi_dung"],
        }
        points.append(models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{src_id}:{index}")),
            vector=vector, payload=payload,
        ))
    return points


# =========================================================
# INGESTION
# =========================================================

def ingest_one(client, src_id, html, meta, args):
    number = (meta["so_ky_hieu"] or "").strip()
    title = (meta["title"] or "").strip()
    if not args.force and da_nap_day_du(client, src_id):
        return "existing", 0, 0
    if len(html or "") > MAX_HTML:
        print(f"  bỏ {src_id} ({number}): HTML quá lớn")
        ghi_log(src_id=src_id, so_hieu=number, status="skip_big", dieu=0, points=0, seconds=0)
        return "skipped", 0, 0

    sections, kind = parse_van_ban(html_to_text(html or ""))
    if not sections:
        print(f"  bỏ {src_id} ({number}): văn bản quá ngắn hoặc rỗng")
        ghi_log(src_id=src_id, so_hieu=number, status="skip_short", dieu=0, points=0, seconds=0)
        return "skipped", 0, 0

    chunks = build_chunks(title, number, sections)
    leaf = [chunk for chunk in chunks if chunk["tim_kiem"]]
    label = {"dieu": "Điều", "muc": "Mục (không chia theo Điều)",
             "doan": "Đoạn (không có cấu trúc)"}[kind]
    print(f"\n>> {number} (id {src_id}): {len(sections)} {label}, {len(leaf)} chunk")
    started = time.time()
    try:
        dense = encode_chunks(leaf)
        points = make_points(src_id, meta, title, number, kind, chunks, leaf, dense)
        # Chỉ xóa dữ liệu cũ sau khi embedding hoàn tất.
        client.delete(COLLECTION, points_selector=models.FilterSelector(filter=src_filter(src_id)))
        for start in range(0, len(points), UPSERT_BATCH):
            client.upsert(COLLECTION, points=points[start:start + UPSERT_BATCH])
    except ResponseHandlingException as e:
        print(f"\n✖ Mất kết nối Qdrant: {str(e)[:200]}\n  Bật lại Qdrant rồi chạy lại lệnh cũ.")
        raise
    except Exception as e:
        print(f"\n  LỖI {src_id} ({number}): {type(e).__name__}: {str(e)[:300]}")
        ghi_log(src_id=src_id, so_hieu=number, status="error", dieu=len(sections), points=0,
                seconds=round(time.time() - started))
        return "failed", 0, len(sections)

    elapsed = max(time.time() - started, 1e-6)
    ghi_log(src_id=src_id, so_hieu=number, status="ok" if kind == "dieu" else f"ok_{kind}",
            dieu=len(sections), points=len(points), seconds=round(elapsed))
    return "ok", len(points), len(sections)


def main():
    args = _parser().parse_args()
    os.makedirs("data", exist_ok=True)
    rows = load_metadata(args)
    client = get_client()
    ensure_collection(client)
    print(f"BM25: {bm25_vi.load_config()}")
    print(f"{len(rows)} văn bản trong phạm vi | lần này nạp mới tối đa {args.limit}"
          + (f" | tối đa {args.max_minutes:g} phút" if args.max_minutes else ""))

    sample = encode(["kiểm tra", "kiểm tra hai"])
    if not (isinstance(sample, list) and len(sample) == 2 and len(sample[0]) == DIM):
        sys.exit(f"✖ embedder.encode() không trả về list 2 vector, mỗi vector {DIM} số. Chạy: python check_setup.py")

    started = time.time()
    new = existing = skipped = failed = total_points = 0
    todo = set(rows)
    try:
        for src_id, html in iter_content(todo):
            if args.max_minutes and time.time() - started > args.max_minutes * 60:
                print(f"\nĐã chạy đủ {args.max_minutes:g} phút, dừng. Chạy lại lệnh cũ để nạp tiếp.")
                break
            todo.discard(src_id)
            try:
                status, points, _ = ingest_one(client, src_id, html, rows[src_id], args)
            except ResponseHandlingException:
                sys.exit(1)
            if status == "existing":
                existing += 1
                continue
            if status == "skipped":
                skipped += 1
                continue
            if status == "failed":
                failed += 1
                continue
            new += 1
            total_points += points
            rate = total_points / max(time.time() - started, 1e-6)
            print(f"   xong: {points} points | tổng lần này: {new} văn bản, {total_points} points, TB {rate:.2f} point/giây")
            if new >= args.limit:
                break
    except KeyboardInterrupt:
        print("\nĐã dừng theo yêu cầu (Ctrl+C). Chạy lại sẽ tiếp tục từ văn bản chưa nạp.")

    elapsed = time.time() - started
    print(f"\n=== Xong: nạp mới {new}, đã có sẵn {existing}, bỏ {skipped}, lỗi {failed} ===")
    print(f"Tổng {total_points} points trong {elapsed / 60:.1f} phút | log chi tiết: {LOG_FILE}")


if __name__ == "__main__":
    main()
