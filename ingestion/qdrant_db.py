"""Kết nối Qdrant và cấu hình collection.

Collection 'luat_vn':
  - vector 'dense' : BGE-M3, 1024 chiều, cosine
  - vector 'bm25'  : sparse, bật modifier IDF (Qdrant tự nhân IDF khi truy vấn; cần Qdrant >= 1.10)

Cách dùng:
    python qdrant_db.py               # kiểm tra kết nối, tạo collection nếu chưa có
    python qdrant_db.py --recreate    # XÓA collection luat_vn rồi tạo lại (mất toàn bộ dữ liệu trong đó)
"""
import json
import os
import sys
import urllib.request
from importlib.metadata import version

from qdrant_client import QdrantClient, models

URL = os.getenv("QDRANT_URL", "http://localhost:6333")
API_KEY = os.getenv("QDRANT_API_KEY") or None
PATH = os.getenv("QDRANT_PATH")          # chỉ để thử nghiệm (local mode). Mặc định dùng server.
COLLECTION = "luat_vn"
DIM = 1024
DENSE = "dense"
SPARSE = "bm25"
PAYLOAD_INDEXES = ("src_id", "so_hieu", "tinh_trang", "loai", "dieu", "tier")


def get_client():
    if PATH:
        return QdrantClient(path=PATH)
    return QdrantClient(url=URL, api_key=API_KEY, timeout=60)


def server_info():
    with urllib.request.urlopen(URL, timeout=5) as r:
        return json.load(r)


def ensure_collection(client, recreate=False):
    if recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    if client.collection_exists(COLLECTION):
        return False
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={DENSE: models.VectorParams(size=DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    # Tạo payload index NGAY SAU KHI tạo collection, trước khi nạp dữ liệu (khuyến nghị của Qdrant)
    for field in PAYLOAD_INDEXES:
        client.create_payload_index(COLLECTION, field_name=field,
                                    field_schema=models.PayloadSchemaType.KEYWORD)
    print(f"Đã tạo collection '{COLLECTION}'")
    return True


if __name__ == "__main__":
    if PATH:
        print(f"Chế độ local (thử nghiệm): {PATH}")
    else:
        try:
            info = server_info()
        except Exception as e:
            sys.exit(f"Không kết nối được Qdrant tại {URL}\n  Lỗi: {type(e).__name__}: {e}\n"
                     "  → Server đã chạy chưa? (cd thư mục qdrant, rồi .\\qdrant.exe)")
        sv = str(info.get("version", "0.0"))
        print(f"Server : {info.get('title')} v{sv}  ({URL})")
        try:
            major, minor = (int(x) for x in sv.split(".")[:2])
            if (major, minor) < (1, 10):
                print("⚠ Server cũ hơn 1.10: không có modifier IDF cho vector sparse. Hãy tải bản mới.")
        except ValueError:
            pass
    print(f"Client : qdrant-client {version('qdrant-client')}")
    c = get_client()
    ensure_collection(c, recreate="--recreate" in sys.argv)
    i = c.get_collection(COLLECTION)
    print("Collections:", [x.name for x in c.get_collections().collections])
    print(f"'{COLLECTION}': points={i.points_count} | status={i.status}")
    print("dense :", i.config.params.vectors)
    print("sparse:", i.config.params.sparse_vectors)