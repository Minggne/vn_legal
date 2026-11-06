"""Chọn tập văn bản để nạp: pháp luật Việt Nam theo từng tầng ưu tiên.

Mặc định: tầng A (Bộ luật, Luật, Pháp lệnh, Nghị định) và tầng B (Thông tư, Thông tư liên tịch),
chỉ lấy văn bản CÒN HIỆU LỰC hoặc HẾT HIỆU LỰC MỘT PHẦN, mọi phạm vi (xem --no-local), bỏ bản dịch và
văn bản sửa đổi/bổ sung. Kết quả: data/selected_meta.parquet (có cột 'tier').

Cách dùng:
    python select_all.py                       # tầng A + B
    python select_all.py --tiers A             # chỉ tầng A
    python select_all.py --tiers A B C         # thêm tầng C (Nghị quyết, Quyết định, Chỉ thị, VB hợp nhất)
    python select_all.py --no-local            # bỏ văn bản phạm vi 'Địa phương' (giữ Trung ương, Toàn quốc, trống)
    python select_all.py --tu-nam 1990         # chỉ lấy văn bản ban hành từ năm 1990 trở đi
    python select_all.py --pham-vi "Trung ương"   # chỉ đúng phạm vi này (loại cả 'Toàn quốc')
    python select_all.py --keep-amendments     # giữ cả văn bản sửa đổi/bổ sung
"""
import argparse
import os

import pandas as pd
from datasets import load_dataset

DS = "th1nhng0/vietnamese-legal-documents"

# ================= CẤU HÌNH =================
TIERS = {
    "A": ["Bộ luật", "Luật", "Pháp lệnh", "Nghị định"],
    "B": ["Thông tư", "Thông tư liên tịch"],
    "C": ["Nghị quyết", "Quyết định", "Chỉ thị", "Văn bản hợp nhất"],   # nhiều văn bản cá biệt, cân nhắc
}
TRANG_THAI = ["Còn hiệu lực", "Hết hiệu lực một phần"]
KIEM_TRA = ["45/2019/QH14"]       # in cảnh báo nếu các số hiệu này không nằm trong tập chọn
# ============================================

ap = argparse.ArgumentParser()
ap.add_argument("--tiers", nargs="+", default=["A", "B"], choices=sorted(TIERS))
ap.add_argument("--pham-vi", default="all", help="'Trung ương', 'Địa phương', 'Toàn quốc' hoặc 'all' (mặc định)")
ap.add_argument("--no-local", action="store_true", help="bỏ văn bản có phạm vi 'Địa phương'")
ap.add_argument("--tu-nam", type=int, default=0,
                help="chỉ lấy văn bản ban hành từ năm này trở đi (0 = không lọc; văn bản không rõ ngày được giữ lại)")
ap.add_argument("--keep-amendments", action="store_true")
args = ap.parse_args()

meta = load_dataset(DS, "metadata", split="data").to_pandas()
meta["id"] = meta["id"].astype(str)
S = lambda c: meta[c].fillna("").astype(str).str.strip()

loai2tier = {l.lower(): t for t in args.tiers for l in TIERS[t]}
m = pd.Series(True, index=meta.index)
print("=== Phễu lọc ===")
print(f"  {len(meta):>7}  ban đầu")
buoc = [
    (f"Trạng thái thuộc {TRANG_THAI}", S("tinh_trang_hieu_luc").isin(TRANG_THAI)),
    (f"Loại thuộc tầng {args.tiers}", S("loai_van_ban").str.lower().isin(loai2tier)),
]
if args.pham_vi != "all":
    buoc.append((f"Phạm vi = {args.pham_vi}", S("pham_vi") == args.pham_vi))
if args.no_local:
    buoc.append(("Không phải phạm vi 'Địa phương'", S("pham_vi") != "Địa phương"))
if args.tu_nam:
    nam = pd.to_datetime(S("ngay_ban_hanh"), format="%d/%m/%Y", errors="coerce").dt.year
    buoc.append((f"Ban hành từ năm {args.tu_nam} (giữ văn bản không rõ ngày)", nam.isna() | (nam >= args.tu_nam)))
if not args.keep_amendments:
    buoc.append(("Không phải văn bản sửa đổi/bổ sung/bãi bỏ/đính chính",
                 ~S("title").str.contains("sửa đổi|bổ sung|bãi bỏ|đính chính", case=False, regex=True)))
for ten, cond in buoc:
    m &= cond
    print(f"  {int(m.sum()):>7}  sau điều kiện: {ten}")

sel = meta[m].copy()
sel["tier"] = S("loai_van_ban")[m].str.lower().map(loai2tier)
sel["_d"] = pd.to_datetime(sel["ngay_ban_hanh"], format="%d/%m/%Y", errors="coerce")
sel = sel.sort_values(["tier", "_d"], ascending=[True, False]).drop(columns="_d")

print(f"\n=== ĐÃ CHỌN {len(sel)} văn bản ===")
print(sel.groupby(["tier", "loai_van_ban"]).size().to_string())
print("\nTheo trạng thái:")
print(sel["tinh_trang_hieu_luc"].value_counts().to_string())
trung = sel["so_ky_hieu"].fillna("").str.strip()
n_trung = int(trung[trung != ""].duplicated(keep=False).sum())
print(f"\nVăn bản có số hiệu trùng với văn bản khác trong tập chọn: {n_trung} "
      "(có thể là bản gốc và bản hợp nhất, hoặc hai bản ghi của cùng văn bản)")
for so in KIEM_TRA:
    co = sel[trung == so]
    print(f"Kiểm tra {so}: " + (f"CÓ ({', '.join(co['id'])})" if len(co) else "⚠ KHÔNG có trong tập chọn"))

os.makedirs("data", exist_ok=True)
sel.to_parquet("data/selected_meta.parquet", index=False)
print("\nĐã lưu data/selected_meta.parquet")
print("Bước tiếp theo: python estimate_scope.py  (ước lượng số chunk, thời gian nạp, và hiệu chỉnh BM25)")