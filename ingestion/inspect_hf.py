import pandas as pd
from datasets import load_dataset

DS = "th1nhng0/vietnamese-legal-documents"
pd.set_option("display.max_colwidth", 80, "display.width", 200)

# 1. Metadata: thông tin từng văn bản
meta = load_dataset(DS, "metadata", split="data").to_pandas()
print("METADATA:", meta.shape)
print("Cột:", list(meta.columns), "\r\n")
print(meta.head(2).T, "\r\n")
for col in ["loai_van_ban", "tinh_trang_hieu_luc"]:
    print(f"--- {col} ---")
    print(meta[col].value_counts(dropna=False).head(15), "\r\n")
print("--- linh_vuc (mẫu) ---")
print(meta["linh_vuc"].dropna().head(10).tolist(), "\r\n")

# 2. Content: stream vài dòng đầu, không tải hết ~4GB
stream = load_dataset(DS, "content", split="data", streaming=True)
for i, row in enumerate(stream):
    h = row["content_html"]
    print(f"CONTENT: cột={list(row.keys())} | id={row['id']!r} | độ dài HTML={len(h):,}")
    print(h[:300].replace("\r\n", " "), "\r\n")
    if i >= 1:
        break