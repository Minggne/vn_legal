import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DS = "th1nhng0/vietnamese-legal-documents"


def content_path():
    """Tải (một lần, có cache) file nội dung và trả về đường dẫn local."""
    return hf_hub_download(DS, "data/content.parquet", repo_type="dataset")


def iter_content(ids=None, batch_rows=64):
    """Sinh (id, content_html). Nếu truyền ids thì chỉ lấy các văn bản đó."""
    pf = pq.ParquetFile(content_path())
    wanted = pa.array(sorted(ids), type=pa.string()) if ids is not None else None
    for batch in pf.iter_batches(batch_size=batch_rows, columns=["id", "content_html"]):
        if wanted is not None:
            mask = pc.is_in(pc.cast(batch.column("id"), pa.string()), value_set=wanted)
            batch = batch.filter(mask)
            if batch.num_rows == 0:
                continue
        for sid, html in zip(batch.column("id").to_pylist(), batch.column("content_html").to_pylist()):
            yield str(sid), html