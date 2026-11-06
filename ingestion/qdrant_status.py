from ingestion.qdrant_db import get_client, URL

c = get_client()
cols = [x.name for x in c.get_collections().collections]
print("Server:", URL, "| collections:", cols)
for name in cols:
    i = c.get_collection(name)
    print(f"\n[{name}] status={i.status} | optimizer={i.optimizer_status}")
    print(f"  points={i.points_count} | indexed_vectors={i.indexed_vectors_count} | segments={i.segments_count}")
    print(f"  dense : {i.config.params.vectors}")
    print(f"  sparse: {i.config.params.sparse_vectors}")
    print(f"  payload index: {list((i.payload_schema or {}).keys())}")