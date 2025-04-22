import os

DATA_DIR = os.environ.get("DATA_DIR", "data/data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

MILVUS_PATH = os.environ.get("MILVUS_PATH", os.path.join(DATA_DIR, "embeddings.db"))
