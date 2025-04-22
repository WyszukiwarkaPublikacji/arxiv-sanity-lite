import os
from pymilvus import connections

DATA_DIR = os.environ.get("DATA_DIR", "./data/db")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
MILVUS_PATH = os.environ.get("MILVUS_PATH", os.path.join(DATA_DIR, "embeddings.db"))

class MilvusInstance:
    @staticmethod
    def connect_to_instance(host='localhost', port='19530')-> bool:
        try:
            connections.connect(alias="default", host=host, port=port)
            print("Milvus connection successful!")
            return True
        except Exception as e:
            print(e)
            return False;