import os
from pymilvus import connections
from aslite.env import MILVUS_PATH

class MilvusInstance:
    @staticmethod
    def connect_to_instance()-> bool:
        try:
            connections.connect(alias="default", uri=MILVUS_PATH)
            print("Milvus connection successful!")
            return True
        except Exception as e:
            print(e)
            return False
