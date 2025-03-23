import os
import sys

from pymilvus import utility, FieldSchema, CollectionSchema, DataType, Collection, Index

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

from db.Milvus.MilvusInstance import MilvusInstance


class MilvusSetterDB:
    COLLECTION_NAME = "similar_publications"

    @staticmethod
    def create_collection_similar_publications() -> bool:
        try:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="publication_embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
                FieldSchema(name="publication_id", dtype=DataType.INT64),
                FieldSchema(name="categories", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=1024),
            ]

            MilvusInstance.connect_to_instance()

            if utility.has_collection(MilvusSetterDB.COLLECTION_NAME):
                print(f"Collection '{MilvusSetterDB.COLLECTION_NAME}' already exists.")
                return True

            schema = CollectionSchema(fields, description="Similar Publications Collection")
            collection = Collection(name=MilvusSetterDB.COLLECTION_NAME, schema=schema)

            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT"
            }

            collection.create_index(field_name="publication_embedding", index_params=index_params)

            print(f"Collection '{MilvusSetterDB.COLLECTION_NAME}' created successfully with an index!")
            return True

        except Exception as e:
            print(f"Error creating collection: {e}")
            return False
