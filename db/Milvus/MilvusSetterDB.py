import os
import sys

from pymilvus import utility, FieldSchema, CollectionSchema, DataType, Collection

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

from db.Milvus.MilvusInstance import MilvusInstance


class MilvusSetterDB:
    COLLECTION_NAME = "similar_publications"

    @staticmethod
    def create_collection_similar_publications() -> bool:
        try:
            # Define the fields schema for the collection
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="publication_embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
                FieldSchema(name="publication_id", dtype=DataType.INT64),
                FieldSchema(name="categories", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=1024),
            ]

            # Connect to the Milvus instance
            MilvusInstance.connect_to_instance()

            # Check if the collection already exists
            if utility.has_collection(MilvusSetterDB.COLLECTION_NAME):
                print(f"Collection '{MilvusSetterDB.COLLECTION_NAME}' already exists.")
                return True  # Return True if collection exists

            # Create collection schema
            schema = CollectionSchema(fields, description="Similar Publications Collection")

            # Create the collection
            collection = Collection(name=MilvusSetterDB.COLLECTION_NAME, schema=schema)

            print(f"Collection '{MilvusSetterDB.COLLECTION_NAME}' created successfully!")
            return True

        except Exception as e:
            print(f"Error creating collection: {e}")
            return False
