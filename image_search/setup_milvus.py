from pymilvus import utility, FieldSchema, CollectionSchema, DataType, Collection, Index
from db.Milvus.MilvusInstance import MilvusInstance

    
COLLECTION_NAME = "images"

    
def create_image_collection():
    try:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512 + 384),
        ]

        MilvusInstance.connect_to_instance()

        if utility.has_collection(COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' already exists.")
            return True

        schema = CollectionSchema(fields, description="Similar Publications Collection")
        collection = Collection(name=COLLECTION_NAME, schema=schema)

        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT"
        }

        collection.create_index(field_name="publication_embedding", index_params=index_params)

        print(f"Collection '{COLLECTION_NAME}' created successfully with an index!")
        return True

    except Exception as e:
        print(f"Error creating collection: {e}")
        return False