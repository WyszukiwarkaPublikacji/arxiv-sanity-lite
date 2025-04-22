from pymilvus import utility, FieldSchema, CollectionSchema, DataType, Collection, Index

from db.Milvus.MilvusInstance import MilvusInstance


class MilvusSetterDB:
    COLLECTION_NAME = "metas"
    COLLECTION_NAME2 = "papers"

    @staticmethod
    def create_collectio_metas() -> bool:
        from algorithms.algorithm_data_preprocessor import DIM
        
        try:
            fields = [
                FieldSchema(name="key", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
                FieldSchema(name="value", dtype=DataType.FLOAT_VECTOR, dim=DIM)
            ]

            MilvusInstance.connect_to_instance()

            if utility.has_collection(MilvusSetterDB.COLLECTION_NAME):
                utility.drop_collection(MilvusSetterDB.COLLECTION_NAME)

            schema = CollectionSchema(fields, description="Similar Publications meta", auto_id=False)
            collection = Collection(name=MilvusSetterDB.COLLECTION_NAME, schema=schema)

            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT"
            }

            collection.create_index("value", index_params=index_params)

            print(f"Collection '{MilvusSetterDB.COLLECTION_NAME}' created successfully with an index!")
            return True

        except Exception as e:
            print(f"Error creating collection: {e}")
            return False

    @staticmethod
    def create_collection_papers() -> bool:
        from algorithms.algorithm_data_preprocessor import DIM
        
        try:
            fields = [
                FieldSchema(name="key", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
                FieldSchema(name="value", dtype=DataType.FLOAT_VECTOR, dim=DIM)
            ]

            MilvusInstance.connect_to_instance()
            
            if utility.has_collection(MilvusSetterDB.COLLECTION_NAME2):
                utility.drop_collection(MilvusSetterDB.COLLECTION_NAME2)

            schema = CollectionSchema(fields, description="Similar Publications papers", auto_id=False)
            collection = Collection(name=MilvusSetterDB.COLLECTION_NAME2, schema=schema)

            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT"
            }

            collection.create_index("value", index_params=index_params)

            print(f"Collection '{MilvusSetterDB.COLLECTION_NAME2}' created successfully with an index!")
            return True

        except Exception as e:
            print(f"Error creating collection: {e}")
            return False