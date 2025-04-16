from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from pymilvus import utility

class MilvusMetaRepository:
    def get_collection(self, collection_name: str):
        if (utility.has_collection(collection_name)):
            self.collection = Collection(name=collection_name)
        else:
            Exception("Collection {} not found".format(collection_name))

    def insert(self, collection_name:str, keys, vectors):
        self.get_collection(collection_name)
        data_to_insert = [keys, vectors]
        return self.collection.insert(data=data_to_insert)

    def search(self, query_vector, top_k=5):
        self.collection.load()
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["key"]
        )
        return results
