from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from pymilvus import utility

class MilvusMetaRepository:
    def get_collection(self, collection_name: str):
        if (utility.has_collection(collection_name)):
            self.collection = Collection("papers", using="default")

        else:
            Exception("Collection {} not found".format(collection_name))

    def insert(self, collection_name:str, keys, vectors):
        self.get_collection(collection_name)
        print(self.collection)

        # self.collection = Collection(collection_name, using="default")
        count = 0

        for i in range(0, len(keys)):
            self.collection.insert({"key": keys[i], 'value': vectors[i]})
            count += 1

        self.collection.load()
        print(f"Wstawiono {count} rekordów do {collection_name}")

    def search(self, query_vector, top_k=5):
        self.collection.load()
        results = self.collection.search(
            data=[query_vector],
            anns_field="value",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["key"]
        )
        return results
