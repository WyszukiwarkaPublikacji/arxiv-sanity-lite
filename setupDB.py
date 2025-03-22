from db.Milvus.MilvusSetterDB import MilvusSetterDB

def run_db() -> None:
    MilvusSetterDB.create_collection_similar_publications()

if __name__ == '__main__':
    run_db()