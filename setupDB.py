from db.Milvus.MilvusSetterDB import MilvusSetterDB
from db.SQLLite.OrmDB import creation_with_drop


def main():
    # MilvusSetterDB.create_collectio_metas()
    # MilvusSetterDB.create_collection_papers()
    creation_with_drop()

if __name__ == "__main__":
    main()








