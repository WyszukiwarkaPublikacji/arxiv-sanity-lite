from sqlalchemy import select
from sqlalchemy.orm import Session
from db.SQLLite.OrmDB import Papers

from db.SQLLiteAlchemyInstance import SQLAlchemyInstance
from db.Milvus.MilvusSetterDB import MilvusSetterDB
from pymilvus import utility
from db.SQLLite.OrmDB import creation


def main():
    instance = SQLAlchemyInstance()

    MilvusSetterDB.create_collectio_metas()
    MilvusSetterDB.create_collection_papers()
    #creation()

    return None
if __name__ == "__main__":
    main()

def run_db() -> None:
    MilvusSetterDB.create_collection_similar_publications()

if __name__ == '__main__':
    run_db()
