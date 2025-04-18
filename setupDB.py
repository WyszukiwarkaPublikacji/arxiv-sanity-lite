from sqlalchemy import select
from sqlalchemy.orm import Session
from db.SQLLite.OrmDB import Papers
from db.SQLLiteAlchemyInstance import SQLAlchemyInstance
from db.Milvus.MilvusSetterDB import MilvusSetterDB
from pymilvus import utility


def main():
    instance = SQLAlchemyInstance()
    engine = instance.get_engine()

    MilvusSetterDB.create_collectio_metas()
    MilvusSetterDB.create_collection_papers()
    # with Session(engine) as session:
    #     stmt = select(Papers).limit(2)
    #     result = session.execute(stmt)
    #     repo = MilvusMetaRepository()
    #
    #     for row in result:
    #         key = row[0]
    #         binary_vector = row[1]
    #         repo.insert('papers', key, binary_vector)


    return None
if __name__ == "__main__":
    main()








