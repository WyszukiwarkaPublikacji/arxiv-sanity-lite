import os
import sys

from pymilvus import Collection, utility

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

from MilvusInstance import MilvusInstance
from db.DataValidator import DataValidator


class MilvusORMDB:
    @staticmethod
    def insert_into_milvus(id: int, publication_id: int, collection_name:str, **kwargs)-> bool:
        if (DataValidator.validate_data(kwargs)):
            data_for_insertion = [id,
                                  publication_id,
                                  DataValidator.get_validated_array_with_data_for_insertion()]
            try:
                MilvusInstance.connect_to_instance()
                collection = Collection(name = collection_name)
                collection.insert(data_for_insertion)
                return True
            except Exception as e:
                print(e)
                return False

    @staticmethod
    def list_collections()-> list[Collection]:
        if (MilvusInstance.connect_to_instance()):
            collections = utility.list_collections()
            return collections





