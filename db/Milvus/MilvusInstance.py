from pymilvus import connections

class MilvusInstance:
    @staticmethod
    def connect_to_instance(host='localhost', port='19530')-> bool:
        try:
            connections.connect(alias="default", host=host, port=port)
            print("Milvus connection successful!")
            return True
        except Exception as e:
            print(e)
            return False;
