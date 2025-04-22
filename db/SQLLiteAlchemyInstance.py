import sys
import os
import sqlalchemy as db

class SQLAlchemyInstance:
    def __init__(self):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        SQLITE_PATH = os.path.join(BASE_DIR, 'data', 'papers.db')
        sys.path.append(BASE_DIR)

        self.engine = db.create_engine("sqlite:////" + SQLITE_PATH)

        self.conn = self.engine.connect()
        self.metadata = db.MetaData()

    def get_sqllite_metadata(self):
        return self.metadata

    def get_conn(self):
        return self.conn

    def get_engine(self):
        return self.engine