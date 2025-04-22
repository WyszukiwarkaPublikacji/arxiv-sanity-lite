import sys
import os
import sqlalchemy as db
from aslite.env import DATA_DIR

class SQLAlchemyInstance:
    def __init__(self):
        SQLITE_PATH = os.path.join(DATA_DIR, "papers.db")

        self.engine = db.create_engine("sqlite:////" + SQLITE_PATH)

        self.conn = self.engine.connect()
        self.metadata = db.MetaData()

    def get_sqllite_metadata(self):
        return self.metadata

    def get_conn(self):
        return self.conn

    def get_engine(self):
        return self.engine