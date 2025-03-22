import sqlite3
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DB_PATH = os.path.join(ROOT_DIR, 'data', 'papers.db')

os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)
from db.Milvus.MilvusORMDB import MilvusORMDB


class TransferWholeDataFromToMivlus:
    def transferData(self):
        pass
        # sqliteConnection = sqlite3.connect('./data/papers.db')
        # cursor = sqliteConnection.cursor()
        # cursor.execute('select * from papers')
        # rows = cursor.fetchall()
        # sqliteConnection.close()
        # for row in rows:
        #     print(row)
    # def sendToMivlus(self, row):


trans = TransferWholeDataFromToMivlus()
trans.transferData()