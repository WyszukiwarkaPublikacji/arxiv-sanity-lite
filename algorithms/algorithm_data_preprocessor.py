import logging

from typing import Dict
import sys, os
from sklearn.decomposition import TruncatedSVD
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
print(os.getcwd())

import aslite.db as db
from db.Milvus.MilvusMetaRepository import MilvusMetaRepository
from db.Milvus.MilvusSetterDB import MilvusSetterDB
from db.Milvus.MilvusInstance import MilvusInstance


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Stałe konfiguracyjne
DIM = 100  # Liczba wymiarów po redukcji SVD

def reduce_dimensionality(features: Dict) -> np.ndarray:
    svd = TruncatedSVD(n_components=DIM, random_state=42)
    X_reduced = svd.fit_transform(features["x"])
    return X_reduced

def construct_index(features: dict, X_reduced: np.ndarray):
    MilvusInstance.connect_to_instance()
    repo = MilvusMetaRepository()

    repo.insert(MilvusSetterDB.COLLECTION_NAME2,keys=features['pids'],vectors=X_reduced.tolist())

def preprocess():
    logging.info("Ładowanie cech z bazy danych...")
    features = db.load_features()
    logging.info("Ładowanie bazy danych publikacji...")
    papers_db = db.get_papers_db(flag='c')
        
    # Tworzenie wektorów o obniżonej wymiarowości
    logging.info("Obniżanie wymiarowości wektorów...")
    X_reduced = reduce_dimensionality(features)
    logging.info("Konstruowanie indeksu...")
    
    logging.info("Przypisywanie publikacjom z bazy danych ich wektorów...")
    counter = 0
    for arxiv_id,vector in zip(features['pids'],X_reduced):
        current_paper = papers_db[arxiv_id]
        if 'paper_vector' in current_paper.keys():
            continue
        current_paper['paper_vector'] = vector
        papers_db[arxiv_id] = current_paper
        counter += 1
    logging.info(f"Przypisano wektory {counter}/{len(features['pids'])} publikacjom!")

    construct_index(features, X_reduced)

if __name__ == "__main__":
    preprocess()
