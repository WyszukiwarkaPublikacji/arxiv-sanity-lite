import logging
import aslite.db as db
from typing import Dict
from annoy import AnnoyIndex
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Stałe konfiguracyjne
DIM = 100  # Liczba wymiarów po redukcji SVD
N_TREES = 10  # Liczba drzew w Annoy Index (dla Approximate Nearest Neighbors)

def reduce_dimensionality(features: Dict):
    svd = TruncatedSVD(n_components=DIM, random_state=42)
    X_reduced = svd.fit_transform(features["x"])
    return X_reduced

def construct_index(X_reduced: np.ndarray):
    index = AnnoyIndex(DIM, "euclidean")
    for idx, vec in enumerate(X_reduced):
        index.add_item(idx, vec)
    index.build(N_TREES)
    return index

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
        
    for id in papers_db:
        if 'paper_vector' not in papers_db[id].keys():
            del papers_db[id]
    
    index = construct_index(X_reduced)
    
    return index
    