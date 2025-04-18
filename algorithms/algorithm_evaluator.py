import logging
from typing import List, Dict
import aslite.db as db
from algorithms.paper_local_sampling import PaperLocalSampling
from algorithms.individual_centroid_sampling import IndividualCentroidSampling
from algorithms.global_centroid_sampling import GlobalCentroidSampling
from algorithms.random_sampling import RandomSampling
from algorithms.algorithm_data_preprocessor import preprocess
from evaluation_methods.simulated_evaluation.simulated_evaluation import SimulatedEvaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_pid_mapping(pids: List[str]) -> Dict[str, int]:
    return {pid: idx for idx, pid in enumerate(pids)}

# Uruchamia wszystkie algorytmy na ewaluacji
if __name__ == "__main__":
    logging.info("Rozpoczynanie preprocessingu danych.")
    index = preprocess()
    
    logging.info("Ładowanie bazy danych publikacji w wersji read-only...")
    papers_db = db.get_papers_db()
    
    evaluation = SimulatedEvaluation(papers_db)
    
    recommender_classes = [RandomSampling, PaperLocalSampling, IndividualCentroidSampling, GlobalCentroidSampling]
    
    logging.info("Rozpoczynam rekomendowanie...")
    for recommender_class in recommender_classes:
        recommender = recommender_class(index, papers_db)
        
        metrics = evaluation.evaluate(recommender, recommend_count=5)
        
        print("\nWyniki ewaluacji dla:", recommender_class.__name__)
        print("-" * 50)
        for metric_name, value in metrics.items():
            print(f"{metric_name:<35}: {value:.4f}")
        print("-" * 50)
        
        