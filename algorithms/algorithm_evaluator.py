import logging
from typing import List, Dict
import aslite.db as db
from algorithms.paper_local_sampling import PaperLocalSampling
from algorithms.random_sampling import RandomSampling
from evaluation_methods.simulated_evaluation.simulated_evaluation import SimulatedEvaluation
from algorithms.rl_algorithm.rl_algorithm import RLAlgorithm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_pid_mapping(pids: List[str]) -> Dict[str, int]:
    return {pid: idx for idx, pid in enumerate(pids)}

# Uruchamia wszystkie algorytmy na ewaluacji
if __name__ == "__main__":    
    logging.info("Ładowanie bazy danych publikacji w wersji read-only...")
    papers_db = db.get_papers_db()
    
    # Ostatnie 2000 autorów są po to aby wytrenować model reinforcement learningiem 
    evaluation = SimulatedEvaluation(papers_db, begin_author=0, end_author=-2000)
    
    recommender_classes = [RLAlgorithm, PaperLocalSampling, RandomSampling]
    
    logging.info("Rozpoczynam rekomendowanie...")
    for recommender_class in recommender_classes:
        recommender = recommender_class(papers_db)
        
        metrics = evaluation.evaluate(recommender, recommend_count=5)
        
        print("\nWyniki ewaluacji dla:", recommender_class.__name__)
        print("-" * 50)
        for metric_name, value in metrics.items():
            print(f"{metric_name:<35}: {value:.4f}")
        print("-" * 50)
        
        