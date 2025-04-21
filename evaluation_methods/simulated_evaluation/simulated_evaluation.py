from evaluation_methods.evaluation_method import EvaluationMethod
from aslite.db import CompressedSqliteDict
from typing import Dict, List
from algorithms.algorithm import Algorithm
from evaluation_methods.simulated_evaluation.prepare_data import prepare_data, pad_tensor, group_papers_by_authors, read_test_authors
from evaluation_methods.simulated_evaluation.train_model import train_model
import numpy as np
import torch
import random
from papers.paper import Paper
from tqdm.auto import tqdm

NO_INTEREST_IN_A_ROW_THRESHOLD = 3
RUNS_PER_METHOD = 1000
MAXIMUM_SESSIONS = 50
RANDOM_CLICK_RATE = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SimulatedEvaluation(EvaluationMethod):
    def __init__(self, papers_db: CompressedSqliteDict):
        super().__init__(papers_db)
        
        self.papers_db = papers_db
        # Zacznę od wytrenowania modelu, który przewidzi łączne prawdopodobieństwo, że publikacja należy do zbioru opublikowanych
        train_dataset, eval_dataset = prepare_data(papers_db)
        print(f"Skonstruowano zbiory danych o wielkościach: {len(train_dataset)}, {len(eval_dataset)}")
        self.model = train_model(train_dataset, eval_dataset, dimensionality=100)
        self.evaluation_authors = read_test_authors()
    
    # Zastosowanie wzorku wyprowadzonego z Twierdzenia Bayesa
    @classmethod
    def compute_real_probability(cls, model_probability: float, random_click_rate: float):
        return (random_click_rate * model_probability) / (random_click_rate * model_probability + (1 - random_click_rate) * (1 - model_probability))
    
    def recommend(self, initial_papers: List[Paper], papers_to_recommend: List[Paper]) -> Paper:
        initial_papers_vectors = [paper.vector for paper in initial_papers]
        
        X_set = torch.tensor(np.array(initial_papers_vectors))
        X_set = pad_tensor(X_set,98).unsqueeze(dim=0).to(DEVICE)
        
        probabilities = []
        for paper in papers_to_recommend:
            vec = paper.vector
            X_vec = torch.tensor(vec).unsqueeze(dim=0).to(DEVICE)
            prob = self.model(X_set,X_vec,training=False)
            probabilities.append(SimulatedEvaluation.compute_real_probability(prob.item(), random_click_rate=RANDOM_CLICK_RATE))
    
        for paper,probability in zip(papers_to_recommend,probabilities):
            if random.random() < probability:
                return paper
        
        return None
    
    def evaluate(self, algorithm: Algorithm, recommend_count: int) -> Dict[str, float]:
        metrics = {
            'Average run length': 0.0,
            'Run length variance': 0.0,
            'Reached maximum sessions probability': 0.0,
        }
        
        papers_by_authors = group_papers_by_authors(self.papers_db)
        
        run_lengths = []
        # Każdy algorytm jest ewaluowany na RUNS_PER_METHOD losowych osobach
        for run in tqdm(range(RUNS_PER_METHOD)):
            author = random.choice(self.evaluation_authors)
            initial_papers = [Paper.from_id(paper, self.papers_db) for paper in papers_by_authors[1][author]]
            
            # Zakładamy, że osoba już nie powraca po NO_INTEREST_AFTER_BLANK_SESSION sesjach w których nie była zainteresowana żadną poleconą publikacją
            no_interest_counter = 0
            # Każda osoba ma maksymalnie MAXIMUM_SESSIONS sesji
            papers_so_far = []
            
            counter = 0
            for session in range(MAXIMUM_SESSIONS):
                if no_interest_counter == NO_INTEREST_IN_A_ROW_THRESHOLD:
                    break
                
                clicked_paper = self.recommend(initial_papers, algorithm.recommend(papers=papers_so_far,recommend_count=recommend_count))
                
                if not clicked_paper:
                    no_interest_counter += 1
                    continue
                no_interest_counter = 0
                    
                papers_so_far.append(clicked_paper)
                
                counter += 1
            
            run_lengths.append(counter)
        
        metrics['Average run length'] = np.mean(run_lengths)
        metrics['Reached maximum sessions probability'] = np.mean(np.array(run_lengths) == MAXIMUM_SESSIONS)
        metrics['Run length variance'] = np.var(run_lengths)
        
        return metrics
    