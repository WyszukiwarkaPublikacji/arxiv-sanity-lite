from typing import List, Tuple, Dict, Set
from aslite.evaluation_metrics import EvaluationMetrics

class Evaluator:
    def __init__(self, test_data: List[Tuple[str, Set[str]]], k: int):
        self.test_data = test_data
        self.k = k

    def evaluate(self) -> Dict[str, float]:
        precision_scores = []
        recall_scores = []
        ndcg_scores = []

        for user_context, relevant_items in self.test_data:
            recommended_items = self.generate_recommendations(user_context)
            precision_scores.append(EvaluationMetrics.precision_at_k(recommended_items, relevant_items, self.k))
            recall_scores.append(EvaluationMetrics.recall_at_k(recommended_items, relevant_items, self.k))
            ndcg_scores.append(EvaluationMetrics.ndcg_at_k(recommended_items, relevant_items, self.k))

        avg_precision = sum(precision_scores) / len(precision_scores)
        avg_recall = sum(recall_scores) / len(recall_scores)
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)

        return {
            'precision@k': avg_precision,
            'recall@k': avg_recall,
            'ndcg@k': avg_ndcg
        }

    def generate_recommendations(self, user_context: str) -> List[str]:
        return []


