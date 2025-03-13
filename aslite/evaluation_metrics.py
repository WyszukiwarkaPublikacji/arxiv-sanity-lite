from typing import List, Set, Dict
from math import log2

class EvaluationMetrics:
    @staticmethod
    def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        recommended_at_k = recommended[:k]
        relevant_at_k = [item for item in recommended_at_k if item in relevant]
        return len(relevant_at_k) / k

    @staticmethod
    def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        recommended_at_k = recommended[:k]
        relevant_at_k = [item for item in recommended_at_k if item in relevant]
        return len(relevant_at_k) / len(relevant)

    @staticmethod
    def dcg_at_k(recommended: List[str], relevant: Dict[str, int], k: int) -> float:
        dcg = 0.0
        for i in range(k):
            if recommended[i] in relevant:
                dcg += relevant[recommended[i]] / log2(i + 2)
        return dcg

    @staticmethod
    def idcg_at_k(relevant: Dict[str, int], k: int) -> float:
        idcg = 0.0
        sorted_relevant = sorted(relevant.values(), reverse=True)
        for i in range(min(k, len(sorted_relevant))):
            idcg += sorted_relevant[i] / log2(i + 2)
        return idcg

    @staticmethod
    def ndcg_at_k(recommended: List[str], relevant: Dict[str, int], k: int) -> float:
        dcg = EvaluationMetrics.dcg_at_k(recommended, relevant, k)
        idcg = EvaluationMetrics.idcg_at_k(relevant, k)
        return dcg / idcg if idcg > 0 else 0.0