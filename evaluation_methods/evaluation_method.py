from abc import ABC, abstractmethod
from typing import Dict
from algorithms.algorithm import Algorithm
from aslite.db import CompressedSqliteDict

class EvaluationMethod(ABC):
    def __init__(self, papers_db: CompressedSqliteDict):
        self.papers_db = papers_db

    # Zwraca słownik z wynikami różnych metryk tą konkretną metodą
    @abstractmethod
    def evaluate(self, algorithm: Algorithm, recommend_count: int) -> Dict[str, float]:
        pass
