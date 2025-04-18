from typing import List, Dict
from papers.paper import Paper
import random
from algorithms.algorithm import Algorithm
from aslite.db import CompressedSqliteDict

class RandomSampling(Algorithm):
    def __init__(self, papers_db: CompressedSqliteDict):
        super().__init__(papers_db)
        self.ids = list(self.papers_db.keys())
    
    def recommend(self, papers: List[Paper], recommend_count: int) -> List[Paper]:
        return [Paper.from_id(random.choice(self.ids), db=self.papers_db) for _ in range(recommend_count)]
