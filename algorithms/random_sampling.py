from typing import List, Dict
from papers.paper import Paper
from annoy import AnnoyIndex
import random
from algorithms.algorithm import Algorithm
from aslite.db import CompressedSqliteDict

class RandomSampling(Algorithm):
    def __init__(self, index: AnnoyIndex, papers_db: CompressedSqliteDict):
        super().__init__(index, papers_db)
    
    def recommend(self, papers: List[Paper], recommend_count: int) -> List[Paper]:
        return [Paper.from_id(self.papers_db[str(random.randint(0, self.index.get_n_items() - 1))]['_id'], db=self.papers_db) for _ in range(recommend_count)]
