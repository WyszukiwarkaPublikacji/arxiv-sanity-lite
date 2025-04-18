from abc import ABC, abstractmethod
from typing import List
from papers.paper import Paper
from annoy import AnnoyIndex
from aslite.db import CompressedSqliteDict

class Algorithm(ABC):
    def __init__(self, index: AnnoyIndex, papers_db: CompressedSqliteDict):
        self.index = index
        self.papers_db = papers_db
    
    @abstractmethod
    def recommend(self, papers: List[Paper], recommend_count: int) -> List[Paper]:
        pass
