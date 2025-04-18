from abc import ABC, abstractmethod
from typing import List
from papers.paper import Paper
from aslite.db import CompressedSqliteDict

class Algorithm(ABC):
    def __init__(self, papers_db: CompressedSqliteDict):
        self.papers_db = papers_db
    
    @abstractmethod
    def recommend(self, papers: List[Paper], recommend_count: int) -> List[Paper]:
        pass
