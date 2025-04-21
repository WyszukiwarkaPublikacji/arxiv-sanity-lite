from typing import List
import numpy as np
from aslite.db import CompressedSqliteDict
class Paper:
    def __init__(self, arxiv_id: str, title: str, authors: List[str], abstract: str, vector):
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.arxiv_id = arxiv_id
        self.vector = vector

    @classmethod
    def from_id(cls, arxiv_id: str, db: CompressedSqliteDict, vector):
        paper_map = db[arxiv_id]
        print(paper_map)
        return Paper(arxiv_id=arxiv_id, title=paper_map['title'], authors=paper_map['authors'],
                     abstract=paper_map['summary'], vector=vector)

    def __repr__(self):
        return f"Paper(arxiv_id='{self.arxiv_id}', title='{self.title}', authors={self.authors})"