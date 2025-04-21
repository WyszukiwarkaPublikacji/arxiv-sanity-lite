from typing import List
from papers.paper import Paper
import numpy as np
from algorithms.algorithm import Algorithm
from aslite.db import CompressedSqliteDict
from algorithms.random_sampling import RandomSampling
from db.Milvus.MilvusMetaRepository import MilvusMetaRepository
from db.Milvus.MilvusSetterDB import MilvusSetterDB
from db.Milvus.MilvusInstance import MilvusInstance

class PaperLocalSampling(Algorithm):
    def __init__(self, papers_db: CompressedSqliteDict):
        super().__init__(papers_db)
        
        MilvusInstance.connect_to_instance()
    
    def recommend(self, papers: List[Paper], recommend_count: int) -> List[Paper]:
        if len(papers) == 0:
            # Przy braku jakichkolwiek informacji zwracamy losowe publikacje
            return RandomSampling(self.papers_db).recommend(papers, recommend_count=recommend_count)
        
        read_vectors = [paper.vector for paper in papers]
        expected_vector = self.sample_from_multi_dim(read_vectors, sigma=0.1)
        return [Paper.from_id(self.find_closest_paper(expected_vector, read_vectors), db=self.papers_db) for _ in range(recommend_count)]
    
    def sample_from_one_dim(self, papers_one_dim: np.ndarray, sigma: float = 1.0) -> float:
        random_paper = np.random.choice(papers_one_dim)
        return np.random.normal(loc=random_paper, scale=sigma)
    
    def sample_from_multi_dim(self, papers: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        np_papers = np.array(papers)
        return np.array([self.sample_from_one_dim(np_papers[:, dim], sigma) for dim in range(np_papers.shape[1])])
    
    def find_closest_paper(self, expected_vector: np.ndarray, read_vectors: List[np.ndarray], num_neighbors: int = 55) -> str:
        milvus_repository = MilvusMetaRepository()
        milvus_repository.get_collection(MilvusSetterDB.COLLECTION_NAME2)
        
        for res in milvus_repository.search(expected_vector.tolist(), top_k=num_neighbors)[0]:
            item = res.key
            item_vector = np.array(self.papers_db[item]['paper_vector'])
            if not any(np.array_equal(item_vector, v) for v in read_vectors):
                read_vectors.append(item_vector)
                return item
        return None


