from typing import List
from papers.paper import Paper
from annoy import AnnoyIndex
import numpy as np
from algorithms.algorithm import Algorithm
from algorithms.random_sampling import RandomSampling
from sklearn.cluster import KMeans
from aslite.db import CompressedSqliteDict

GLOBAL_CENTROIDS = 200

class GlobalCentroidSampling(Algorithm):
    def __init__(self, index: AnnoyIndex, papers_db: CompressedSqliteDict):
        super().__init__(index, papers_db)
        
        vectors = np.zeros((0,len(index.get_item_vector(0))))
        for idx in range(index.get_n_items()):
            vector = index.get_item_vector(idx)
            vectors = np.concatenate((vectors,np.expand_dims(np.array(vector),axis=0)),axis=0)
        
        k = min(index.get_n_items(), GLOBAL_CENTROIDS)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(vectors)
        self.centroids = kmeans.cluster_centers_
    
    def recommend(self, papers: List[Paper], recommend_count: int) -> List[Paper]:
        if len(papers) == 0:
            # Przy braku jakichkolwiek informacji zwracamy losowe publikacje
            return RandomSampling(self.index, self.papers_db).recommend(papers, recommend_count=recommend_count)
        
        read_vectors = [self.index.get_item_vector(self.papers_db[paper.arxiv_id]['index_id']) for paper in papers]
        
        expected_vector = self.sample_from_multi_dim(self.centroids, sigma=0.1)
        
        return [Paper.from_id(self.find_closest_paper(expected_vector, read_vectors), db=self.papers_db) for _ in range(recommend_count)]
    
    def sample_from_one_dim(self, papers_one_dim: np.ndarray, sigma: float = 1.0) -> float:
        random_paper = np.random.choice(papers_one_dim)
        return np.random.normal(loc=random_paper, scale=sigma)
    
    def sample_from_multi_dim(self, papers: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        np_papers = np.array(papers)
        return np.array([self.sample_from_one_dim(np_papers[:, dim], sigma) for dim in range(np_papers.shape[1])])
    
    def find_closest_paper(self, expected_vector: np.ndarray, read_vectors: List[np.ndarray], num_neighbors: int = 100) -> str:
        for item in self.index.get_nns_by_vector(expected_vector, num_neighbors):
            item_vector = np.array(self.index.get_item_vector(item))
            if not any(np.array_equal(item_vector, v) for v in read_vectors):
                read_vectors.append(item_vector)
                return self.papers_db[str(item)]['_id']
        return None


