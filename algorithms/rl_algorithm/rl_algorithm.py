from algorithms.algorithm import Algorithm
from algorithms.rl_algorithm.models import Policy
from aslite.db import CompressedSqliteDict
from papers.paper import Paper
from typing import List
from db.Milvus.MilvusInstance import MilvusInstance
from db.Milvus.MilvusMetaRepository import MilvusMetaRepository
from db.Milvus.MilvusSetterDB import MilvusSetterDB
from db.Milvus.MilvusInstance import MilvusInstance
from evaluation_methods.simulated_evaluation.prepare_data import pad_tensor
from algorithms.rl_algorithm.replay_buffer import ReplayBuffer
from algorithms.algorithm_data_preprocessor import DIM

import torch
import numpy as np

class RLAlgorithm(Algorithm):
    def __init__(self, papers_db: CompressedSqliteDict, dimensionality=DIM, recommend_count=5, policy: Policy = None, replay_buffer: ReplayBuffer = None, max_sessions: int = 50):
        super().__init__(papers_db=papers_db)
        
        self.dimensionality = dimensionality
        if not policy:
            from algorithms.rl_algorithm.rl_training import train_policy
            policy = train_policy(papers_db=papers_db, dimensionality=dimensionality, recommend_count=recommend_count)
        
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.max_sesions = max_sessions
        
        self.previous_len = 0
        
        self.previous_reward = 0.0
        self.previous_len = 0
        self.previous_state = None
        self.previous_action = None
        
        MilvusInstance.connect_to_instance()
    
    def clean(self):
        self.previous_reward = 0.0
        self.previous_len = 0
        self.previous_state = None
        self.previous_action = None
    
    def recommend(self, papers: List[Paper], recommend_count: int) -> List[Paper]:
        if len(papers) != self.previous_len:
            self.previous_reward = 1.0
        else:
            self.previous_reward = 0.0
        
        vectors_list = [paper.vector for paper in papers]
        
        milvus_repository = MilvusMetaRepository()
        milvus_repository.get_collection(MilvusSetterDB.COLLECTION_NAME2)
        
        device = next(self.policy.parameters()).device
        vectors = torch.tensor(np.array(vectors_list)).to(device)
        
        if not vectors_list:
            vectors = torch.zeros(0,self.dimensionality).to(device)
        
        vectors = pad_tensor(vectors, target_length=self.max_sesions)
        
        vectors = vectors.unsqueeze(dim=0)
        
        with torch.no_grad():
            distribution_output = self.policy(vectors)
        
        expected_vectors, _ = Policy.sample(distribution_output, dimensionality=self.dimensionality)
        expected_vectors = expected_vectors.squeeze(dim=0)
        
        if self.replay_buffer is not None and self.previous_state is not None:         
            #min_dist = torch.mean(torch.min(torch.sqrt(torch.sum((self.previous_action.unsqueeze(dim=0) - self.previous_state.unsqueeze(dim=1)) ** 2, dim=-1)), dim=0)[0])
                        
            #self.previous_reward -= min_dist / 25
            
            self.replay_buffer.add(self.previous_state, self.previous_action, self.previous_reward, vectors.squeeze(dim=0), done_signal=1 if len(vectors_list) == self.max_sesions else 0)
        
        self.previous_len = len(papers)
        self.previous_state = vectors.squeeze(dim=0)
        self.previous_action = expected_vectors
                
        recommended_papers = []
        for expected_vector in expected_vectors:
            for res in milvus_repository.search(expected_vector.tolist(), top_k=self.max_sesions + recommend_count + 1)[0]:
                item = res.key
                if item not in [paper.arxiv_id for paper in papers] and item not in [paper.arxiv_id for paper in recommended_papers]:
                    paper = Paper.from_id(arxiv_id=item, db=self.papers_db)
                    recommended_papers.append(paper)
                    break
        
        return recommended_papers
        