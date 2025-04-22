from algorithms.rl_algorithm.models import Policy, QModel
from evaluation_methods.simulated_evaluation.simulated_evaluation import SimulatedEvaluation
from aslite.db import CompressedSqliteDict
from algorithms.rl_algorithm.replay_buffer import ReplayBuffer
from algorithms.rl_algorithm.rl_algorithm import RLAlgorithm
import aslite.db as db
from aslite.env import DATA_DIR

import torch
import torch.nn.functional as F
import copy
import os
from tqdm.auto import tqdm
import numpy as np
import random

import logging

BATCH_SIZE = 256
HIDDEN_UNITS = 64
LAYERS = 1
CRITIC_LR = 3e-4
ACTOR_LR = 1e-4

RUNS_COUNT = 2000
UPDATES_PER_RUN = 8
DISCOUNT_FACTOR = 0.0
ALPHA = 1.0
TAU = 0.1

UPDATE_EVERY = 2

START_RUNS = 0

EVALUATE_EVERY = 400

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 12321

MODEL_PATH = os.path.join(DATA_DIR, "trained_policy.pth")
Q_MODEL0_PATH = os.path.join(DATA_DIR, "trained_q0.pth")
Q_MODEL1_PATH = os.path.join(DATA_DIR, "trained_q1.pth")

def train_policy(papers_db: CompressedSqliteDict, recommend_count: int, dimensionality: int, force_train=False) -> Policy:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    sim_eval = SimulatedEvaluation(papers_db=papers_db, runs=1, begin_author=-2000, end_author=-1)
    policy = Policy(recommend_count=recommend_count, dimensionality=dimensionality,
                    hidden_units=HIDDEN_UNITS, layers=LAYERS).to(DEVICE)
    
    if os.path.exists(MODEL_PATH) and not force_train:
        logging.info(f"Znaleziono model w pliku {MODEL_PATH}. Ładowanie modelu zamiast trenowania.")
        policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        policy.eval()
        return policy
    
    if os.path.exists(MODEL_PATH) and force_train:
        logging.info(f"Znaleziono model w pliku {MODEL_PATH}. Kontynuowanie trenowania.")
        policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    replay_buffer = ReplayBuffer(max_size=100000)
    q_models = [
        QModel(recommend_count=recommend_count, dimensionality=dimensionality,
               hidden_units=HIDDEN_UNITS, layers=LAYERS).to(DEVICE)
        for _ in range(2)
    ]
    
    if os.path.exists(Q_MODEL0_PATH) and force_train:
        logging.info(f"Znaleziono model w pliku {Q_MODEL0_PATH}. Kontynuowanie trenowania.")
        q_models[0].load_state_dict(torch.load(Q_MODEL0_PATH, map_location=DEVICE))

    if os.path.exists(Q_MODEL1_PATH) and force_train:
        logging.info(f"Znaleziono model w pliku {Q_MODEL1_PATH}. Kontynuowanie trenowania.")
        q_models[1].load_state_dict(torch.load(Q_MODEL1_PATH, map_location=DEVICE))
    
    q_target_models = [copy.deepcopy(q) for q in q_models]
    for tgt in q_target_models:
        for p in tgt.parameters():
            p.requires_grad = False

    q_optimizers = [torch.optim.Adam(q.parameters(), lr=CRITIC_LR, betas=(0.5,0.7)) for q in q_models]
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=ACTOR_LR, betas=(0.5,0.7))

    metrics = []
    entropies = []
    q_losses = []
    p_losses = []
    q_values = []
    y_values = []
    
    best_score_so_far = 0.0
    
    rl_alg = RLAlgorithm(papers_db, policy=policy, replay_buffer=replay_buffer)
    for run in tqdm(range(1, RUNS_COUNT + 1)):
        policy.eval()
        res = sim_eval.evaluate(rl_alg, recommend_count=recommend_count)
        metric = res['Average run length']
        metrics.append(metric)
        
        rl_alg.clean()

        if run % EVALUATE_EVERY == 0:
            sim_eval_to_save = SimulatedEvaluation(papers_db=papers_db, runs=100, begin_author=0, end_author=-2000)
            rl_alg_to_save = RLAlgorithm(papers_db=papers_db, policy=policy)
            
            res_to_save = sim_eval_to_save.evaluate(rl_alg_to_save, recommend_count=recommend_count)
            metric_to_save = res_to_save['Average run length']
            
            if metric_to_save > best_score_so_far:
                tqdm.write(f"Znaleziono nowy najlepszy model z metryką: {metric_to_save}")
                
                # Zapisz wytrenowany model do pliku
                torch.save(policy.state_dict(), MODEL_PATH)
                tqdm.write(f"Model zapisany do {MODEL_PATH}")
    
                torch.save(q_models[0].state_dict(), Q_MODEL0_PATH)
                tqdm.write(f"Model zapisany do {Q_MODEL0_PATH}")
    
                torch.save(q_models[1].state_dict(), Q_MODEL1_PATH)
                tqdm.write(f"Model zapisany do {Q_MODEL1_PATH}")
                
                best_score_so_far = metric_to_save

        if len(replay_buffer.buffer) < BATCH_SIZE or run < START_RUNS or run % UPDATE_EVERY != 0:
            continue

        if run % 50 == 0:
            tqdm.write(f"Run {run} | Current metric value: {np.mean(metrics[-50:]):.4f} | Current entropy value: {np.mean(entropies[-50:]):.4f} | Current p-loss: {np.mean(p_losses[-50:]):.4f} | Current q-loss: {np.mean(q_losses[-50:]):.4f} | Current q-value: {np.mean(q_values[-50:]):.4f} | Current y value: {np.mean(y_values[-50:]):.4f}")

        policy.train()
        for _ in range(UPDATES_PER_RUN):
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = [i.to(DEVICE) for i in [states, actions, rewards, next_states, dones]]
            
            with torch.no_grad():
                a_next, logp_next = Policy.sample(policy(next_states), dimensionality=dimensionality)
                q1_targ = q_target_models[0](next_states, a_next)
                q2_targ = q_target_models[1](next_states, a_next)
                q_targ  = torch.min(q1_targ, q2_targ)
                
                logp_next = logp_next.squeeze(dim=-1)
                entropies.append(torch.mean(logp_next).item())

                y = rewards + DISCOUNT_FACTOR * (1 - dones) * (q_targ - ALPHA * logp_next)
                
                y_values.append(y.mean().item())

            for q, opt in zip(q_models, q_optimizers):
                loss_q = F.mse_loss(q(states, actions), y)
                q_losses.append(loss_q.item())
                opt.zero_grad()
                loss_q.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
                opt.step()

            a_curr, logp_curr = Policy.sample(policy(states), dimensionality=dimensionality)
            entropies.append(torch.mean(logp_curr).item())
            q1_curr = q_models[0](states, a_curr)
            q2_curr = q_models[1](states, a_curr)
            q_curr_min = torch.min(q1_curr, q2_curr)
            loss_pi = (ALPHA * logp_curr - q_curr_min).mean()
            p_losses.append(loss_pi.item())
            q_values.append(q_curr_min.mean().item())

            policy_optimizer.zero_grad()
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            policy_optimizer.step()

            for tgt, src in zip(q_target_models, q_models):
                for p_t, p in zip(tgt.parameters(), src.parameters()):
                    p_t.data.mul_(1 - TAU)
                    p_t.data.add_(TAU * p.data)
    
    if os.path.exists(MODEL_PATH):
        policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    return policy

if __name__ == "__main__":
    papers_db = db.get_papers_db()
    train_policy(papers_db, recommend_count=5, force_train=True)
