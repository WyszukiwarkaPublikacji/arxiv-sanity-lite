import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from evaluation_methods.simulated_evaluation.model import SimulatedEvaluationModel
import logging
import numpy as np
import random
from tqdm.auto import tqdm
import os

BATCH_SIZE = 32
NUM_EXPERTS = 1
LAYERS = 1
HIDDEN_UNITS = 128
EPOCHS = 4
LR = 0.0005

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1431

MODEL_PATH = "data/trained_model.pth"

def train_model(train_dataset: Dataset, eval_dataset: Dataset, dimensionality: int) -> SimulatedEvaluationModel:    
    model = SimulatedEvaluationModel(num_experts=NUM_EXPERTS, dimensionality=dimensionality, hidden_units=HIDDEN_UNITS, layers=LAYERS).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        logging.info(f"Model file found at {MODEL_PATH}. Loading model instead of training.")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        average_train_loss = 0.0
        average_train_accuracy = 0.0
        train_losses = []
        train_accuracies = []

        model.train()
        for batch_idx, (X, y) in enumerate(tqdm(train_dataloader)):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
                        
            optimizer.zero_grad()
            X_set = X[:, :-1]
            X_vec = X[:, -1]
            
            out = model(X_set, X_vec)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            
            average_train_loss += loss.item() / len(train_dataloader)
            batch_acc = torch.mean((torch.round(torch.sigmoid(out)) == y).float())
            average_train_accuracy += batch_acc / len(train_dataloader)
            
            train_losses.append(loss.item())
            train_accuracies.append(batch_acc.item())
            
            #if batch_idx % 200 == 199:
            #    logging.info(f"Batch {batch_idx + 1}/{len(train_dataloader)} | Loss avg: {np.mean(train_losses[-50:]):.4f} | Acc avg: {np.mean(train_accuracies[-50:]) * 100:.4f}%")
        
        average_eval_loss = 0.0
        average_eval_accuracy = 0.0
        
        model.eval()
        with torch.no_grad():
            for X, y in tqdm(eval_dataloader):
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                
                X_set = X[:, :-1]
                X_vec = X[:, -1]
                
                out = model(X_set, X_vec, training=False)
                loss = loss_fn(out, y)
                
                average_eval_loss += loss.item() / len(eval_dataloader)
                average_eval_accuracy += torch.mean((torch.round(out) == y).float()) / len(eval_dataloader)

        logging.info(f"Epoch {epoch + 1} | Train loss avg: {average_train_loss:.4f} | Train acc avg: {average_train_accuracy * 100:.4f}% | Eval loss avg: {average_eval_loss:.4f} | Eval acc avg: {average_eval_accuracy * 100:.4f}%")

    # Zapisz wytrenowany model do pliku
    torch.save(model.state_dict(), MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")

    return model
