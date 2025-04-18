import torch
import torch.nn as nn

class MOESubModel(nn.Module):
    def __init__(self, dimensionality: int, hidden_units: int, layers: int):
        super().__init__()
        
        self.dimensionality = dimensionality
        
        self.encoder = nn.GRU(input_size=2 * dimensionality, hidden_size=hidden_units, num_layers=layers, bidirectional=True, batch_first=True)
        self.mha = nn.MultiheadAttention(embed_dim=2 * hidden_units, num_heads=2 * hidden_units // 16, batch_first=True)
        self.layer_norm = nn.LayerNorm(normalized_shape=(2 * hidden_units))
        self.batch_norm = nn.BatchNorm1d(num_features=2 * dimensionality)
    
    def forward(self, X) -> torch.Tensor:
        shape_cp = X.shape
        
        X = X.reshape(X.shape[0] * X.shape[1], self.dimensionality * 2)
        X = self.batch_norm(X)
        X = X.reshape(shape_cp[0], shape_cp[1], shape_cp[2])
        
        out, _ = self.encoder(X)
        out = self.layer_norm(out)
        out, _ = self.mha(out, out, out)
        
        return out

class SimulatedEvaluationModel(nn.Module):
    def __init__(self, num_experts: int, dimensionality: int, hidden_units: int, layers: int):
        super().__init__()
        
        self.hidden_units = hidden_units
        self.experts = nn.ModuleList([MOESubModel(dimensionality=dimensionality, hidden_units=hidden_units, layers=layers) for i in range(num_experts)])
        self.fc = nn.Linear(in_features=2 * hidden_units, out_features=1)
        self.gate = nn.Linear(in_features=2 * dimensionality, out_features=num_experts)
    
    # Model przyjmuje zbiór i ma określić czy w prawdziwym zbiorze znajduje się ten wektor razem z pozostałymi
    def forward(self, X_set, X_vec, training=True) -> torch.Tensor:
        X_vec = X_vec.unsqueeze(dim=1)
        X_vec = X_vec.expand(-1, X_set.shape[1], -1)
        X = torch.cat((X_set,X_vec),dim=-1)
        
        gate_res = torch.softmax(self.gate(X),dim=-1)
        out = torch.zeros(X_set.shape[0], X_set.shape[1], 2 * self.hidden_units).to(X_set.device)
        for idx, expert in enumerate(self.experts):
            result = gate_res[:,:,idx].unsqueeze(dim=-1)
            out += result * expert(X)
        
        X = torch.mean(out, dim=1)
        X = torch.relu(X)
        X = self.fc(X)
                
        # Gdy model jest poza treningiem to niech zwraca prawdopodobieństwa
        if not training:
            X = torch.sigmoid(X)
        
        X = X.squeeze(dim=-1)
        
        return X


