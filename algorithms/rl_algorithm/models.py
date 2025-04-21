import torch
import torch.nn as nn
import math

class QBlock(nn.Module):
    def __init__(self, recommend_count: int, dimensionality: int, hidden_units: int, layers: int):
        super().__init__()
        
        self.dimensionality = dimensionality
        self.recommend_count = recommend_count
        
        self.encoder = nn.GRU(input_size=dimensionality + recommend_count * dimensionality, hidden_size=hidden_units, num_layers=layers, bidirectional=True, batch_first=True)
        self.mha = nn.MultiheadAttention(embed_dim=2 * hidden_units, num_heads=2 * hidden_units // 16, batch_first=True)
    
    def forward(self, X) -> torch.Tensor:        
        out, _ = self.encoder(X)
        out, _ = self.mha(out, out, out)
        
        return out

class QModel(nn.Module):
    def __init__(self, recommend_count: int, dimensionality: int, hidden_units: int, layers: int):
        super().__init__()
        
        self.recommend_count = recommend_count
        self.hidden_units = hidden_units
        self.block = QBlock(recommend_count=recommend_count, dimensionality=dimensionality, hidden_units=hidden_units, layers=layers)
        self.fc = nn.Linear(in_features=2 * hidden_units, out_features=hidden_units)
        self.fc2 = nn.Linear(in_features=hidden_units, out_features=1)
    
    # Model przyjmuje stan (X_set) i recommend_count akcji (X_vec) i ma przewidzieć discounted future reward
    def forward(self, X_set, X_vec) -> torch.Tensor:
        # X_vec ma kształt [batch_size, recommend count, dimensionality]
        X_vec = torch.flatten(X_vec, start_dim=-2, end_dim=-1)
        
        # Teraz kształt X_vec to [batch_size, recommend_count * dimensionality]
        
        X_vec = X_vec.unsqueeze(dim=1)
        X_vec = X_vec.expand(-1, X_set.shape[1], -1)
        X = torch.cat((X_set,X_vec),dim=-1)
        
        X = self.block(X)
        X = torch.relu(X)
        X = self.fc(X)
        
        X = torch.mean(X, dim=1)
        X = torch.relu(X)
        X = self.fc2(X)
        
        X = X.squeeze(dim=-1)
                
        return X

class PolicyBlock(nn.Module):
    def __init__(self, dimensionality: int, hidden_units: int, layers: int):
        super().__init__()
        
        self.dimensionality = dimensionality
        
        self.encoder = nn.GRU(input_size=dimensionality, hidden_size=hidden_units, num_layers=layers, bidirectional=True, batch_first=True)
        self.mha = nn.MultiheadAttention(embed_dim=2 * hidden_units, num_heads=2 * hidden_units // 16, batch_first=True)
    
    def forward(self, X) -> torch.Tensor:        
        out, _ = self.encoder(X)
        out, _ = self.mha(out, out, out)
        
        return out

class Policy(nn.Module):
    def __init__(self, recommend_count: int, dimensionality: int, hidden_units: int, layers: int):
        super().__init__()
        
        self.recommend_count = recommend_count
        self.hidden_units = hidden_units
        self.block = PolicyBlock(dimensionality=dimensionality, hidden_units=hidden_units, layers=layers)
        self.fc = nn.Linear(in_features=2 * hidden_units, out_features=hidden_units)
        self.fc2 = nn.Linear(in_features=hidden_units, out_features=2 * recommend_count * dimensionality)
    
    @classmethod
    def sample(
        cls,
        model_output: torch.Tensor,
        dimensionality: int,
        minimum: float = -2.5,
        maximum: float = 2.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, total_logits = model_output.shape
        total_action_dim = total_logits // 2
        assert total_logits == 2 * total_action_dim, (
            f"model_output width must be even, got {total_logits}"
        )
        assert (
            total_action_dim % dimensionality == 0
        ), (
            f"total_action_dim ({total_action_dim}) must be divisible by dimensionality ({dimensionality})"
        )
        num_actions = total_action_dim // dimensionality

        mean = model_output[:, :total_action_dim]
        log_std = model_output[:, total_action_dim:]

        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)

        eps = torch.randn_like(mean)
        u = mean + std * eps

        scale = (maximum - minimum) / 2.0
        bias = (maximum + minimum) / 2.0
        tanh_u = torch.tanh(u)
        action = tanh_u * scale + bias

        normal = torch.distributions.Normal(mean, std)
        logp_u = normal.log_prob(u).mean(dim=1)

        log_det_jacobian = torch.log(scale * (1 - tanh_u.pow(2)) + 1e-6)
        log_det_jacobian = log_det_jacobian.mean(dim=1)
        
        logp = logp_u - log_det_jacobian

        actions = action.view(batch_size, num_actions, dimensionality)
        return actions, logp

    
    # Model przyjmuje stan i zwraca średnie i wariacje dystrybucji normalnych
    def forward(self, X_set) -> torch.Tensor:
        X = self.block(X_set)
        X = torch.relu(X)
        X = self.fc(X)
        X = torch.mean(X, dim=1)
        X = torch.relu(X)
        X = self.fc2(X)
        
        X = X.squeeze(dim=-1)
        
        return X
