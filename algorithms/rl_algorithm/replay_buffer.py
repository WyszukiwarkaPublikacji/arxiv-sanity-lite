from collections import deque
import random
import torch
import numpy as np

# Klasa, która przechowuje informacje o stanie, akcji, nagrodzie, kolejnym stanie i czy stan jest ostatnim stanem
class ReplayBufferState:
    def __init__(self, state: torch.Tensor, action: torch.Tensor, reward: float, next_state: torch.Tensor, done_signal: int):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done_signal = done_signal

# Implementacja klasy do przechowywania informacji o podjętych akcjach i ich skutkach
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done_signal):
        state = state.cpu()
        action = action.cpu()
        next_state = next_state.cpu()
        
        experience = ReplayBufferState(state, action, reward, next_state, done_signal)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states = torch.tensor(np.array([e.state for e in batch]))
        actions = torch.tensor(np.array([e.action for e in batch]))
        rewards = torch.tensor([e.reward for e in batch])
        next_states = torch.tensor(np.array([e.next_state for e in batch]))
        done_signals = torch.tensor([e.done_signal for e in batch])

        return states, actions, rewards, next_states, done_signals

    def __len__(self):
        return len(self.buffer)
