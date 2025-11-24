"""
Prioritized Experience Replay Buffer for DQN
Samples experiences based on TD-error priority (high TD-error = more to learn)
"""
import numpy as np
import torch
import heapq
import random


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    Uses heap-based structure to prioritize experiences with high TD-error.
    High TD-error means the agent's prediction was far off, so there's more to learn.
    
    Based on the successful implementation that achieved 500+ lines.
    """
    
    def __init__(self, capacity, state_shape, device='cpu', alpha=0.6, beta=0.4, beta_increment=1e-6):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
            state_shape: Shape of state (C, H, W)
            device: PyTorch device
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent (starts at beta, anneals to 1.0)
            beta_increment: How much to increment beta per sample
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.beta_start = beta
        self.beta_increment = beta_increment
        self.max_beta = 1.0
        
        # Storage: list of (priority, counter, experience)
        # Using negative priority for min-heap (we want max priority)
        self.memory = []
        self.counter = 0
        
        # For efficient sampling, we'll also maintain arrays
        # This allows faster batch operations
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0  # Initial priority for new experiences
    
    def add(self, state, action, reward, next_state, done, td_error=None):
        """
        Add transition to buffer with priority
        
        Args:
            state: Current state (C, H, W)
            action: Action taken
            reward: Reward received
            next_state: Next state (C, H, W)
            done: Episode done flag
            td_error: TD error for priority (if None, uses max_priority)
        """
        # Convert to numpy
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().item()
        
        # Calculate priority from TD error
        if td_error is None:
            priority = self.max_priority
        else:
            if isinstance(td_error, torch.Tensor):
                td_error = td_error.detach().cpu().numpy().item()
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
        
        # Store in arrays
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.priorities[self.pos] = priority
        
        # Also store in heap for efficient max-priority retrieval
        experience = (state, action, reward, next_state, done)
        if self.size < self.capacity:
            heapq.heappush(self.memory, (-priority, self.counter, self.pos, experience))
            self.size += 1
        else:
            # Replace oldest experience
            heapq.heappushpop(self.memory, (-priority, self.counter, self.pos, experience))
        
        self.pos = (self.pos + 1) % self.capacity
        self.counter += 1
    
    def sample(self, batch_size):
        """
        Sample batch based on priority distribution
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary with batched tensors and importance sampling weights
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Get priorities for all stored experiences
        priorities = self.priorities[:self.size]
        
        # Compute sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = self._fast_sample(batch_size, probabilities)
        indices = np.clip(indices, 0, self.size - 1)
        
        # Collect batch
        batch = {
            'states': torch.from_numpy(self.states[indices]).to(self.device),
            'actions': torch.from_numpy(self.actions[indices]).to(self.device),
            'rewards': torch.from_numpy(self.rewards[indices]).to(self.device),
            'next_states': torch.from_numpy(self.next_states[indices]).to(self.device),
            'dones': torch.from_numpy(self.dones[indices]).to(self.device),
            'indices': indices,  # For updating priorities later
        }
        
        # Compute importance sampling weights
        weights = (1.0 / self.size / probabilities[indices]) ** self.beta
        weights /= weights.max()  # Normalize by max weight
        batch['weights'] = torch.from_numpy(weights).float().to(self.device)
        
        # Anneal beta towards 1.0
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
        
        return batch
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities for sampled experiences
        
        Args:
            indices: Indices of experiences to update
            td_errors: New TD errors for these experiences
        """
        for idx, td_error in zip(indices, td_errors):
            if isinstance(td_error, torch.Tensor):
                td_error = td_error.detach().cpu().numpy().item()
            
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def _fast_sample(self, batch_size, probabilities):
        """
        Fast sampling using cumulative distribution function
        
        Args:
            batch_size: Number of samples
            probabilities: Probability distribution
            
        Returns:
            Sampled indices
        """
        cumulative_probs = np.cumsum(probabilities)
        random_vals = np.random.rand(batch_size)
        indices = np.searchsorted(cumulative_probs, random_vals)
        return indices
    
    def __len__(self):
        """Return current buffer size"""
        return self.size

