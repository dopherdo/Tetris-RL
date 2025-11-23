"""
Experience Replay Buffer for DQN
Stores and samples transitions for off-policy learning
"""
import numpy as np
import torch
import random


class ReplayBuffer:
    """
    Experience replay buffer for DQN
    
    Stores: (state, action, reward, next_state, done)
    
    Key advantage over PPO's rollout buffer:
    - Stores experiences long-term (100K+ transitions)
    - Can sample rare good experiences repeatedly
    - Perfect for sparse rewards like line clears!
    """
    
    def __init__(self, capacity, state_shape, device='cpu'):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
            state_shape: Shape of state (C, H, W)
            device: PyTorch device
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.device = device
        
        # Storage
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.pos = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        """
        Add transition to buffer
        
        Args:
            state: Current state (C, H, W)
            action: Action taken
            reward: Reward received  
            next_state: Next state (C, H, W)
            done: Episode done flag
        """
        # Convert to numpy
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().item()
        
        # Store
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Sample random batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Dictionary with batched tensors
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'states': torch.from_numpy(self.states[indices]).to(self.device),
            'actions': torch.from_numpy(self.actions[indices]).to(self.device),
            'rewards': torch.from_numpy(self.rewards[indices]).to(self.device),
            'next_states': torch.from_numpy(self.next_states[indices]).to(self.device),
            'dones': torch.from_numpy(self.dones[indices]).to(self.device)
        }
    
    def __len__(self):
        """Return current buffer size"""
        return self.size


if __name__ == "__main__":
    # Test replay buffer
    print("Testing ReplayBuffer...")
    print("=" * 70)
    
    buffer = ReplayBuffer(capacity=10000, state_shape=(2, 20, 10), device='cpu')
    
    # Add experiences
    for i in range(100):
        state = np.random.randn(2, 20, 10)
        action = np.random.randint(0, 40)
        reward = np.random.randn()
        next_state = np.random.randn(2, 20, 10)
        done = (i % 10 == 9)
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"✓ Added 100 transitions")
    print(f"  Buffer size: {len(buffer)}")
    print()
    
    # Sample batch
    batch = buffer.sample(32)
    
    print(f"✓ Sampled batch of 32:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    print()
    print("✓ ReplayBuffer working!")


