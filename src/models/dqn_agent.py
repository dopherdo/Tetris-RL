"""
Double DQN Agent for Tetris
Implements the three key techniques from Deep Q-Learning:
1. Experience Replay - efficient use of experiences
2. Fixed Q-Target - stable training with target network
3. Double DQN - reduces Q-value overestimation bias

Much better than PPO for sparse rewards + experience replay
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from src.models.dqn_network import TetrisDQN
from src.utils.prioritized_replay_buffer import PrioritizedReplayBuffer


class DQNAgent:
    """
    Double DQN Agent with experience replay and fixed Q-target
    
    Implements three key techniques for stable Deep Q-Learning:
    1. Experience Replay - efficient use of experiences, learn from rare events
    2. Fixed Q-Target - target network updated periodically for stable training
    3. Double DQN - decouples action selection from evaluation to reduce overestimation
    
    Key advantages over PPO for Tetris:
    1. Experience replay - learn from line clears repeatedly
    2. Off-policy - more sample efficient
    3. Simpler - just predict Q-values, no policy distribution
    4. Double DQN - more accurate Q-value estimates, better performance
    """
    
    def __init__(
        self,
        state_shape=(2, 20, 10),
        num_actions=40,
        device='cpu',
        learning_rate=1e-4,
        gamma=0.999,  # High discount for long-term rewards (handles delayed credit assignment)
        epsilon_start=1.0,
        epsilon_end=0.05,  # Increased from 0.01 to maintain exploration
        epsilon_decay=0.99995,  # Slower decay
        batch_size=256,  # Increased from 128
        buffer_capacity=100000,
        target_update_freq=750,  # Updated to 750 steps (sweet spot)
        learning_starts=1000,  # Start learning earlier
        use_per=True,  # Use Prioritized Experience Replay
        per_alpha=0.6,  # Priority exponent
        per_beta=0.4,  # Importance sampling exponent
        per_beta_increment=1e-6  # Beta annealing rate
    ):
        """
        Initialize DQN agent
        
        Args:
            state_shape: Shape of state
            num_actions: Number of actions
            device: PyTorch device
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate per step
            batch_size: Batch size for training
            buffer_capacity: Replay buffer size
            target_update_freq: Update target network every N steps
            learning_starts: Start training after N steps
        """
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        
        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.q_network = TetrisDQN(in_channels=state_shape[0], num_actions=num_actions).to(device)
        self.target_network = TetrisDQN(in_channels=state_shape[0], num_actions=num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer (learning rate can be scheduled)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer (Prioritized Experience Replay)
        self.use_per = use_per
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_capacity, state_shape, device,
                alpha=per_alpha, beta=per_beta, beta_increment=per_beta_increment
            )
        else:
            from src.utils.replay_buffer import ReplayBuffer
            self.replay_buffer = ReplayBuffer(buffer_capacity, state_shape, device)
        
        # Tracking
        self.total_steps = 0
        self.training_steps = 0
        self.last_td_errors = None  # Store TD errors for PER priority updates
    
    def select_action(self, state, legal_actions_mask=None, eval_mode=False):
        """
        Select action using epsilon-greedy policy with action masking
        
        Args:
            state: State tensor (C, H, W) or (B, C, H, W)
            legal_actions_mask: Boolean mask of legal actions (shape: (num_actions,))
                               If None, all actions are legal
            eval_mode: If True, use greedy policy (no exploration)
        
        Returns:
            action: Selected action (int)
        """
        # Add batch dimension if needed
        if state.dim() == 3:
            state = state.unsqueeze(0)
        
        # Epsilon-greedy
        if not eval_mode and random.random() < self.epsilon:
            # Explore: random action from legal actions only
            if legal_actions_mask is not None:
                legal_indices = np.where(legal_actions_mask)[0]
                if len(legal_indices) > 0:
                    return np.random.choice(legal_indices)
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: best action from legal actions only
            with torch.no_grad():
                q_values = self.q_network(state)
                
                # Mask out illegal actions by setting Q-values to -inf
                if legal_actions_mask is not None:
                    mask_tensor = torch.from_numpy(legal_actions_mask).float().to(self.device)
                    q_values = q_values.clone()
                    q_values[0, ~legal_actions_mask] = float('-inf')
                
                action = torch.argmax(q_values, dim=1).item()
            return action
    
    def store_transition(self, state, action, reward, next_state, done, td_error=None):
        """
        Store transition in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
            td_error: TD error for PER priority (if None, uses max priority)
        """
        if self.use_per:
            self.replay_buffer.add(state, action, reward, next_state, done, td_error)
        else:
            self.replay_buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1
    
    def train_step(self):
        """
        Perform one training step (if enough data collected)
        
        Returns:
            loss value or None if not training yet
        """
        # Don't train until we have enough experiences
        if len(self.replay_buffer) < self.learning_starts:
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Compute current Q-values
        current_q_values = self.q_network(batch['states'])
        current_q = current_q_values.gather(1, batch['actions'].unsqueeze(1)).squeeze()
        
        # Compute target Q-values using Double DQN
        # Double DQN: Main network selects action, target network evaluates it
        # This decouples selection from evaluation, reducing overestimation bias
        with torch.no_grad():
            # Main network selects the best action
            next_q_values_main = self.q_network(batch['next_states'])
            best_actions = next_q_values_main.argmax(dim=1)
            
            # Target network evaluates the selected action
            next_q_values_target = self.target_network(batch['next_states'])
            max_next_q = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze()
            
            target_q = batch['rewards'] + self.gamma * max_next_q * (1 - batch['dones'])
        
        # Compute TD errors for PER priority updates
        td_errors = target_q - current_q
        
        # Compute loss (Huber loss is more stable than MSE)
        # Apply importance sampling weights if using PER
        if self.use_per and 'weights' in batch:
            weights = batch['weights']
            loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
            loss = (weights * loss).mean()
        else:
            loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities in PER buffer
        if self.use_per and 'indices' in batch:
            with torch.no_grad():
                td_errors_abs = torch.abs(td_errors)
            self.replay_buffer.update_priorities(batch['indices'], td_errors_abs)
        
        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Learning rate scheduling (based on blurb: decay LR over time for stability)
        # Decay LR every 20K steps: 1e-4 → 5e-5 → 2.5e-5
        # Don't decay below 1e-4 to prevent collapse
        if self.training_steps % 20000 == 0 and self.training_steps > 0:
            current_lr = self.learning_rate
            new_lr = current_lr * 0.5  # Halve learning rate every 20K steps
            new_lr = max(1e-4, new_lr)  # Don't go below 1e-4 (prevent collapse)
            self.set_learning_rate(new_lr)
        
        return loss.item()
    
    def set_learning_rate(self, lr):
        """
        Update learning rate (for scheduling based on exploration/exploitation phases)
        
        Args:
            lr: New learning rate
        """
        self.learning_rate = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def set_epsilon(self, epsilon):
        """
        Set epsilon value (for exploration/exploitation scheduling)
        
        Args:
            epsilon: New epsilon value
        """
        self.epsilon = max(self.epsilon_end, min(1.0, epsilon))
    
    def save(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'training_steps': self.training_steps
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        self.training_steps = checkpoint['training_steps']


if __name__ == "__main__":
    # Test DQN agent
    print("Testing DQN Agent...")
    print("=" * 70)
    
    from src.env.tetris_env import TetrisEnv
    from src.utils.preprocessing import TetrisPreprocessor
    
    agent = DQNAgent(
        state_shape=(2, 20, 10),
        num_actions=40,
        device='cpu',
        buffer_capacity=10000
    )
    
    print(f"✓ Agent created:")
    print(f"  Q-network parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print(f"  Replay buffer capacity: {agent.replay_buffer.capacity:,}")
    print(f"  Initial epsilon: {agent.epsilon}")
    print()
    
    # Test with environment
    env = TetrisEnv(render_mode=None)
    preprocessor = TetrisPreprocessor(use_active_piece=True, device='cpu')
    
    print("Collecting experiences...")
    obs, _ = env.reset()
    
    for i in range(20):
        state = preprocessor(obs)
        action = agent.select_action(state)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = preprocessor(next_obs)
        done = terminated or truncated
        
        agent.store_transition(state, action, reward, next_state, done)
        
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
    
    print(f"✓ Stored {len(agent.replay_buffer)} transitions")
    print()
    
    print("✓ DQN Agent working!")
    env.close()

