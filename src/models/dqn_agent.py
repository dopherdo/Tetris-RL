"""
DQN Agent with Double DQN and Prioritized Experience Replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dqn_network import DQNNetwork


@dataclass
class DQNConfig:
    """Configuration for DQN agent."""
    gamma: float = 0.99  # Discount factor
    learning_rate: float = 1e-4  # Adam optimizer learning rate
    buffer_size: int = 100_000  # Replay buffer capacity
    batch_size: int = 64  # Minibatch size for training
    target_update_freq: int = 1_000  # Steps between target network updates
    epsilon_start: float = 1.0  # Initial exploration rate
    epsilon_end: float = 0.01  # Minimum exploration rate
    epsilon_decay_steps: int = 10_000  # Steps to decay epsilon
    per_alpha: float = 0.6  # Priority exponent (0 = uniform, 1 = full priority)
    per_beta_start: float = 0.4  # Importance sampling weight (annealed to 1.0)
    per_beta_frames: int = 100_000  # Steps to anneal beta to 1.0
    double_dqn: bool = True  # Use Double DQN
    max_grad_norm: float = 10.0  # Gradient clipping threshold


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Stores transitions with priorities based on TD-error.
    Samples transitions proportional to their priority for efficient learning.
    """

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: Priority exponent (0 = uniform, 1 = full priority)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer with maximum priority."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """
        Sample a batch of transitions based on priorities.
        
        Args:
            batch_size: Number of transitions to sample
            beta: Importance sampling weight (0 = no correction, 1 = full correction)
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer ({self.size} < {batch_size})")

        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        # Extract transitions
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    """
    DQN Agent with Double DQN and Prioritized Experience Replay.
    
    Features:
    - Fixed Q-Targets: Separate target network for stable learning
    - Double DQN: Reduces Q-value overestimation
    - Prioritized Experience Replay: Samples important transitions
    - Epsilon-greedy exploration with decay
    """

    def __init__(
        self,
        board_shape: Tuple[int, int],
        n_actions: int,
        device: torch.device | None = None,
        config: DQNConfig | None = None,
    ) -> None:
        """
        Initialize DQN agent.
        
        Args:
            board_shape: Shape of the Tetris board (height, width)
            n_actions: Number of discrete actions
            device: PyTorch device (CPU or CUDA)
            config: DQN configuration
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or DQNConfig()
        self.n_actions = n_actions

        # Q-networks
        self.q_network = DQNNetwork(board_shape, n_actions).to(self.device)
        self.target_network = DQNNetwork(board_shape, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)

        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_size,
            alpha=self.config.per_alpha,
        )

        # Tracking
        self.steps = 0
        self.epsilon = self.config.epsilon_start

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current board state
            eval_mode: If True, always select greedy action (no exploration)
        
        Returns:
            Selected action index
        """
        if not eval_mode and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)

    def train_step(self) -> dict:
        """
        Perform one training step.
        
        Returns:
            Dictionary with training metrics (loss, Q-values, etc.)
        """
        if len(self.buffer) < self.config.batch_size:
            return {}

        # Anneal beta for importance sampling
        beta = min(
            1.0,
            self.config.per_beta_start
            + (1.0 - self.config.per_beta_start) * self.steps / self.config.per_beta_frames,
        )

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(
            self.config.batch_size, beta
        )

        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Compute current Q-values
        q_values = self.q_network(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: select action with online network, evaluate with target network
                next_actions = self.q_network(next_states_t).argmax(dim=1)
                next_q_values = self.target_network(next_states_t).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                # Standard DQN: select and evaluate with target network
                next_q_values = self.target_network(next_states_t).max(dim=1)[0]

            # Compute target Q-values
            target_q_values = rewards_t + self.config.gamma * next_q_values * (1 - dones_t)

        # Compute TD-errors for priority updates
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors + 1e-6)

        # Compute weighted loss (importance sampling)
        loss = (weights_t * nn.functional.mse_loss(q_values, target_q_values, reduction="none")).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        # Update target network
        if self.steps % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start
            - (self.config.epsilon_start - self.config.epsilon_end)
            * self.steps
            / self.config.epsilon_decay_steps,
        )

        self.steps += 1

        return {
            "loss": float(loss.item()),
            "q_mean": float(q_values.mean().item()),
            "q_max": float(q_values.max().item()),
            "epsilon": self.epsilon,
            "beta": beta,
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint.get("steps", 0)
        self.epsilon = checkpoint.get("epsilon", self.config.epsilon_end)

