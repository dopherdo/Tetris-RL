"""
CNN-based Q-Network for DQN on Tetris.

This network takes the board state as input and outputs Q-values for all possible actions.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Convolutional Q-Network for Tetris.
    
    Architecture:
    - Input: (B, 1, H, W) board state (binary or integer representation)
    - CNN backbone: Extracts spatial features (holes, heights, patterns)
    - Fully connected layers: Process features
    - Output: Q-values for each action (composite rotation + column placement)
    
    Args:
        board_shape: Tuple of (height, width) for the Tetris board
        n_actions: Number of discrete actions (composite placements)
    """

    def __init__(self, board_shape: Tuple[int, int], n_actions: int) -> None:
        super().__init__()
        h, w = board_shape
        
        # CNN backbone for feature extraction
        # Designed to detect:
        # - Holes and gaps
        # - Column heights
        # - Surface irregularities (bumpiness)
        # - Wells and overhangs
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        # Compute conv output size with dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, 1, h, w)
            conv_out_size = self.conv(dummy).shape[1]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        # Q-value head: outputs Q(s, a) for each action
        self.q_head = nn.Linear(256, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Q-network.
        
        Args:
            x: Board state tensor of shape (B, H, W) or (B, 1, H, W)
        
        Returns:
            Q-values for each action, shape (B, n_actions)
        """
        x = x.float()
        if x.ndim == 3:  # (B, H, W) -> (B, 1, H, W)
            x = x.unsqueeze(1)
        
        features = self.fc(self.conv(x))
        q_values = self.q_head(features)
        return q_values


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture (optional enhancement).
    
    Separates Q-values into:
    - V(s): State value function
    - A(s, a): Advantage function
    
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
    
    This helps learning by explicitly separating state value from action advantages.
    """

    def __init__(self, board_shape: Tuple[int, int], n_actions: int) -> None:
        super().__init__()
        h, w = board_shape
        
        # Shared CNN backbone
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, h, w)
            conv_out_size = self.conv(dummy).shape[1]

        # Shared feature extraction
        self.shared_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(inplace=True),
        )

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dueling architecture.
        
        Args:
            x: Board state tensor of shape (B, H, W) or (B, 1, H, W)
        
        Returns:
            Q-values for each action, shape (B, n_actions)
        """
        x = x.float()
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        features = self.shared_fc(self.conv(x))
        
        # Compute value and advantage
        value = self.value_stream(features)  # (B, 1)
        advantage = self.advantage_stream(features)  # (B, n_actions)
        
        # Combine using mean subtraction for identifiability
        # Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

