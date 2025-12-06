"""
CNN-based Q-Network for DQN on Tetris.

This network takes the board state as input and outputs Q-values for all possible actions.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Convolutional Q-Network for Tetris."""

    def __init__(self, board_shape: Tuple[int, int], n_actions: int) -> None:
        super().__init__()
        h, w = board_shape
        
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

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        self.q_head = nn.Linear(256, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Q-network."""
        x = x.float()
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        features = self.fc(self.conv(x))
        q_values = self.q_head(features)
        return q_values


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN architecture separating state value and advantage."""

    def __init__(self, board_shape: Tuple[int, int], n_actions: int) -> None:
        super().__init__()
        h, w = board_shape
        
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

        self.shared_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(inplace=True),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dueling architecture."""
        x = x.float()
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        features = self.shared_fc(self.conv(x))
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

