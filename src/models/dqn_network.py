"""
DQN Network for Tetris
Outputs Q-values for each action
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TetrisDQN(nn.Module):
    """
    Deep Q-Network for Tetris
    
    Input: (batch, 2, 20, 10) - board state
    Output: (batch, 40) - Q-value for each action
    
    Simpler than PPO Actor-Critic (no policy head, just Q-values)
    """
    
    def __init__(self, in_channels=2, num_actions=40):
        """
        Initialize DQN
        
        Args:
            in_channels: Number of input channels
            num_actions: Number of actions
        """
        super(TetrisDQN, self).__init__()
        
        # CNN Feature Extractor
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Flatten: 64 * 20 * 10 = 12,800
        self.flatten_size = 64 * 20 * 10
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, num_actions)  # Output Q-values
        
        self.num_actions = num_actions
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: State tensor (batch, 2, 20, 10)
        
        Returns:
            Q-values for each action (batch, 40)
        """
        # Conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # FC layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values


if __name__ == "__main__":
    # Test network
    print("Testing DQN Network...")
    print("=" * 70)
    
    model = TetrisDQN(in_channels=2, num_actions=40)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()
    
    # Test forward pass
    dummy_input = torch.randn(4, 2, 20, 10)
    q_values = model(dummy_input)
    
    print(f"Forward pass:")
    print(f"  Input: {dummy_input.shape}")
    print(f"  Q-values: {q_values.shape} (should be [4, 40])")
    print(f"  Q-value range: [{q_values.min():.2f}, {q_values.max():.2f}]")
    print()
    
    # Test action selection
    best_actions = torch.argmax(q_values, dim=1)
    print(f"Action selection (argmax):")
    print(f"  Best actions: {best_actions}")
    print()
    
    print("✓ DQN Network working!")


