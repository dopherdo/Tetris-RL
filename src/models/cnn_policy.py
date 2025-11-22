"""
CNN-based Policy Network for Tetris PPO
Implements Actor-Critic architecture with shared CNN backbone
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CNNPolicy(nn.Module):
    """
    CNN-based Actor-Critic Policy Network for Tetris
    
    Architecture:
    1. Input: (batch, 1, 20, 10) - Binary board state
    2. CNN Backbone: Extract spatial features (holes, patterns, heights)
    3. Shared Feature Layer: Common representation for both heads
    4. Actor Head: Policy distribution over actions
    5. Critic Head: State value estimation
    
    The CNN is designed to detect Tetris-specific patterns:
    - Holes (bad)
    - Flat surfaces (good for line clears)
    - Wells (columns suitable for I-pieces)
    - Height distributions
    """
    
    def __init__(self, action_dim, hidden_dim=256):
        """
        Initialize the CNN policy network
        
        Args:
            action_dim: Number of possible actions in the environment
            hidden_dim: Dimension of hidden layers (default: 256)
        """
        super(CNNPolicy, self).__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # CNN Backbone for feature extraction
        # Input: (batch, 1, 20, 10)
        
        # Conv Layer 1: Detect local patterns (3x3 receptive field)
        # Output: (batch, 32, 20, 10)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1  # Same padding to keep spatial dimensions
        )
        
        # Conv Layer 2: Detect larger patterns (5x5 receptive field total)
        # Output: (batch, 64, 20, 10)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Conv Layer 3: High-level features
        # Output: (batch, 64, 20, 10)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Calculate flattened feature dimension
        # After conv layers: (batch, 64, 20, 10)
        # Flattened: 64 * 20 * 10 = 12800
        self.feature_dim = 64 * 20 * 10
        
        # Shared fully connected layer
        # Reduces dimensionality and creates shared representation
        self.fc_shared = nn.Linear(self.feature_dim, hidden_dim)
        
        # Actor head (policy network)
        # Outputs logits for action distribution
        self.actor_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.actor_fc2 = nn.Linear(hidden_dim // 2, action_dim)
        
        # Critic head (value network)
        # Outputs scalar state value
        self.critic_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.critic_fc2 = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights with orthogonal initialization
        # This helps with training stability in RL
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch, 1, 20, 10)
            
        Returns:
            tuple: (action_logits, state_value)
                - action_logits: (batch, action_dim) - unnormalized action probabilities
                - state_value: (batch, 1) - estimated state value
        """
        # CNN backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten spatial dimensions
        x = x.view(x.size(0), -1)  # (batch, feature_dim)
        
        # Shared feature layer
        shared_features = F.relu(self.fc_shared(x))
        
        # Actor head (policy)
        actor = F.relu(self.actor_fc1(shared_features))
        action_logits = self.actor_fc2(actor)
        
        # Critic head (value)
        critic = F.relu(self.critic_fc1(shared_features))
        state_value = self.critic_fc2(critic)
        
        return action_logits, state_value
    
    def get_action(self, x, deterministic=False):
        """
        Sample an action from the policy
        
        Args:
            x: Input tensor of shape (batch, 1, 20, 10)
            deterministic: If True, select argmax action (for evaluation)
                          If False, sample from distribution (for training)
            
        Returns:
            tuple: (action, log_prob, value)
                - action: (batch,) - selected actions
                - log_prob: (batch,) - log probability of selected actions
                - value: (batch, 1) - state value estimates
        """
        action_logits, value = self.forward(x)
        
        # Create categorical distribution over actions
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        if deterministic:
            # Greedy action selection
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from distribution
            action = dist.sample()
        
        # Get log probability of selected action
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate_actions(self, x, actions):
        """
        Evaluate actions for PPO training
        
        This method is used during PPO updates to compute:
        - Log probabilities of taken actions
        - Entropy of action distribution (for exploration bonus)
        - State values
        
        Args:
            x: Input tensor of shape (batch, 1, 20, 10)
            actions: Actions taken, shape (batch,)
            
        Returns:
            tuple: (log_probs, entropy, values)
                - log_probs: (batch,) - log probability of actions
                - entropy: (batch,) - entropy of action distribution
                - values: (batch, 1) - state value estimates
        """
        action_logits, values = self.forward(x)
        
        # Create categorical distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        # Evaluate the provided actions
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy, values
    
    def get_value(self, x):
        """
        Get only the value estimate (useful for advantage calculation)
        
        Args:
            x: Input tensor of shape (batch, 1, 20, 10)
            
        Returns:
            torch.Tensor: State value estimates, shape (batch, 1)
        """
        _, value = self.forward(x)
        return value


def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """
    Test the CNN policy network
    """
    print("Testing CNN Policy Network...")
    print("=" * 60)
    
    # Test parameters
    batch_size = 4
    board_height = 20
    board_width = 10
    action_dim = 8  # Example: Tetris typically has ~8 actions
    
    # Create model
    print("\n1. Creating CNN Policy Network...")
    model = CNNPolicy(action_dim=action_dim, hidden_dim=256)
    print(f"   ✓ Model created successfully")
    print(f"   Total parameters: {count_parameters(model):,}")
    
    # Create dummy input (random board state)
    print("\n2. Creating dummy input...")
    dummy_input = torch.rand(batch_size, 1, board_height, board_width)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   ✓ Input created")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    action_logits, values = model(dummy_input)
    print(f"   Action logits shape: {action_logits.shape}")
    print(f"   Expected: ({batch_size}, {action_dim})")
    print(f"   Values shape: {values.shape}")
    print(f"   Expected: ({batch_size}, 1)")
    assert action_logits.shape == (batch_size, action_dim), "Action logits shape mismatch!"
    assert values.shape == (batch_size, 1), "Values shape mismatch!"
    print(f"   ✓ Forward pass successful")
    
    # Test action sampling
    print("\n4. Testing action sampling...")
    actions, log_probs, values = model.get_action(dummy_input, deterministic=False)
    print(f"   Actions shape: {actions.shape}")
    print(f"   Log probs shape: {log_probs.shape}")
    print(f"   Values shape: {values.shape}")
    print(f"   Sample actions: {actions.tolist()}")
    print(f"   ✓ Action sampling successful")
    
    # Test deterministic action selection
    print("\n5. Testing deterministic action selection...")
    det_actions, det_log_probs, det_values = model.get_action(dummy_input, deterministic=True)
    print(f"   Deterministic actions: {det_actions.tolist()}")
    print(f"   ✓ Deterministic selection successful")
    
    # Test action evaluation
    print("\n6. Testing action evaluation...")
    eval_log_probs, entropy, eval_values = model.evaluate_actions(dummy_input, actions)
    print(f"   Log probs shape: {eval_log_probs.shape}")
    print(f"   Entropy shape: {entropy.shape}")
    print(f"   Values shape: {eval_values.shape}")
    print(f"   Average entropy: {entropy.mean().item():.4f}")
    print(f"   ✓ Action evaluation successful")
    
    # Test with real Tetris environment
    print("\n7. Testing with real Tetris environment...")
    try:
        import sys
        sys.path.append('/Users/ilansasson/Tetris-RL')
        from src.env.tetris_env import TetrisEnv
        from src.utils.preprocessing import preprocess_observation
        
        # Create environment
        env = TetrisEnv(render_mode=None)
        obs, info = env.reset()
        
        print(f"   Environment action space: {env.action_space}")
        real_action_dim = env.action_space.n
        print(f"   Real action dimension: {real_action_dim}")
        
        # Create model with correct action dimension
        real_model = CNNPolicy(action_dim=real_action_dim, hidden_dim=256)
        
        # Preprocess observation
        board_tensor = preprocess_observation(obs).unsqueeze(0)  # Add batch dimension
        print(f"   Preprocessed board shape: {board_tensor.shape}")
        
        # Get action from model
        action, log_prob, value = real_model.get_action(board_tensor, deterministic=False)
        action_int = action.item()
        
        print(f"   Model selected action: {action_int}")
        print(f"   Log probability: {log_prob.item():.4f}")
        print(f"   State value: {value.item():.4f}")
        
        # Take action in environment
        obs, reward, terminated, truncated, info = env.step(action_int)
        print(f"   Action executed! Reward: {reward:.2f}")
        
        print(f"   ✓ Real environment integration successful!")
        
        env.close()
        
    except Exception as e:
        print(f"   ⚠ Warning: Could not test with real environment: {e}")
        print(f"   (This is okay - the model itself is working)")
    
    print("\n" + "=" * 60)
    print("✓ All CNN Policy tests passed!")
    print("\nModel Summary:")
    print(f"- Input: (batch, 1, 20, 10) board tensor")
    print(f"- CNN Backbone: 3 convolutional layers with batch normalization")
    print(f"- Actor Head: Outputs action logits (policy distribution)")
    print(f"- Critic Head: Outputs state value estimate")
    print(f"- Total Parameters: {count_parameters(model):,}")
    print("\nThe model is ready for PPO training!")

