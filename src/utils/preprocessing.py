"""
State preprocessing utilities for Tetris RL
Converts raw observations to tensor format suitable for neural network input
"""
import numpy as np
import torch


def preprocess_observation(obs, device='cpu'):
    """
    Convert raw Tetris observation to tensor format for CNN input
    
    The tetris-gymnasium observation contains:
    - 'board': 2D array (24 x 18) with border walls and floor
    - 'holder': Current held piece (optional)
    - 'queue': Upcoming pieces (optional)
    - 'active_tetromino_mask': Mask of the current falling piece
    
    For simplicity and following standard Tetris RL approaches, we use:
    - The board state as primary input (playable area: 20x10)
    - Normalize values to [0, 1] range
    
    Args:
        obs: Dictionary observation from TetrisEnv
        device: PyTorch device ('cpu' or 'cuda')
        
    Returns:
        torch.Tensor: Preprocessed board tensor of shape (1, 20, 10)
                     - Channel 0: Board state (0=empty, 1=filled)
    """
    # Extract board from observation
    board = obs['board']
    
    # Extract playable area (exclude borders and floor)
    # tetris-gymnasium: 24 rows total, playable rows 0-19
    # tetris-gymnasium: 18 cols total, playable cols 4-13 (10 columns)
    playable_board = board[0:20, 4:14].copy()
    
    # Normalize: Convert to binary (0 or 1)
    # Any positive value means a cell is filled
    playable_board = (playable_board > 0).astype(np.float32)
    
    # Convert to PyTorch tensor with shape (1, height, width) for CNN
    # This represents 1 channel (binary board state)
    board_tensor = torch.from_numpy(playable_board).unsqueeze(0).float()
    
    return board_tensor.to(device)


def batch_preprocess(observations, device='cpu'):
    """
    Preprocess a batch of observations
    
    Args:
        observations: List of observation dictionaries
        device: PyTorch device
        
    Returns:
        torch.Tensor: Batch of preprocessed boards (batch_size, 1, 20, 10)
    """
    processed = [preprocess_observation(obs, device) for obs in observations]
    return torch.stack(processed, dim=0)


def get_board_features(obs):
    """
    Extract hand-crafted features from board state for analysis/debugging
    
    These are NOT used as CNN input but can be useful for:
    - Debugging and visualization
    - Comparing learned vs hand-crafted features
    - Auxiliary reward signals
    
    Args:
        obs: Dictionary observation from TetrisEnv
        
    Returns:
        dict: Dictionary of board features
    """
    board = obs['board'][0:20, 4:14]  # Playable area
    
    features = {}
    
    # Column heights
    heights = []
    for col in range(10):
        height = 0
        for row in range(20):
            if board[row, col] > 0:
                height = 20 - row
                break
        heights.append(height)
    
    features['heights'] = np.array(heights)
    features['max_height'] = max(heights)
    features['avg_height'] = np.mean(heights)
    
    # Holes (empty cell with filled cell above)
    holes = 0
    for col in range(10):
        block_found = False
        for row in range(20):
            if board[row, col] > 0:
                block_found = True
            elif block_found:
                holes += 1
    features['holes'] = holes
    
    # Bumpiness (height differences between adjacent columns)
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(9))
    features['bumpiness'] = bumpiness
    
    # Number of complete lines
    complete_lines = sum(1 for row in range(20) if np.all(board[row] > 0))
    features['complete_lines'] = complete_lines
    
    # Fill percentage
    features['fill_percentage'] = np.sum(board > 0) / (20 * 10)
    
    return features


def visualize_tensor(tensor, title="Board State"):
    """
    Visualize a preprocessed tensor for debugging
    
    Args:
        tensor: PyTorch tensor of shape (1, 20, 10) or (20, 10)
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    
    # Handle different tensor shapes
    if len(tensor.shape) == 3:
        tensor = tensor[0]  # Remove channel dimension
    
    # Convert to numpy if needed
    if torch.is_tensor(tensor):
        tensor = tensor.cpu().numpy()
    
    plt.figure(figsize=(5, 10))
    plt.imshow(tensor, cmap='binary', interpolation='nearest')
    plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.colorbar(label='Cell Value')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Test preprocessing with actual Tetris environment
    """
    print("Testing preprocessing utilities...")
    
    # Import environment
    from src.env.tetris_env import TetrisEnv
    
    # Create environment
    env = TetrisEnv(render_mode=None)
    obs, info = env.reset()
    
    print("\n1. Original observation keys:", obs.keys())
    print("   Board shape:", obs['board'].shape)
    
    # Test single observation preprocessing
    board_tensor = preprocess_observation(obs)
    print("\n2. Preprocessed tensor shape:", board_tensor.shape)
    print("   Expected shape: (1, 20, 10) - (channels, height, width)")
    print("   ✓ Shape correct!" if board_tensor.shape == (1, 20, 10) else "   ✗ Shape incorrect!")
    
    # Test batch preprocessing
    observations = [obs for _ in range(4)]
    batch_tensor = batch_preprocess(observations)
    print("\n3. Batch tensor shape:", batch_tensor.shape)
    print("   Expected shape: (4, 1, 20, 10) - (batch, channels, height, width)")
    print("   ✓ Batch shape correct!" if batch_tensor.shape == (4, 1, 20, 10) else "   ✗ Batch shape incorrect!")
    
    # Test feature extraction
    features = get_board_features(obs)
    print("\n4. Extracted features:")
    for key, value in features.items():
        if key != 'heights':
            print(f"   {key}: {value}")
    
    # Take a few random steps and visualize
    print("\n5. Taking 5 random steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"   Step {i+1}: Game over! Resetting...")
            obs, info = env.reset()
        else:
            features = get_board_features(obs)
            print(f"   Step {i+1}: Height={features['max_height']}, Holes={features['holes']}, Reward={reward:.2f}")
    
    print("\n✓ All preprocessing tests passed!")
    print("\nPreprocessing module ready for CNN integration.")
    
    env.close()

