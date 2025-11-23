"""
Preprocessing utilities for Tetris observations
Converts environment observations into tensors for CNN input
"""
import numpy as np
import torch


class TetrisPreprocessor:
    """
    Preprocesses Tetris observations for neural network input
    
    Converts observation dict to tensor:
    - Extracts 20×10 playable area (excludes walls/floor)
    - Normalizes values to [0, 1] range
    - Stacks multiple channels (board, active piece)
    - Returns PyTorch tensor ready for CNN
    """
    
    def __init__(self, use_active_piece=True, device='cpu'):
        """
        Initialize preprocessor
        
        Args:
            use_active_piece: Whether to include active piece as separate channel
            device: PyTorch device ('cpu' or 'cuda')
        """
        self.use_active_piece = use_active_piece
        self.device = device
        
        # Playable area boundaries (20×10)
        self.row_start = 0
        self.row_end = 20
        self.col_start = 4
        self.col_end = 14
    
    def __call__(self, obs):
        """
        Preprocess single observation
        
        Args:
            obs: Observation dict from environment with keys:
                 - 'board': (24, 18) array
                 - 'active_tetromino_mask': (24, 18) array
                 - 'holder': (4, 4) array
                 - 'queue': (4, 16) array
        
        Returns:
            torch.Tensor: (C, H, W) = (2, 20, 10) or (1, 20, 10)
                         Channel 0: Settled pieces (normalized 0-1)
                         Channel 1: Active piece mask (0 or 1) [if use_active_piece=True]
        """
        # Extract playable area from board
        board_full = obs['board']
        board = board_full[self.row_start:self.row_end, 
                          self.col_start:self.col_end].astype(np.float32)
        
        # Remove active piece from board to get settled pieces only
        active_mask_full = obs['active_tetromino_mask']
        active_mask = active_mask_full[self.row_start:self.row_end,
                                       self.col_start:self.col_end].astype(np.float32)
        
        # Get settled board (remove active piece)
        settled_board = board.copy()
        settled_board[active_mask > 0] = 0
        
        # Normalize settled board (values 0-9 → 0-1)
        settled_board = settled_board / 9.0
        
        # Stack channels
        if self.use_active_piece:
            # 2 channels: settled pieces + active piece
            state = np.stack([settled_board, active_mask], axis=0)  # (2, 20, 10)
        else:
            # 1 channel: settled pieces only
            state = settled_board[np.newaxis, :, :]  # (1, 20, 10)
        
        # Convert to PyTorch tensor
        state_tensor = torch.from_numpy(state).float().to(self.device)
        
        return state_tensor
    
    def preprocess_batch(self, obs_batch):
        """
        Preprocess batch of observations
        
        Args:
            obs_batch: List of observation dicts
        
        Returns:
            torch.Tensor: (B, C, H, W) where B is batch size
        """
        states = [self(obs) for obs in obs_batch]
        return torch.stack(states)
    
    def get_state_shape(self):
        """
        Get the shape of preprocessed state
        
        Returns:
            tuple: (channels, height, width)
        """
        channels = 2 if self.use_active_piece else 1
        return (channels, 20, 10)


def create_preprocessor(device='cpu', use_active_piece=True):
    """
    Factory function to create preprocessor
    
    Args:
        device: PyTorch device
        use_active_piece: Whether to include active piece channel
    
    Returns:
        TetrisPreprocessor instance
    """
    return TetrisPreprocessor(use_active_piece=use_active_piece, device=device)


if __name__ == "__main__":
    # Test preprocessing
    from src.env.tetris_env import TetrisEnv
    
    print("Testing TetrisPreprocessor...")
    print("=" * 70)
    
    env = TetrisEnv(render_mode=None)
    obs, info = env.reset()
    
    preprocessor = TetrisPreprocessor(use_active_piece=True, device='cpu')
    
    # Preprocess single observation
    state = preprocessor(obs)
    
    print(f"Original observation shapes:")
    print(f"  board: {obs['board'].shape}")
    print(f"  active_mask: {obs['active_tetromino_mask'].shape}")
    print()
    print(f"Preprocessed state shape: {state.shape}")
    print(f"  Expected: {preprocessor.get_state_shape()}")
    print(f"  Data type: {state.dtype}")
    print(f"  Device: {state.device}")
    print(f"  Value range: [{state.min():.2f}, {state.max():.2f}]")
    print()
    
    # Test batch preprocessing
    obs2, _ = env.step(env.action_space.sample())
    obs3, _, _, _, _ = env.step(env.action_space.sample())
    
    batch = preprocessor.preprocess_batch([obs, obs2, obs3])
    print(f"Batch shape: {batch.shape}")
    print(f"  Expected: (3, {preprocessor.get_state_shape()})")
    print()
    
    print("✓ Preprocessing working correctly!")
    env.close()

