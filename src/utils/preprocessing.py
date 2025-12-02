"""
State preprocessing utilities for Tetris RL.
"""

from __future__ import annotations

import numpy as np


def preprocess_observation(obs: dict | np.ndarray) -> np.ndarray:
    """
    Convert environment observation into (H, W) float32 array.
    
    Tetris-Gymnasium returns observations as dicts with 'board' key.
    The board includes padding (4 pixels on each side), so we crop to playable area.
    
    Args:
        obs: Either a dict with 'board' key or a numpy array
    
    Returns:
        Preprocessed board state as (H, W) float32 array normalized to [0, 1]
    """
    # Extract board from observation
    if isinstance(obs, dict):
        board = obs['board']
    else:
        board = obs
    
    board = np.asarray(board)
    
    # Remove padding (tetris-gymnasium adds 4-pixel padding on each side)
    # Original board is 24x18, playable area is 20x10
    padding = 4
    if board.shape[0] == 24 and board.shape[1] == 18:
        # Crop: remove bottom 4 rows and left/right 4 columns
        board = board[:-padding, padding:-padding]
    
    # Handle multi-channel observations
    if board.ndim == 3:
        board = board[..., 0]
    
    # Convert to float32 and normalize
    board = board.astype(np.float32)
    
    # Normalize to [0, 1] range
    # Tetris pieces are represented as integers (0 = empty, 1+ = filled)
    if board.max() > 0:
        board = board / board.max()
    
    return board


def extract_features(board: np.ndarray) -> dict:
    """
    Extract hand-crafted features from board state (optional, for analysis).
    
    These features can be useful for:
    - Debugging and visualization
    - Reward shaping
    - Comparing with heuristic agents
    
    Args:
        board: Board state as (H, W) array
    
    Returns:
        Dictionary of extracted features
    """
    height, width = board.shape
    
    # Column heights
    column_heights = []
    for col in range(width):
        col_height = 0
        for row in range(height):
            if board[row, col] > 0:
                col_height = height - row
                break
        column_heights.append(col_height)
    
    # Holes: empty cells with filled cells above
    holes = 0
    for col in range(width):
        found_block = False
        for row in range(height):
            if board[row, col] > 0:
                found_block = True
            elif found_block and board[row, col] == 0:
                holes += 1
    
    # Bumpiness: sum of height differences between adjacent columns
    bumpiness = sum(abs(column_heights[i] - column_heights[i + 1]) 
                   for i in range(len(column_heights) - 1))
    
    # Aggregate height
    total_height = sum(column_heights)
    max_height = max(column_heights) if column_heights else 0
    
    # Number of filled cells
    filled_cells = int(np.sum(board > 0))
    
    return {
        'column_heights': column_heights,
        'holes': holes,
        'bumpiness': bumpiness,
        'total_height': total_height,
        'max_height': max_height,
        'filled_cells': filled_cells,
    }
