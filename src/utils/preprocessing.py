"""State preprocessing utilities for Tetris RL."""
from __future__ import annotations

import numpy as np


def preprocess_observation(obs: dict | np.ndarray) -> np.ndarray:
    """Convert environment observation into (H, W) float32 array."""
    if isinstance(obs, dict):
        board = obs['board']
    else:
        board = obs
    
    board = np.asarray(board)
    
    padding = 4
    if board.shape[0] == 24 and board.shape[1] == 18:
        board = board[:-padding, padding:-padding]
    
    if board.ndim == 3:
        board = board[..., 0]
    
    board = board.astype(np.float32)
    
    if board.max() > 0:
        board = board / board.max()
    
    return board


def extract_features(board: np.ndarray) -> dict:
    """Extract hand-crafted features from board state."""
    height, width = board.shape
    
    column_heights = []
    for col in range(width):
        col_height = 0
        for row in range(height):
            if board[row, col] > 0:
                col_height = height - row
                break
        column_heights.append(col_height)
    
    holes = 0
    for col in range(width):
        found_block = False
        for row in range(height):
            if board[row, col] > 0:
                found_block = True
            elif found_block and board[row, col] == 0:
                holes += 1
    
    bumpiness = sum(abs(column_heights[i] - column_heights[i + 1]) 
                   for i in range(len(column_heights) - 1))
    
    total_height = sum(column_heights)
    max_height = max(column_heights) if column_heights else 0
    
    filled_cells = int(np.sum(board > 0))
    
    return {
        'column_heights': column_heights,
        'holes': holes,
        'bumpiness': bumpiness,
        'total_height': total_height,
        'max_height': max_height,
        'filled_cells': filled_cells,
    }
