"""
Composite Action Wrapper for Tetris Environment

Converts atomic actions (left, right, rotate, drop) into composite actions
that represent complete piece placements (rotation + column position).
"""

import gymnasium as gym
import numpy as np


class CompositeActionWrapper(gym.Wrapper):
    """
    Wrapper that converts atomic actions to composite placement actions.
    
    Instead of agent choosing: [rotate, left, left, drop]
    Agent chooses: Action 23 = "Place at rotation 2, column 3"
    
    Action Space: Discrete(40)
    - Actions 0-9: Rotation 0, columns 0-9
    - Actions 10-19: Rotation 1, columns 0-9
    - Actions 20-29: Rotation 2, columns 0-9
    - Actions 30-39: Rotation 3, columns 0-9
    """
    
    def __init__(self, env):
        """
        Initialize composite action wrapper.
        
        Args:
            env: Base Tetris environment (should be TetrisEnv or similar)
        """
        super().__init__(env)
        
        # Override action space to composite actions
        # 4 rotations × 10 columns = 40 total actions
        self.action_space = gym.spaces.Discrete(40)
        
        # Keep track of atomic action mapping
        # Assuming standard tetris-gymnasium action indices:
        self.MOVE_LEFT = 0
        self.MOVE_RIGHT = 1
        self.MOVE_DOWN = 2
        self.ROTATE_CW = 3
        self.ROTATE_CCW = 4
        self.HARD_DROP = 5
        self.SWAP = 6
        self.NOOP = 7
        
        self.board_width = 10  # Standard Tetris width
        self.max_steps_per_placement = 50  # Safety limit
        
    def reset(self, **kwargs):
        """Reset environment."""
        return self.env.reset(**kwargs)
    
    def step(self, composite_action):
        """
        Execute a composite action (rotation + column placement).
        
        Args:
            composite_action: Integer 0-39 representing (rotation, column)
        
        Returns:
            observation, cumulative_reward, terminated, truncated, info
        """
        # Decode composite action into rotation and target column
        rotation = composite_action // self.board_width  # 0-3
        target_column = composite_action % self.board_width  # 0-9
        
        # Execute the placement as a sequence of atomic actions
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        
        steps_taken = 0
        
        # Step 1: Rotate the piece
        for _ in range(rotation):
            obs, reward, terminated, truncated, info = self.env.step(self.ROTATE_CW)
            total_reward += reward
            steps_taken += 1
            
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info
        
        # Step 2: Get current piece position (we'll try to move to target column)
        # We'll move left/right until we reach target or hit max steps
        
        # Get current x position from board observation
        # In tetris-gymnasium, the active piece position is tracked internally
        # We'll use a simple strategy: try to move to target column
        
        current_col = self._estimate_piece_column(obs)
        
        # Move to target column
        moves_needed = target_column - current_col
        
        if moves_needed > 0:
            # Move right
            for _ in range(abs(moves_needed)):
                obs, reward, terminated, truncated, info = self.env.step(self.MOVE_RIGHT)
                total_reward += reward
                steps_taken += 1
                
                if terminated or truncated or steps_taken >= self.max_steps_per_placement:
                    break
        elif moves_needed < 0:
            # Move left
            for _ in range(abs(moves_needed)):
                obs, reward, terminated, truncated, info = self.env.step(self.MOVE_LEFT)
                total_reward += reward
                steps_taken += 1
                
                if terminated or truncated or steps_taken >= self.max_steps_per_placement:
                    break
        
        # Step 3: Hard drop to place the piece
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = self.env.step(self.HARD_DROP)
            total_reward += reward
            steps_taken += 1
        
        # Add metadata about composite action
        info['composite_action'] = composite_action
        info['target_rotation'] = rotation
        info['target_column'] = target_column
        info['atomic_steps_taken'] = steps_taken
        
        return obs, total_reward, terminated, truncated, info
    
    def _estimate_piece_column(self, obs):
        """
        Estimate the current column of the active piece.
        
        Since tetris-gymnasium doesn't directly expose piece position,
        we'll use a heuristic: start from middle (column 4-5) and let
        the movement logic handle it.
        
        Args:
            obs: Current observation
        
        Returns:
            Estimated column (default to middle of board)
        """
        # Default starting position is typically middle of board
        # Tetris pieces usually spawn at column 3-5
        return 4
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        return self.env.close()


def make_composite_tetris_env(render_mode=None, **kwargs):
    """
    Factory function to create a Tetris environment with composite actions.
    
    Args:
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        **kwargs: Additional arguments for TetrisEnv
    
    Returns:
        CompositeActionWrapper wrapping TetrisEnv
    """
    from .tetris_env import TetrisEnv
    
    base_env = TetrisEnv(render_mode=render_mode, **kwargs)
    composite_env = CompositeActionWrapper(base_env)
    
    return composite_env

