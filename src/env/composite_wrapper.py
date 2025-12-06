"""Composite Action Wrapper for Tetris Environment."""
import gymnasium as gym
import numpy as np


class CompositeActionWrapper(gym.Wrapper):
    """Wrapper that converts atomic actions to composite placement actions."""
    
    def __init__(self, env):
        """Initialize composite action wrapper."""
        super().__init__(env)
        
        self.action_space = gym.spaces.Discrete(40)
        
        self.MOVE_LEFT = 0
        self.MOVE_RIGHT = 1
        self.MOVE_DOWN = 2
        self.ROTATE_CW = 3
        self.ROTATE_CCW = 4
        self.HARD_DROP = 5
        self.SWAP = 6
        self.NOOP = 7
        
        self.board_width = 10
        self.max_steps_per_placement = 50
        
    def reset(self, **kwargs):
        """Reset environment."""
        return self.env.reset(**kwargs)
    
    def step(self, composite_action):
        """Execute a composite action (rotation + column placement)."""
        rotation = composite_action // self.board_width
        target_column = composite_action % self.board_width
        
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        
        steps_taken = 0
        
        for _ in range(rotation):
            obs, reward, terminated, truncated, info = self.env.step(self.ROTATE_CW)
            total_reward += reward
            steps_taken += 1
            
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info
        
        current_col = self._estimate_piece_column(obs)
        moves_needed = target_column - current_col
        
        if moves_needed > 0:
            for _ in range(abs(moves_needed)):
                obs, reward, terminated, truncated, info = self.env.step(self.MOVE_RIGHT)
                total_reward += reward
                steps_taken += 1
                
                if terminated or truncated or steps_taken >= self.max_steps_per_placement:
                    break
        elif moves_needed < 0:
            for _ in range(abs(moves_needed)):
                obs, reward, terminated, truncated, info = self.env.step(self.MOVE_LEFT)
                total_reward += reward
                steps_taken += 1
                
                if terminated or truncated or steps_taken >= self.max_steps_per_placement:
                    break
        
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = self.env.step(self.HARD_DROP)
            total_reward += reward
            steps_taken += 1
        
        info['composite_action'] = composite_action
        info['target_rotation'] = rotation
        info['target_column'] = target_column
        info['atomic_steps_taken'] = steps_taken
        
        return obs, total_reward, terminated, truncated, info
    
    def _estimate_piece_column(self, obs):
        """Estimate the current column of the active piece."""
        return 4
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        return self.env.close()


def make_composite_tetris_env(render_mode=None, **kwargs):
    """Factory function to create a Tetris environment with composite actions."""
    from .tetris_env import TetrisEnv
    
    base_env = TetrisEnv(render_mode=render_mode, **kwargs)
    composite_env = CompositeActionWrapper(base_env)
    
    return composite_env

