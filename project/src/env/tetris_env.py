"""Custom Tetris Environment Wrapper with reward engineering."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import tetris_gymnasium.envs


class TetrisEnv(gym.Wrapper):
    """Custom Tetris Environment Wrapper with reward engineering."""
    
    def __init__(self, 
                 render_mode=None,
                 height=20, 
                 width=10,
                 reward_line_clear=100.0,
                 reward_hole_penalty=-0.5,
                 reward_bumpiness_penalty=-0.1,
                 reward_height_penalty=-0.005,
                 reward_survival=0.2):
        env = gym.make('tetris_gymnasium/Tetris', 
                      render_mode=render_mode,
                      height=height,
                      width=width)
        super().__init__(env)
        
        self.reward_line_clear = reward_line_clear
        self.reward_hole_penalty = reward_hole_penalty
        self.reward_bumpiness_penalty = reward_bumpiness_penalty
        self.reward_height_penalty = reward_height_penalty
        self.reward_survival = reward_survival
        
        self.total_lines_cleared = 0
        self.total_score = 0
        self.episode_steps = 0
        
        self.prev_holes = 0
        self.prev_lines = 0
        self.prev_height = 0
        
    def reset(self, **kwargs):
        """Reset the environment and statistics."""
        obs, info = self.env.reset(**kwargs)
        
        self.total_lines_cleared = 0
        self.total_score = 0
        self.episode_steps = 0
        self.prev_holes = 0
        self.prev_lines = 0
        self.prev_height = 0
        
        return obs, info
    
    def step(self, action):
        """Execute action and return observation with custom reward."""
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        board = obs['board']
        reward = self._calculate_custom_reward(board, info, terminated)
        
        self.episode_steps += 1
        self.total_score += reward
        
        info['custom_reward'] = reward
        info['episode_steps'] = self.episode_steps
        info['holes'] = self._count_holes(board)
        info['bumpiness'] = self._calculate_bumpiness(board)
        info['max_height'] = self._get_max_height(board)
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_custom_reward(self, board, info, terminated):
        """Calculate custom reward based on game state analysis."""
        reward = 0.0
        
        current_lines = info.get('lines_cleared', 0)
        lines_delta = current_lines - self.prev_lines
        if lines_delta > 0:
            reward += self.reward_line_clear * (2 ** lines_delta - 1)
            self.total_lines_cleared += lines_delta
        self.prev_lines = current_lines
        
        almost_complete_bonus = self._reward_almost_complete_rows(board)
        reward += almost_complete_bonus
        
        current_holes = self._count_holes(board)
        holes_delta = current_holes - self.prev_holes
        if holes_delta != 0:
            reward += self.reward_hole_penalty * holes_delta
        self.prev_holes = current_holes
        
        bumpiness = self._calculate_bumpiness(board)
        reward += self.reward_bumpiness_penalty * bumpiness
        
        if bumpiness <= 3:
            reward += 5.0
        elif bumpiness <= 5:
            reward += 3.0
        elif bumpiness <= 8:
            reward += 1.0
        
        max_height = self._get_max_height(board)
        reward += self.reward_height_penalty * max_height
        
        if not terminated:
            reward += self.reward_survival
        else:
            reward -= 100.0
        
        return reward
    
    def _reward_almost_complete_rows(self, board):
        """Reward rows that are almost complete to guide agent toward line clearing."""
        padding = 4
        if board.shape[0] == 24 and board.shape[1] == 18:
            playable_board = board[:-padding, padding:-padding]
        else:
            playable_board = board
        
        height, width = playable_board.shape
        reward = 0.0
        
        for row in playable_board:
            filled_cells = sum(1 for cell in row if cell > 0)
            
            if filled_cells == width:
                pass
            elif filled_cells == width - 1:
                reward += 8.0
            elif filled_cells == width - 2:
                reward += 4.0
            elif filled_cells >= width - 3:
                reward += 2.0
        
        return reward
    
    def _count_holes(self, board):
        """Count the number of holes in the board."""
        holes = 0
        height, width = board.shape

        for col in range(width):
            filled_rows = [row for row in range(height) if board[row, col] > 0]
            if len(filled_rows) <= 1:
                continue

            top = filled_rows[0]
            bottom = filled_rows[-1]

            for row in range(top + 1, bottom):
                if board[row, col] == 0:
                    holes += 1

        return holes
    
    def _calculate_bumpiness(self, board):
        """Calculate bumpiness (sum of absolute height differences between adjacent columns)."""
        heights = self._get_column_heights(board)
        bumpiness = 0
        
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        
        return bumpiness
    
    def _get_column_heights(self, board):
        """Get the height of each column."""
        height, width = board.shape
        heights = []
        
        for col in range(width):
            column_height = 0
            for row in range(height):
                if board[row, col] > 0:
                    column_height = height - row
                    break
            heights.append(column_height)
        
        return heights
    
    def _get_max_height(self, board):
        """Get the maximum column height."""
        heights = self._get_column_heights(board)
        return max(heights) if heights else 0
    
    def get_observation_space(self):
        """Return the observation space of the environment"""
        return self.env.observation_space
    
    def get_action_space(self):
        """Return the action space of the environment"""
        return self.env.action_space


if __name__ == "__main__":
    import time

    env = TetrisEnv(render_mode=None)
    obs, info = env.reset()

    print("Starting TetrisEnv smoke test (20 random steps)...")
    total_reward = 0.0

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"Step {step+1:02d} | "
            f"Action={action} | "
            f"Reward={reward:.3f} | "
            f"Holes={info['holes']} | "
            f"MaxHeight={info['max_height']} | "
            f"Done={terminated or truncated}"
        )

        if terminated or truncated:
            print("Episode finished early, resetting env...")
            obs, info = env.reset()

        time.sleep(0.05)

    print(f"Smoke test finished. Total reward over 20 steps: {total_reward:.3f}")

    board = obs["board"]
    padding = 4
    playable_board = board[:-padding, padding:-padding]
    
    print("\nFinal board state (playable area only):")
    print(" " + "-" * (playable_board.shape[1] * 2 + 1))
    for row in playable_board:
        print(" |" + "".join(["█" if cell > 0 else " " for cell in row]) + "|")
    print(" " + "-" * (playable_board.shape[1] * 2 + 1))

    env.close()

