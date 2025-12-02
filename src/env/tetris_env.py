"""
Custom Tetris Environment Wrapper for RL Training
Built on top of tetris-gymnasium with custom reward engineering
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Import tetris_gymnasium to register the environment
import tetris_gymnasium.envs


class TetrisEnv(gym.Wrapper):
    """
    Custom Tetris Environment Wrapper with Reward Engineering
    
    Wraps the tetris-gymnasium environment and adds:
    - Custom reward shaping based on game state analysis
    - Simplified observation space (board grid + current piece)
    - Enhanced metrics tracking for training analysis
    """
    
    def __init__(self, 
                 render_mode=None,
                 height=20, 
                 width=10,
                 reward_line_clear=100.0,
                 reward_hole_penalty=-0.5,
                 reward_bumpiness_penalty=-0.1,
                 reward_height_penalty=-0.005,
                 reward_survival=0.2):
        """
        Initialize the Tetris environment with custom parameters
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            height: Board height (default: 20)
            width: Board width (default: 10)
            reward_line_clear: Reward per line cleared
            reward_hole_penalty: Penalty per hole created
            reward_bumpiness_penalty: Penalty for uneven column heights
            reward_height_penalty: Penalty for high stacks
            reward_survival: Small reward for surviving each step
        """
        # Create the base environment
        env = gym.make('tetris_gymnasium/Tetris', 
                      render_mode=render_mode,
                      height=height,
                      width=width)
        super().__init__(env)
        
        # Store reward parameters
        self.reward_line_clear = reward_line_clear
        self.reward_hole_penalty = reward_hole_penalty
        self.reward_bumpiness_penalty = reward_bumpiness_penalty
        self.reward_height_penalty = reward_height_penalty
        self.reward_survival = reward_survival
        
        # Track game statistics
        self.total_lines_cleared = 0
        self.total_score = 0
        self.episode_steps = 0
        
        # Store previous state for reward calculation
        self.prev_holes = 0
        self.prev_lines = 0
        self.prev_height = 0
        
    def reset(self, **kwargs):
        """Reset the environment and statistics"""
        obs, info = self.env.reset(**kwargs)
        
        # Reset statistics
        self.total_lines_cleared = 0
        self.total_score = 0
        self.episode_steps = 0
        self.prev_holes = 0
        self.prev_lines = 0
        self.prev_height = 0
        
        return obs, info
    
    def step(self, action):
        """
        Execute action and return observation with custom reward
        
        Args:
            action: Action to take in the environment
            
        Returns:
            observation: Current game state
            reward: Custom shaped reward
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Execute the action
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        # Extract board state
        board = obs['board']
        
        # Calculate custom reward components
        reward = self._calculate_custom_reward(board, info, terminated)
        
        # Update statistics
        self.episode_steps += 1
        self.total_score += reward
        
        # Add metrics to info
        info['custom_reward'] = reward
        info['episode_steps'] = self.episode_steps
        info['holes'] = self._count_holes(board)
        info['bumpiness'] = self._calculate_bumpiness(board)
        info['max_height'] = self._get_max_height(board)
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_custom_reward(self, board, info, terminated):
        """
        Calculate custom reward based on game state analysis
        
        Reward components:
        - Lines cleared: Large positive reward
        - Almost-complete rows: Positive reward (guides toward line clearing)
        - Holes created: Negative penalty
        - Column height variance (bumpiness): Negative penalty
        - Flat surface bonus: Positive reward for low bumpiness
        - Maximum height: Negative penalty
        - Survival: Small positive reward (reduced to not dominate)
        - Game over: Large negative penalty
        """
        reward = 0.0
        
        # Reward for lines cleared
        current_lines = info.get('lines_cleared', 0)
        lines_delta = current_lines - self.prev_lines
        if lines_delta > 0:
            # Exponential bonus for clearing multiple lines at once
            reward += self.reward_line_clear * (2 ** lines_delta - 1)
            self.total_lines_cleared += lines_delta
        self.prev_lines = current_lines
        
        # NEW: Reward for almost-complete rows (guides toward line clearing)
        almost_complete_bonus = self._reward_almost_complete_rows(board)
        reward += almost_complete_bonus
        
        # Reward/penalty based on change in number of holes
        current_holes = self._count_holes(board)
        holes_delta = current_holes - self.prev_holes
        # If holes increase (holes_delta > 0) this is negative (penalty),
        # if holes decrease (holes_delta < 0) this becomes positive (reward).
        if holes_delta != 0:
            reward += self.reward_hole_penalty * holes_delta
        self.prev_holes = current_holes
        
        # Penalty for bumpiness (uneven columns)
        bumpiness = self._calculate_bumpiness(board)
        reward += self.reward_bumpiness_penalty * bumpiness
        
        # NEW: Bonus for flat surface (inverse of bumpiness)
        if bumpiness <= 3:  # Very flat surface
            reward += 5.0
        elif bumpiness <= 5:  # Moderately flat
            reward += 3.0
        elif bumpiness <= 8:  # Somewhat flat
            reward += 1.0
        
        # Penalty for stack height
        max_height = self._get_max_height(board)
        reward += self.reward_height_penalty * max_height
        
        # Survival bonus (HEAVILY REDUCED - should not dominate)
        if not terminated:
            reward += self.reward_survival  # Already set low at 0.2
        else:
            # VERY large penalty for game over (must be worse than cumulative penalties)
            reward -= 100.0
        
        return reward
    
    def _reward_almost_complete_rows(self, board):
        """
        Reward rows that are almost complete to guide agent toward line clearing.
        
        This helps the agent discover that filling rows is valuable BEFORE
        it accidentally clears a line.
        
        Args:
            board: 2D numpy array representing the game board
            
        Returns:
            Reward for almost-complete rows
        """
        # Remove padding to get playable area
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
                # Complete row (should be cleared by environment)
                pass
            elif filled_cells == width - 1:
                # 9/10 filled - very close! HUGE reward
                reward += 8.0
            elif filled_cells == width - 2:
                # 8/10 filled - getting there
                reward += 4.0
            elif filled_cells >= width - 3:
                # 7/10 filled - making progress
                reward += 2.0
        
        return reward
    
    def _count_holes(self, board):
        """
        Count the number of holes in the board.

        Here we treat a hole as an empty cell that is *inside* a stack of
        blocks in a column: it has at least one filled cell above and at
        least one filled cell below in that column. This avoids counting
        empty cells that are just below a falling piece.
        """
        holes = 0
        height, width = board.shape

        for col in range(width):
            # Find all filled cells in this column
            filled_rows = [row for row in range(height) if board[row, col] > 0]
            if len(filled_rows) <= 1:
                # With 0 or 1 filled cells there can't be an internal hole
                continue

            top = filled_rows[0]
            bottom = filled_rows[-1]

            # Any empty cells strictly between top and bottom are holes
            for row in range(top + 1, bottom):
                if board[row, col] == 0:
                    holes += 1

        return holes
    
    def _calculate_bumpiness(self, board):
        """
        Calculate bumpiness (sum of absolute height differences between adjacent columns)
        
        Args:
            board: 2D numpy array representing the game board
            
        Returns:
            Total bumpiness value
        """
        heights = self._get_column_heights(board)
        bumpiness = 0
        
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        
        return bumpiness
    
    def _get_column_heights(self, board):
        """
        Get the height of each column (number of filled cells from bottom)
        
        Args:
            board: 2D numpy array representing the game board
            
        Returns:
            List of column heights
        """
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
        """
        Get the maximum column height
        
        Args:
            board: 2D numpy array representing the game board
            
        Returns:
            Maximum height
        """
        heights = self._get_column_heights(board)
        return max(heights) if heights else 0
    
    def get_observation_space(self):
        """Return the observation space of the environment"""
        return self.env.observation_space
    
    def get_action_space(self):
        """Return the action space of the environment"""
        return self.env.action_space


if __name__ == "__main__":
    """
    Simple manual test for the Tetris environment.
    
    Run with:
        python3 -m src.env.tetris_env
    """
    import time

    # Create environment without rendering for a quick smoke test
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

        # Small sleep so output is readable if run with render_mode='human'
        time.sleep(0.05)

    print(f"Smoke test finished. Total reward over 20 steps: {total_reward:.3f}")

    # Print final board state from the last observation
    board = obs["board"]
    
    # Crop padding to show only playable area (remove walls/floor added by tetris-gymnasium)
    # tetris-gymnasium adds 4-pixel padding on each side
    padding = 4
    playable_board = board[:-padding, padding:-padding]
    
    print("\nFinal board state (playable area only):")
    print(" " + "-" * (playable_board.shape[1] * 2 + 1))
    for row in playable_board:
        print(" |" + "".join(["█" if cell > 0 else " " for cell in row]) + "|")
    print(" " + "-" * (playable_board.shape[1] * 2 + 1))

    env.close()

