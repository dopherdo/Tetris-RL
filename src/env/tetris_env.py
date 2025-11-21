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
                 reward_line_clear=1.0,
                 reward_hole_penalty=-0.5,
                 reward_bumpiness_penalty=-0.1,
                 reward_height_penalty=-0.05,
                 reward_survival=0.01):
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
        - Lines cleared: Positive reward
        - Holes created: Negative penalty
        - Column height variance (bumpiness): Negative penalty
        - Maximum height: Negative penalty
        - Survival: Small positive reward
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
        
        # Penalty for creating holes
        current_holes = self._count_holes(board)
        holes_delta = current_holes - self.prev_holes
        if holes_delta > 0:
            reward += self.reward_hole_penalty * holes_delta
        self.prev_holes = current_holes
        
        # Penalty for bumpiness (uneven columns)
        bumpiness = self._calculate_bumpiness(board)
        reward += self.reward_bumpiness_penalty * bumpiness
        
        # Penalty for stack height
        max_height = self._get_max_height(board)
        reward += self.reward_height_penalty * max_height
        
        # Small survival bonus (encourages longer games)
        if not terminated:
            reward += self.reward_survival
        else:
            # Large penalty for game over
            reward -= 5.0
        
        return reward
    
    def _count_holes(self, board):
        """
        Count the number of holes in the board
        A hole is an empty cell with at least one filled cell above it
        
        Args:
            board: 2D numpy array representing the game board
            
        Returns:
            Number of holes
        """
        holes = 0
        height, width = board.shape
        
        for col in range(width):
            block_found = False
            for row in range(height):
                if board[row, col] > 0:
                    block_found = True
                elif block_found and board[row, col] == 0:
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

