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
                 reward_line_clear=10.0,      # Increased: Line clears are VERY good
                 reward_hole_penalty=-1.0,     # Increased: Holes are bad
                 reward_bumpiness_penalty=-0.05, # Decreased: Small changes OK
                 reward_height_penalty=-0.1,   # Increased slightly
                 reward_survival=0.1):
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
        # gravity=True: Each action immediately places the piece (standard Tetris RL)
        # This makes learning faster and more practical
        env = gym.make('tetris_gymnasium/Tetris', 
                      render_mode=render_mode,
                      height=height,
                      width=width,
                      gravity=True)
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
        
        # Store previous state for reward calculation (delta-based)
        self.prev_holes = 0
        self.prev_lines = 0
        self.prev_height = 0
        self.prev_bumpiness = 0
        
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
        self.prev_bumpiness = 0
        
        return obs, info
    
    def step(self, action):
        """
        Execute action and return observation with custom reward
        
        With gravity=True, every action places a piece, so we always
        calculate placement rewards.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            observation: Current game state
            reward: Custom shaped reward
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Execute the action (with gravity=True, this places the piece)
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate custom reward for this placement
        reward = self._calculate_custom_reward(obs, info, terminated)
        
        # Update statistics
        self.episode_steps += 1
        self.total_score += reward
        
        # Add metrics to info
        info['custom_reward'] = reward
        info['episode_steps'] = self.episode_steps
        info['holes'] = self._count_holes(obs['board'], obs.get('active_tetromino_mask'))
        info['bumpiness'] = self._calculate_bumpiness(obs['board'])
        info['max_height'] = self._get_max_height(obs['board'])
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_custom_reward(self, obs, info, terminated):
        """
        Calculate custom reward based on DELTA (changes) from this action
        
        With gravity=True, every action places a piece, so we always
        evaluate the placement quality.
        
        Reward components:
        - Lines cleared: Large positive reward (exponential for combos)
        - Holes change: Penalty for creating, reward for filling
        - Bumpiness change: Penalty for increasing, reward for smoothing
        - Height change: Penalty for increasing, reward for lowering
        - Survival: Small positive reward
        - Game over: Large negative penalty
        
        Args:
            obs: Observation dictionary containing 'board' and 'active_tetromino_mask'
            info: Info dictionary from environment
            terminated: Whether the episode has ended
        """
        reward = 0.0
        board = obs['board']
        active_mask = obs.get('active_tetromino_mask')
        
        # Reward for lines cleared (delta-based)
        current_lines = info.get('lines_cleared', 0)
        lines_delta = current_lines - self.prev_lines
        if lines_delta > 0:
            # Exponential bonus for clearing multiple lines at once
            reward += self.reward_line_clear * (2 ** lines_delta - 1)
            self.total_lines_cleared += lines_delta
        self.prev_lines = current_lines
        
        # Reward/penalty for holes change (delta-based)
        # Positive delta (more holes) = penalty
        # Negative delta (fewer holes) = reward!
        current_holes = self._count_holes(board, active_mask)
        holes_delta = current_holes - self.prev_holes
        reward += self.reward_hole_penalty * holes_delta
        self.prev_holes = current_holes
        
        # Reward/penalty for bumpiness change (delta-based)
        # Positive delta (bumpier) = penalty
        # Negative delta (smoother) = reward!
        current_bumpiness = self._calculate_bumpiness(board)
        bumpiness_delta = current_bumpiness - self.prev_bumpiness
        reward += self.reward_bumpiness_penalty * bumpiness_delta
        self.prev_bumpiness = current_bumpiness
        
        # Reward/penalty for height change (delta-based)
        # Positive delta (higher) = penalty
        # Negative delta (lower) = reward!
        current_height = self._get_max_height(board)
        height_delta = current_height - self.prev_height
        reward += self.reward_height_penalty * height_delta
        self.prev_height = current_height
        
        # Small survival bonus (encourages longer games)
        if not terminated:
            reward += self.reward_survival
        else:
            # Large penalty for game over
            reward -= 5.0
        
        return reward
    
    def _count_holes(self, board, active_mask=None):
        """
        Count the number of holes in the playable area
        A hole is an empty cell with at least one filled cell above it
        
        Note: tetris-gymnasium has border walls (columns 0-3 and 14-17).
        We only count holes in the playable area (columns 4-13).
        We exclude the active falling piece from the count.
        
        Args:
            board: 2D numpy array representing the game board
            active_mask: Optional mask of the currently falling piece
            
        Returns:
            Number of holes in playable area (excluding active piece)
        """
        holes = 0
        height, width = board.shape
        
        # Create a board without the active piece
        if active_mask is not None:
            # Subtract active piece from board to count only placed pieces
            board_without_active = board.copy()
            board_without_active[active_mask > 0] = 0
        else:
            board_without_active = board
        
        # Only check playable columns (exclude border walls)
        # tetris-gymnasium uses 4 columns on each side as walls
        playable_start = 4
        playable_end = width - 4
        
        for col in range(playable_start, playable_end):
            block_found = False
            for row in range(height):
                if board_without_active[row, col] > 0:
                    block_found = True
                elif block_found and board_without_active[row, col] == 0:
                    holes += 1
        
        return holes
    
    def _calculate_bumpiness(self, board):
        """
        Calculate bumpiness (sum of absolute height differences between adjacent columns)
        in the playable area only.
        
        Args:
            board: 2D numpy array representing the game board
            
        Returns:
            Total bumpiness value in playable area
        """
        heights = self._get_column_heights(board)
        bumpiness = 0
        
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        
        return bumpiness
    
    def _get_column_heights(self, board):
        """
        Get the height of each column in the playable area
        (number of rows from playable floor to highest filled cell)
        
        Note: 
        - Board has 24 total rows, bottom 4 are floor (rows 20-23)
        - Playable height is 0-20 (standard Tetris)
        - Returns heights for playable columns only (excluding border walls)
        
        Args:
            board: 2D numpy array representing the game board
            
        Returns:
            List of column heights (0-20 scale) for playable area only
        """
        total_rows, width = board.shape
        heights = []
        
        # Only check playable columns (exclude border walls)
        playable_start = 4
        playable_end = width - 4
        
        # Playable rows: 0-19 (rows 20-23 are floor)
        playable_height = 20
        
        for col in range(playable_start, playable_end):
            column_height = 0
            for row in range(playable_height):  # Only check playable rows
                if board[row, col] > 0:
                    # Height from playable floor (row 19)
                    column_height = playable_height - row
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

