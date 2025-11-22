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
    Custom Tetris Environment with Composite Actions and Reward Engineering
    
    Uses composite direct placement actions for easier PPO training:
    - Action space: 40 actions (10 columns × 4 rotations)
    - Each action directly places a piece at (column, rotation)
    - Custom reward shaping based on game state analysis
    - Enhanced metrics tracking for training analysis
    
    Action Encoding:
        Actions 0-9:   Place in columns 0-9 with rotation 0 (0°)
        Actions 10-19: Place in columns 0-9 with rotation 1 (90°)
        Actions 20-29: Place in columns 0-9 with rotation 2 (180°)
        Actions 30-39: Place in columns 0-9 with rotation 3 (270°)
    """
    
    def __init__(self, 
                 render_mode=None,
                 height=20, 
                 width=10,
                 reward_line_clear=10.0,           # Strong positive for clearing lines
                 reward_hole_penalty=-1.0,         # Strongest penalty (per hole created)
                 reward_bumpiness_penalty=-0.1,    # Weak penalty (per bumpiness unit increase)
                 reward_height_penalty=-0.2,       # Medium penalty (per height increase)
                 reward_survival=0.1,              # Small bonus for surviving
                 reward_on_placement_only=True):   # Only give rewards when piece locks             # Small bonus for surviving
        """
        Initialize the Tetris environment with composite actions and custom rewards
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            height: Board height (default: 20)
            width: Board width (default: 10)
            reward_line_clear: Reward multiplier for lines cleared (exponential: 10, 30, 70, 150)
            reward_hole_penalty: Penalty per hole created (positive for filled holes!)
            reward_bumpiness_penalty: Penalty per bumpiness unit (symmetric delta-based)
            reward_height_penalty: Penalty per height unit increase (symmetric delta-based)
            reward_survival: Small reward for surviving each step
            reward_on_placement_only: If True, only give shaped rewards when piece locks.
                                      If False, give rewards every action (default: True)
            
        Action Space:
            Composite actions (40 total): column (0-9) × rotation (0-3)
            - Actions 0-9:   Column 0-9, no rotation (0°)
            - Actions 10-19: Column 0-9, 90° rotation
            - Actions 20-29: Column 0-9, 180° rotation  
            - Actions 30-39: Column 0-9, 270° rotation
            
        Note: All metrics are delta-based (only changes are rewarded/penalized)
        Penalty ratios: Holes (1.0) : Height (0.2) : Bumpiness (0.1)
        """
        # Create the base environment
        env = gym.make('tetris_gymnasium/Tetris', 
                      render_mode=render_mode,
                      height=height,
                      width=width)
        super().__init__(env)
        
        # Override action space for composite actions
        # 10 columns × 4 rotations = 40 actions
        self.action_space = spaces.Discrete(40)
        
        # Atomic action mappings for the base environment
        self.MOVE_LEFT = 0
        self.MOVE_RIGHT = 1
        self.ROTATE = 2
        self.HARD_DROP = 5
        
        # Store reward parameters
        self.reward_line_clear = reward_line_clear
        self.reward_hole_penalty = reward_hole_penalty
        self.reward_bumpiness_penalty = reward_bumpiness_penalty
        self.reward_height_penalty = reward_height_penalty
        self.reward_survival = reward_survival
        self.reward_on_placement_only = reward_on_placement_only
        
        # Track game statistics
        self.total_lines_cleared = 0
        self.total_score = 0
        self.pieces_placed = 0
        
        # Store previous state for reward calculation
        self.prev_holes = 0
        self.prev_lines = 0
        self.prev_height = 0
        self.prev_bumpiness = 0
        
        # Store previous board to detect when piece locks
        self.prev_board_hash = None
        
        # Store last observation for composite actions
        self.last_obs = None
        
    def reset(self, **kwargs):
        """Reset the environment and statistics"""
        obs, info = self.env.reset(**kwargs)
        
        # Reset statistics
        self.total_lines_cleared = 0
        self.total_score = 0
        self.pieces_placed = 0
        self.prev_lines = 0
        
        # Initialize previous state from the actual initial board (settled pieces only)
        # This ensures first move only accounts for the change it causes
        self.prev_holes = self._count_holes_settled(obs)
        self.prev_height = self._get_max_height_settled(obs)
        self.prev_bumpiness = self._calculate_bumpiness_settled(obs)
        settled_board = self._get_settled_board(obs)
        self.prev_board_hash = self._hash_board(settled_board)
        
        # Store observation for composite actions
        self.last_obs = obs
        
        return obs, info
    
    def decode_action(self, composite_action):
        """
        Decode composite action into (column, rotation)
        
        Args:
            composite_action: Integer 0-39
            
        Returns:
            tuple: (column 0-9, rotation 0-3)
        """
        column = composite_action % 10
        rotation = composite_action // 10
        return column, rotation
    
    def step(self, composite_action):
        """
        Execute composite action: directly place piece at (column, rotation)
        
        Args:
            composite_action: Integer 0-39 representing (column, rotation)
            
        Returns:
            observation: Current game state
            reward: Custom shaped reward
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Decode composite action
        target_column, num_rotations = self.decode_action(composite_action)
        
        # Get current piece position
        if self.last_obs is None:
            # Shouldn't happen, but handle gracefully
            obs, reward, terminated, truncated, info = self.env.step(self.HARD_DROP)
            self.last_obs = obs
            return self._finalize_step(obs, reward, terminated, truncated, info, composite_action)
        
        active_mask = self.last_obs['active_tetromino_mask']
        active_positions = np.argwhere(active_mask > 0)
        
        if len(active_positions) == 0:
            # No active piece, just drop
            obs, reward, terminated, truncated, info = self.env.step(self.HARD_DROP)
            self.last_obs = obs
            return self._finalize_step(obs, reward, terminated, truncated, info, composite_action)
        
        # Calculate current column (in playable area 0-9)
        # Full board columns 4-13 map to our columns 0-9
        current_column_full = int(np.mean(active_positions[:, 1]))
        current_column = current_column_full - 4
        
        # Step 1: Apply rotations
        for _ in range(num_rotations):
            obs, reward, terminated, truncated, info = self.env.step(self.ROTATE)
            self.last_obs = obs
            if terminated or truncated:
                return self._finalize_step(obs, reward, terminated, truncated, info, composite_action)
        
        # Step 2: Move to target column
        column_diff = target_column - current_column
        
        if column_diff < 0:
            # Move left
            for _ in range(abs(column_diff)):
                obs, reward, terminated, truncated, info = self.env.step(self.MOVE_LEFT)
                self.last_obs = obs
                if terminated or truncated:
                    return self._finalize_step(obs, reward, terminated, truncated, info, composite_action)
        elif column_diff > 0:
            # Move right
            for _ in range(column_diff):
                obs, reward, terminated, truncated, info = self.env.step(self.MOVE_RIGHT)
                self.last_obs = obs
                if terminated or truncated:
                    return self._finalize_step(obs, reward, terminated, truncated, info, composite_action)
        
        # Step 3: Hard drop to place piece
        obs, reward, terminated, truncated, info = self.env.step(self.HARD_DROP)
        self.last_obs = obs
        
        return self._finalize_step(obs, reward, terminated, truncated, info, composite_action)
    
    def _finalize_step(self, obs, reward, terminated, truncated, info, composite_action):
        """
        Finalize step by calculating custom rewards and updating statistics
        
        Args:
            obs: Observation from environment
            reward: Base reward (ignored, we use custom)
            terminated: Episode ended flag
            truncated: Episode truncated flag
            info: Info dict
            composite_action: The composite action taken
            
        Returns:
            obs, custom_reward, terminated, truncated, updated_info
        """
        # Check if piece has been placed (settled board changed)
        settled_board = self._get_settled_board(obs)
        current_board_hash = self._hash_board(settled_board)
        piece_placed = (current_board_hash != self.prev_board_hash)
        
        # Update hash after checking
        if piece_placed:
            self.prev_board_hash = current_board_hash
        
        # Calculate custom reward
        if self.reward_on_placement_only:
            # Only give shaped rewards when piece locks in
            if piece_placed or terminated:
                custom_reward = self._calculate_custom_reward(obs, info, terminated)
            else:
                # Small survival bonus while maneuvering piece
                custom_reward = 0.01
        else:
            # Give shaped rewards every action
            custom_reward = self._calculate_custom_reward(obs, info, terminated)
        
        # Update statistics
        self.total_score += custom_reward
        if piece_placed:
            self.pieces_placed += 1
        
        # Add metrics to info
        info['custom_reward'] = custom_reward
        info['pieces_placed'] = self.pieces_placed
        info['holes'] = self._count_holes_settled(obs)
        info['bumpiness'] = self._calculate_bumpiness_settled(obs)
        info['max_height'] = self._get_max_height_settled(obs)
        info['piece_placed'] = piece_placed
        info['composite_action'] = composite_action
        info['target_column'] = composite_action % 10
        info['rotation'] = composite_action // 10
        
        return obs, custom_reward, terminated, truncated, info
    
    def _calculate_custom_reward(self, obs, info, terminated):
        """
        Calculate custom reward based on game state changes (delta-based)
        
        Reward components:
        - Lines cleared: Strong positive reward (exponential)
        - Holes delta: Negative if created, POSITIVE if filled! (symmetric)
        - Bumpiness delta: Penalty if increased, reward if decreased (symmetric)
        - Height delta: Penalty if increased, reward if decreased (symmetric)
        - Survival: Small positive reward per step
        - Game over: Large negative penalty
        
        Penalty ratios: Holes (1.0) : Height (0.2) : Bumpiness (0.1)
        This reflects that holes are ~10x worse than bumpiness
        
        Note: All metrics are calculated on SETTLED pieces only (excludes active piece)
        """
        reward = 0.0
        
        # Reward for lines cleared (exponential: 10, 30, 70, 150 for 1-4 lines)
        current_lines = info.get('lines_cleared', 0)
        lines_delta = current_lines - self.prev_lines
        if lines_delta > 0:
            reward += self.reward_line_clear * (2 ** lines_delta - 1)
            self.total_lines_cleared += lines_delta
        self.prev_lines = current_lines
        
        # Symmetric holes reward: penalty for creating, BONUS for filling!
        # Only count holes in settled pieces (exclude active piece)
        current_holes = self._count_holes_settled(obs)
        holes_delta = current_holes - self.prev_holes
        if holes_delta != 0:
            # Negative delta (filled holes) gives positive reward!
            reward += self.reward_hole_penalty * holes_delta
        self.prev_holes = current_holes
        
        # Symmetric bumpiness reward: penalty if increased, bonus if decreased
        bumpiness = self._calculate_bumpiness_settled(obs)
        bumpiness_delta = bumpiness - self.prev_bumpiness
        if bumpiness_delta != 0:
            reward += self.reward_bumpiness_penalty * bumpiness_delta
        self.prev_bumpiness = bumpiness
        
        # Symmetric height reward: penalty if increased, bonus if decreased
        max_height = self._get_max_height_settled(obs)
        height_delta = max_height - self.prev_height
        if height_delta != 0:
            reward += self.reward_height_penalty * height_delta
        self.prev_height = max_height
        
        # Small survival bonus (encourages longer games)
        if not terminated:
            reward += self.reward_survival
        else:
            # Large penalty for game over
            reward -= 5.0
        
        return reward
    
    def _get_settled_board(self, obs):
        """
        Get the board with only settled pieces (excludes active falling piece)
        Also extracts only the playable area: 20 rows × 10 columns
        - Excludes walls (columns 0-3 and 14-17)
        - Excludes floor (rows 20-23)
        
        Args:
            obs: Observation dict with 'board' and 'active_tetromino_mask'
            
        Returns:
            Board array (20×10) with active piece removed, walls and floor excluded
        """
        board = obs['board'].copy()
        active_mask = obs['active_tetromino_mask']
        board[active_mask > 0] = 0  # Remove active piece
        
        # Extract only playable area: rows 0-19, columns 4-13 (20×10)
        playable_board = board[0:20, 4:14]
        return playable_board
    
    def _count_holes_settled(self, obs):
        """
        Count the number of holes in SETTLED pieces only (excludes active piece)
        A hole is an empty cell with at least one settled filled cell above it
        Works on playable area only (20×10, walls and floor already excluded)
        
        Args:
            obs: Observation dict with 'board' and 'active_tetromino_mask'
            
        Returns:
            Number of holes
        """
        board = self._get_settled_board(obs)  # Already 20×10 playable area
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
    
    def _count_holes(self, board):
        """
        Count the number of holes in the board (legacy method, includes active piece)
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
    
    def _calculate_bumpiness_settled(self, obs):
        """
        Calculate bumpiness of SETTLED pieces only (excludes active piece)
        Bumpiness = sum of absolute height differences between adjacent columns
        Works on playable area only (20×10, walls and floor already excluded)
        
        Args:
            obs: Observation dict with 'board' and 'active_tetromino_mask'
            
        Returns:
            Total bumpiness value
        """
        board = self._get_settled_board(obs)  # Already 20×10 playable area
        heights = self._get_column_heights(board)
        bumpiness = 0
        
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        
        return bumpiness
    
    def _calculate_bumpiness(self, board):
        """
        Calculate bumpiness (sum of absolute height differences between adjacent columns)
        Legacy method that includes active piece
        
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
    
    def _get_max_height_settled(self, obs):
        """
        Get the maximum column height of SETTLED pieces only (excludes active piece)
        Works on playable area only (20×10, walls and floor already excluded)
        
        Args:
            obs: Observation dict with 'board' and 'active_tetromino_mask'
            
        Returns:
            Maximum height in playable area (0-20)
        """
        board = self._get_settled_board(obs)  # Already 20×10 playable area
        heights = self._get_column_heights(board)
        return max(heights) if heights else 0
    
    def _get_max_height(self, board):
        """
        Get the maximum column height (legacy method, includes active piece)
        
        Args:
            board: 2D numpy array representing the game board
            
        Returns:
            Maximum height
        """
        heights = self._get_column_heights(board)
        return max(heights) if heights else 0
    
    def _hash_board(self, board):
        """
        Create a hash of the board state to detect when pieces lock
        
        Args:
            board: 2D numpy array representing the game board
            
        Returns:
            Hash of the board state
        """
        return hash(board.tobytes())
    
    def get_observation_space(self):
        """Return the observation space of the environment"""
        return self.env.observation_space
    
    def get_action_space(self):
        """Return the action space of the environment"""
        return self.env.action_space

