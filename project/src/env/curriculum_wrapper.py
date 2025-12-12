"""
Curriculum Learning Wrapper for Tetris Environment

Gradually increases difficulty by starting with partially-filled boards
that are easier to clear, then transitioning to empty boards.
"""

import numpy as np
import gymnasium as gym


class CurriculumTetrisWrapper(gym.Wrapper):
    """
    Curriculum Learning Wrapper that controls task difficulty.
    
    Training Stages:
    - Stage 1 (0-15K steps):   Start with 17-19 rows filled (easiest - need 1-3 pieces to clear)
    - Stage 2 (15K-40K steps): Start with 14-16 rows filled (need 4-6 pieces)
    - Stage 3 (40K-80K steps): Start with 10-13 rows filled (need 7-10 pieces)
    - Stage 4 (80K+ steps):    Empty board (full difficulty)
    """
    
    def __init__(self, env, initial_step=0):
        super().__init__(env)
        self.training_step = initial_step
        self.current_stage = self._get_stage(initial_step)
        
        # Track curriculum progress
        self.stage_transitions = {
            1: 15000,
            2: 40000,
            3: 80000,
            4: float('inf')
        }
        
        # NEW SIMPLE CURRICULUM:
        # Stage 1: ONLY bottom row with EXACTLY 1 gap (easiest possible!)
        # Stage 2: Bottom 2-3 rows with 1-2 gaps each
        # Stage 3: Bottom 4-6 rows with 1-2 gaps each
        # Stage 4: Empty board
        self.stage_config = {
            1: {'num_rows': 1, 'gaps_per_row': 1},          # EASIEST: 1 row, 1 gap!
            2: {'num_rows': (2, 3), 'gaps_per_row': (1, 2)}, # 2-3 rows, 1-2 gaps
            3: {'num_rows': (4, 6), 'gaps_per_row': (1, 2)}, # 4-6 rows, 1-2 gaps
            4: {'num_rows': 0, 'gaps_per_row': 0}            # Empty board
        }
    
    def _get_stage(self, step):
        """Determine current curriculum stage based on training step."""
        if step < 15000:
            return 1
        elif step < 40000:
            return 2
        elif step < 80000:
            return 3
        else:
            return 4
    
    def reset(self, **kwargs):
        """Reset environment with curriculum-appropriate difficulty."""
        obs, info = self.env.reset(**kwargs)
        
        # Update stage
        old_stage = self.current_stage
        self.current_stage = self._get_stage(self.training_step)
        
        if old_stage != self.current_stage:
            print(f"\n{'='*70}")
            print(f"🎓 CURRICULUM STAGE TRANSITION: {old_stage} → {self.current_stage}")
            print(f"{'='*70}\n")
        
        # Apply curriculum difficulty
        if self.current_stage < 4:
            obs = self._create_curriculum_board(obs)
        
        return obs, info
    
    def _create_curriculum_board(self, obs):
        """
        Create a partially-filled board with SIMPLE curriculum.
        
        NEW Strategy:
        - Fill ONLY bottom N rows (not the whole board!)
        - Each row has exactly 1-2 gaps (never complete rows)
        - Stage 1: ONLY 1 row with EXACTLY 1 gap (easiest possible!)
        - Gaps are in random columns so agent learns to find them
        """
        board = obs['board'].copy()
        
        # Identify playable area (tetris-gymnasium uses padding=4)
        if board.shape == (24, 18):
            padding = 4
            play_height = 20
            play_width = 10
            row_start = 0
            row_end = play_height
            col_start = padding
            col_end = col_start + play_width
        else:
            play_height, play_width = board.shape
            row_start = 0
            row_end = play_height
            col_start = 0
            col_end = play_width
        
        # Get stage config
        config = self.stage_config[self.current_stage]
        
        # Stage 4 = empty board
        if config['num_rows'] == 0:
            return obs
        
        # Determine number of rows to fill
        if isinstance(config['num_rows'], tuple):
            num_rows = np.random.randint(config['num_rows'][0], config['num_rows'][1] + 1)
        else:
            num_rows = config['num_rows']
        
        # Fill bottom rows ONLY
        for row_offset in range(num_rows):
            row_idx = row_end - 1 - row_offset  # Start from bottom
            
            # Determine gaps for this row
            if isinstance(config['gaps_per_row'], tuple):
                num_gaps = np.random.randint(config['gaps_per_row'][0], config['gaps_per_row'][1] + 1)
            else:
                num_gaps = config['gaps_per_row']
            
            # Fill entire row first
            board[row_idx, col_start:col_end] = 1
            
            # Create gaps at random positions
            gap_positions = np.random.choice(play_width, size=num_gaps, replace=False)
            for gap_col in gap_positions:
                board[row_idx, col_start + gap_col] = 0
        
        obs['board'] = board
        return obs
    
    def update_training_step(self, step):
        """Update training step counter (call this from training loop)."""
        self.training_step = step
    
    def get_stage_info(self):
        """Get current curriculum stage information."""
        config = self.stage_config[self.current_stage]
        return {
            'stage': self.current_stage,
            'step': self.training_step,
            'next_transition': self.stage_transitions.get(self.current_stage, None),
            'config': config
        }


class ProgressiveCurriculumWrapper(gym.Wrapper):
    """
    Alternative: Progressive curriculum that smoothly increases difficulty
    based on agent performance rather than fixed step thresholds.
    """
    
    def __init__(self, env, initial_difficulty=0.0):
        super().__init__(env)
        self.difficulty = initial_difficulty  # 0.0 = easiest, 1.0 = hardest
        
        # Track performance for adaptive difficulty
        self.recent_rewards = []
        self.recent_lines = []
        self.window_size = 100
        
    def reset(self, **kwargs):
        """Reset with difficulty-based board generation."""
        obs, info = self.env.reset(**kwargs)
        
        if self.difficulty < 1.0:
            obs = self._create_difficulty_board(obs)
        
        return obs, info
    
    def _create_difficulty_board(self, obs):
        """
        Create board based on continuous difficulty parameter.
        
        difficulty = 0.0: 18-19 rows filled (easiest)
        difficulty = 0.5: 9-10 rows filled (medium)
        difficulty = 1.0: 0 rows filled (hardest)
        """
        board = obs['board'].copy()
        height, width = board.shape
        
        # Calculate number of rows to fill
        max_filled = 19
        num_filled_rows = int(max_filled * (1.0 - self.difficulty))
        
        if num_filled_rows > 0:
            # Fill bottom rows with strategic gaps
            for row_offset in range(num_filled_rows):
                row = height - 1 - row_offset
                
                # More gaps as difficulty increases
                num_gaps = 1 + int(self.difficulty * 2)  # 1-3 gaps
                gap_positions = np.random.choice(width, size=min(num_gaps, width), replace=False)
                
                board[row, :] = 1
                for gap_col in gap_positions:
                    board[row, gap_col] = 0
        
        obs['board'] = board
        return obs
    
    def update_difficulty(self, reward, lines_cleared):
        """
        Adaptively update difficulty based on performance.
        
        If agent is doing well (clearing lines), increase difficulty.
        If agent is struggling, decrease difficulty.
        """
        self.recent_rewards.append(reward)
        self.recent_lines.append(lines_cleared)
        
        # Keep only recent window
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)
            self.recent_lines.pop(0)
        
        # Update difficulty based on performance
        if len(self.recent_lines) >= 20:
            avg_lines = np.mean(self.recent_lines[-20:])
            
            # If clearing lines consistently, increase difficulty
            if avg_lines >= 2.0 and self.difficulty < 1.0:
                self.difficulty = min(1.0, self.difficulty + 0.05)
                print(f"📈 Difficulty increased to {self.difficulty:.2f}")
            
            # If not clearing any lines, slightly decrease
            elif avg_lines < 0.5 and self.difficulty > 0.0:
                self.difficulty = max(0.0, self.difficulty - 0.02)
                print(f"📉 Difficulty decreased to {self.difficulty:.2f}")

