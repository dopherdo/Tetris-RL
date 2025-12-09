"""
Tetris Environment Module
"""
from .tetris_env import TetrisEnv
from .composite_wrapper import CompositeActionWrapper, make_composite_tetris_env


def make_tetris_env(render_mode=None, use_composite_actions=False, **kwargs):
    """
    Factory function to create a Tetris environment.
    
    Args:
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        use_composite_actions: If True, use composite action space (40 actions)
                               If False, use atomic actions (8 actions)
        **kwargs: Additional arguments passed to TetrisEnv
    
    Returns:
        TetrisEnv instance (optionally wrapped with CompositeActionWrapper)
    """
    if use_composite_actions:
        return make_composite_tetris_env(render_mode=render_mode, **kwargs)
    else:
        return TetrisEnv(render_mode=render_mode, **kwargs)


__all__ = ['TetrisEnv', 'CompositeActionWrapper', 'make_tetris_env', 'make_composite_tetris_env']
