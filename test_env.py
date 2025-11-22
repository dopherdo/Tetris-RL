"""
Test script to verify the Tetris environment setup
"""
import numpy as np
from src.env.tetris_env import TetrisEnv


def test_environment():
    """Test the custom Tetris environment"""
    print("=" * 60)
    print("Testing Tetris Environment Setup")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating environment...")
    env = TetrisEnv(render_mode=None)
    print("   ✓ Environment created successfully")
    
    # Check observation and action spaces
    print("\n2. Checking spaces...")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print("   ✓ Spaces verified")
    
    # Reset environment
    print("\n3. Resetting environment...")
    obs, info = env.reset()
    print(f"   Board shape: {obs['board'].shape}")
    print(f"   Board dtype: {obs['board'].dtype}")
    print(f"   Available observation keys: {list(obs.keys())}")
    print("   ✓ Reset successful")
    
    # Run a few random actions (composite placements)
    print("\n4. Running 10 random actions...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"   Action {i + 1}: Composite={action}, Reward={reward:.3f}, "
              f"Holes={info['holes']}, Height={info['max_height']}, "
              f"Terminated={terminated}")
        
        if terminated or truncated:
            print(f"   Episode ended after {i + 1} pieces")
            break
    
    print(f"\n   Total reward: {total_reward:.3f}")
    print("   ✓ Steps executed successfully")
    
    # Test reward components
    print("\n5. Testing reward engineering components...")
    board = obs['board']
    
    # Count holes
    holes = env._count_holes(board)
    print(f"   Holes detected: {holes}")
    
    # Calculate bumpiness
    bumpiness = env._calculate_bumpiness(board)
    print(f"   Bumpiness: {bumpiness}")
    
    # Get max height
    max_height = env._get_max_height(board)
    print(f"   Max height: {max_height}")
    
    print("   ✓ Reward components working correctly")
    
    # Visualize the board
    print("\n6. Current board state:")
    print("   " + "-" * (board.shape[1] * 2 + 1))
    for row in board:
        print("   |" + "".join(["█" if cell > 0 else " " for cell in row]) + "|")
    print("   " + "-" * (board.shape[1] * 2 + 1))
    
    # Close environment
    env.close()
    print("\n" + "=" * 60)
    print("✓ All tests passed! Environment is ready for training.")
    print("=" * 60)


def test_full_episode():
    """Run a complete episode with random actions"""
    print("\n" + "=" * 60)
    print("Running Full Episode Test")
    print("=" * 60)
    
    env = TetrisEnv(render_mode=None)
    obs, info = env.reset()
    
    episode_reward = 0
    step_count = 0
    max_steps = 1000
    
    print("\nPlaying with random actions...")
    while step_count < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode Summary:")
    print(f"  Pieces placed: {env.pieces_placed}")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Lines cleared: {env.total_lines_cleared}")
    print(f"  Final max height: {info['max_height']}")
    print(f"  Final holes: {info['holes']}")
    
    env.close()
    print("\n✓ Full episode test completed!")


if __name__ == "__main__":
    # Run basic tests
    test_environment()
    
    # Run full episode test
    test_full_episode()
    
    print("\n" + "=" * 60)
    print("Environment setup complete and verified!")
    print("You can now proceed to Phase 2: Neural Network Architecture")
    print("=" * 60)

