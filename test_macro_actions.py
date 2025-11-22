"""
Test script to verify macro-actions in TetrisEnv
"""
import sys
sys.path.append('/Users/ilansasson/Tetris-RL')

from src.env.tetris_env import TetrisEnv
import numpy as np


def test_macro_actions():
    """Test the macro-action functionality"""
    print("=" * 70)
    print("Testing Macro-Actions in TetrisEnv")
    print("=" * 70)
    
    # Test 1: Create environment with macro-actions
    print("\n1. Creating environment with macro-actions...")
    env_macro = TetrisEnv(render_mode=None, use_macro_actions=True)
    print(f"   Action space: {env_macro.action_space}")
    print(f"   Number of actions: {env_macro.action_space.n}")
    print(f"   Expected: {10 * 4} (10 columns × 4 rotations)")
    
    # Test 2: Create environment with atomic actions (for comparison)
    print("\n2. Creating environment with atomic actions...")
    env_atomic = TetrisEnv(render_mode=None, use_macro_actions=False)
    print(f"   Action space: {env_atomic.action_space}")
    print(f"   Number of actions: {env_atomic.action_space.n}")
    
    # Test 3: Run episode with macro-actions
    print("\n3. Running episode with macro-actions...")
    obs, info = env_macro.reset()
    
    episode_rewards = []
    episode_lines = []
    
    for step in range(20):
        # Sample random macro-action
        action = env_macro.action_space.sample()
        column = action % 10
        rotation = action // 10
        
        obs, reward, terminated, truncated, info = env_macro.step(action)
        
        episode_rewards.append(reward)
        episode_lines.append(info.get('lines_cleared', 0))
        
        if (step + 1) % 5 == 0:
            print(f"   Step {step+1}: Macro-action={action:2d} (col={column}, rot={rotation}), "
                  f"Reward={reward:6.2f}, Lines={info.get('lines_cleared', 0):2d}, "
                  f"Height={info['max_height']:2d}, Holes={info['holes']:2d}")
        
        if terminated or truncated:
            print(f"\n   Episode ended at step {step+1}")
            break
    
    # Statistics
    print(f"\n4. Macro-action episode statistics:")
    print(f"   Total steps: {len(episode_rewards)}")
    print(f"   Total reward: {sum(episode_rewards):.2f}")
    print(f"   Average reward per step: {np.mean(episode_rewards):.2f}")
    print(f"   Lines cleared: {info.get('lines_cleared', 0)}")
    print(f"   Total atomic actions used: {info.get('total_atomic_actions', 0)}")
    if len(episode_rewards) > 0:
        print(f"   Atomic actions per macro: {info.get('total_atomic_actions', 0) / len(episode_rewards):.1f}")
    
    # Test 4: Run episode with atomic actions (for comparison)
    print("\n5. Running episode with atomic actions (for comparison)...")
    obs, info = env_atomic.reset()
    
    atomic_rewards = []
    atomic_pieces_placed = 0
    
    for step in range(100):  # More steps needed for atomic
        action = env_atomic.action_space.sample()
        obs, reward, terminated, truncated, info = env_atomic.step(action)
        
        atomic_rewards.append(reward)
        
        if terminated or truncated:
            print(f"   Episode ended at step {step+1}")
            break
    
    print(f"\n6. Atomic action episode statistics:")
    print(f"   Total steps: {len(atomic_rewards)}")
    print(f"   Total reward: {sum(atomic_rewards):.2f}")
    print(f"   Average reward per step: {np.mean(atomic_rewards):.2f}")
    print(f"   Lines cleared: {info.get('lines_cleared', 0)}")
    
    # Test 5: Sample action meanings
    print(f"\n7. Sample macro-action meanings:")
    meanings = env_macro.get_action_meanings()
    for i in [0, 9, 10, 19, 39]:
        if i < len(meanings):
            print(f"   Action {i:2d}: {meanings[i]}")
    
    env_macro.close()
    env_atomic.close()
    
    print(f"\n" + "=" * 70)
    print("✓ Macro-action integration test complete!")
    print("\nKey Benefits of Macro-Actions:")
    print("  1. Better credit assignment (one action = one outcome)")
    print("  2. Shorter episodes (faster learning)")
    print("  3. Denser rewards (every action meaningful)")
    print("  4. More stable for PPO training")
    print("=" * 70)


if __name__ == "__main__":
    test_macro_actions()

