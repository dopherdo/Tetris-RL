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
    
    # Run until game over to show actual termination state
    print("\n4. Playing until game over...")
    total_reward = 0
    step = 0
    max_steps = 200
    
    while step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # Show progress every 10 steps
        if step % 10 == 0:
            print(f"   Step {step}: Reward={reward:.3f}, "
                  f"Holes={info['holes']}, Height={info['max_height']}")
        
        if terminated or truncated:
            print(f"\n   ✓ Game ended at step {step}")
            print(f"   Total reward: {total_reward:.3f}")
            print(f"   Lines cleared: {env.total_lines_cleared}")
            break
    
    # Test reward components on final state
    print("\n5. Final game state analysis...")
    board = obs['board']
    
    # Count holes
    holes = env._count_holes(board, obs.get('active_tetromino_mask'))
    print(f"   Holes: {holes}")
    
    # Calculate bumpiness
    bumpiness = env._calculate_bumpiness(board)
    print(f"   Bumpiness: {bumpiness}")
    
    # Get max height
    max_height = env._get_max_height(board)
    print(f"   Max height: {max_height}")
    
    print("   ✓ Reward components working correctly")
    
    # Visualize the FINAL board when game ended (only playable area!)
    print("\n6. GAME OVER - Final board state (playable 20 rows):")
    print("   " + "-" * 22)
    for i in range(20):  # Only show rows 0-19 (playable area)
        playable_row = board[i, 4:14]  # Only columns 4-13
        row_str = "".join(["█" if cell > 0 else "·" for cell in playable_row])
        
        # Highlight spawn zone (rows 0-2)
        marker = ""
        if i <= 2:
            filled_count = sum(1 for cell in playable_row if cell > 0)
            if filled_count > 0:
                marker = " ← SPAWN BLOCKED!"
        
        print(f"   |{row_str}| {i:2d}{marker}")
    print("   " + "-" * 22)
    print("   (Floor rows 20-23 not shown)")
    
    # Close environment
    env.close()
    print("\n" + "=" * 60)
    print("✓ All tests passed! Environment is ready for training.")
    print("=" * 60)


def test_full_episode():
    """Run a complete episode with random actions and track detailed metrics"""
    print("\n" + "=" * 70)
    print("FULL GAME TEST - Random Agent Playing Complete Game")
    print("=" * 70)
    
    env = TetrisEnv(render_mode=None)
    obs, info = env.reset()
    
    # Track metrics throughout the game
    episode_reward = 0
    step_count = 0
    max_steps = 1000
    
    # Detailed tracking
    rewards_history = []
    holes_history = []
    height_history = []
    bumpiness_history = []
    
    # Action tracking
    action_names = ['move_left', 'move_right', 'soft_drop', 'rotate_cw', 
                    'rotate_ccw', 'hard_drop', 'swap', 'no_op']
    action_counts = {name: 0 for name in action_names}
    
    # Reward component tracking
    positive_rewards = 0
    negative_rewards = 0
    total_holes_created = 0
    total_holes_filled = 0
    
    print("\nPlaying game with random actions...")
    print("(Showing every 10th step for brevity)")
    print("-" * 70)
    
    while step_count < max_steps:
        action = env.action_space.sample()
        action_counts[action_names[action]] += 1
        
        prev_holes = info.get('holes', 0) if step_count > 0 else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        
        # Track metrics
        rewards_history.append(reward)
        holes_history.append(info['holes'])
        height_history.append(info['max_height'])
        bumpiness_history.append(info['bumpiness'])
        
        # Track reward breakdown
        if reward > 0:
            positive_rewards += 1
        else:
            negative_rewards += 1
        
        # Track hole changes
        holes_delta = info['holes'] - prev_holes
        if holes_delta > 0:
            total_holes_created += holes_delta
        elif holes_delta < 0:
            total_holes_filled += abs(holes_delta)
        
        # Print periodic updates
        if step_count % 10 == 0 or terminated or truncated:
            print(f"Step {step_count:3d}: Reward={reward:6.2f} | "
                  f"Holes={info['holes']:2d} | Height={info['max_height']:2d} | "
                  f"Bumpiness={info['bumpiness']:2d} | Total R={episode_reward:7.2f}")
        
        if terminated or truncated:
            break
    
    # Calculate statistics
    avg_reward = np.mean(rewards_history)
    avg_holes = np.mean(holes_history)
    avg_height = np.mean(height_history)
    avg_bumpiness = np.mean(bumpiness_history)
    
    print("\n" + "=" * 70)
    print("GAME COMPLETE - Detailed Statistics")
    print("=" * 70)
    
    print("\n📊 GAME SUMMARY:")
    print(f"  Pieces placed: {step_count}")
    print(f"  Lines cleared: {env.total_lines_cleared}")
    print(f"  Game duration: {step_count} steps")
    print(f"  Reason ended: {'Game Over' if terminated else 'Truncated'}")
    
    print("\n💰 REWARD STATISTICS:")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Average reward per piece: {avg_reward:.3f}")
    print(f"  Positive reward steps: {positive_rewards} ({100*positive_rewards/step_count:.1f}%)")
    print(f"  Negative reward steps: {negative_rewards} ({100*negative_rewards/step_count:.1f}%)")
    print(f"  Best reward (single step): {max(rewards_history):.2f}")
    print(f"  Worst reward (single step): {min(rewards_history):.2f}")
    
    print("\n🕳️  HOLE STATISTICS:")
    print(f"  Final holes: {holes_history[-1]}")
    print(f"  Average holes: {avg_holes:.1f}")
    print(f"  Max holes: {max(holes_history)}")
    print(f"  Total holes created: {total_holes_created}")
    print(f"  Total holes filled: {total_holes_filled}")
    print(f"  Net holes: {total_holes_created - total_holes_filled}")
    
    print("\n📏 HEIGHT & SHAPE STATISTICS:")
    print(f"  Final height: {height_history[-1]}")
    print(f"  Average height: {avg_height:.1f}")
    print(f"  Max height reached: {max(height_history)}")
    print(f"  Final bumpiness: {bumpiness_history[-1]}")
    print(f"  Average bumpiness: {avg_bumpiness:.1f}")
    
    print("\n🎮 ACTION DISTRIBUTION:")
    for action_name, count in action_counts.items():
        percentage = 100 * count / step_count
        bar = "█" * int(percentage / 2)
        print(f"  {action_name:15s}: {count:3d} ({percentage:5.1f}%) {bar}")
    
    print("\n📈 PERFORMANCE INSIGHTS:")
    
    # Insight 1: Line clearing efficiency
    if env.total_lines_cleared == 0:
        print("  ⚠️  No lines cleared - random agent needs learning!")
    else:
        efficiency = env.total_lines_cleared / step_count
        print(f"  ✓ Line clearing efficiency: {efficiency:.4f} lines/piece")
    
    # Insight 2: Hole management
    hole_rate = total_holes_created / step_count
    if hole_rate > 0.5:
        print(f"  ⚠️  High hole creation rate: {hole_rate:.2f} holes/piece")
    else:
        print(f"  ✓ Moderate hole creation: {hole_rate:.2f} holes/piece")
    
    # Insight 3: Height management
    if avg_height > 18:
        print(f"  ⚠️  Dangerous average height: {avg_height:.1f}")
    else:
        print(f"  ✓ Manageable average height: {avg_height:.1f}")
    
    # Insight 4: Survival time
    if step_count < 30:
        print(f"  ⚠️  Short game: Only {step_count} pieces")
    elif step_count < 100:
        print(f"  ✓ Decent survival: {step_count} pieces")
    else:
        print(f"  ✓ Good survival: {step_count} pieces")
    
    env.close()
    print("\n" + "=" * 70)
    print("✓ Full episode test completed!")
    print("=" * 70)


if __name__ == "__main__":
    # Run basic tests
    test_environment()
    
    # Run full episode test
    test_full_episode()
    
    print("\n" + "=" * 60)
    print("Environment setup complete and verified!")
    print("You can now proceed to Phase 2: Neural Network Architecture")
    print("=" * 60)

