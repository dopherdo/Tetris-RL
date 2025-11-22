"""
Integration Test for Phase 2: CNN Architecture
Tests the complete flow from environment to CNN policy
"""
import sys
sys.path.append('/Users/ilansasson/Tetris-RL')

import torch
import numpy as np
from src.env.tetris_env import TetrisEnv
from src.utils.preprocessing import preprocess_observation, get_board_features
from src.models.cnn_policy import CNNPolicy, count_parameters


def test_single_episode():
    """
    Test a complete episode with CNN policy
    """
    print("\n" + "="*70)
    print("TEST 1: Single Episode with CNN Policy")
    print("="*70)
    
    # Create environment
    env = TetrisEnv(render_mode=None)
    obs, info = env.reset()
    
    # Create CNN policy
    action_dim = env.action_space.n
    model = CNNPolicy(action_dim=action_dim, hidden_dim=256)
    model.eval()  # Set to evaluation mode
    
    print(f"\n✓ Environment created: {env.action_space.n} actions")
    print(f"✓ CNN Policy created: {count_parameters(model):,} parameters")
    
    # Run episode
    episode_rewards = []
    episode_actions = []
    episode_values = []
    max_steps = 100
    
    print(f"\nRunning episode (max {max_steps} steps)...")
    
    for step in range(max_steps):
        # Preprocess observation
        board_tensor = preprocess_observation(obs).unsqueeze(0)
        
        # Get action from policy
        with torch.no_grad():
            action, log_prob, value = model.get_action(board_tensor, deterministic=False)
        
        action_int = action.item()
        value_float = value.item()
        
        # Take action in environment
        obs, reward, terminated, truncated, info = env.step(action_int)
        
        # Store data
        episode_rewards.append(reward)
        episode_actions.append(action_int)
        episode_values.append(value_float)
        
        # Print progress every 20 steps
        if (step + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_value = np.mean(episode_values[-20:])
            print(f"  Step {step+1:3d}: Avg Reward={avg_reward:6.2f}, Avg Value={avg_value:6.2f}, "
                  f"Lines={info.get('lines_cleared', 0):2d}, Height={info['max_height']:2d}")
        
        if terminated or truncated:
            print(f"\n  Episode ended at step {step+1}")
            break
    
    # Print statistics
    print(f"\nEpisode Statistics:")
    print(f"  Total steps: {len(episode_rewards)}")
    print(f"  Total reward: {sum(episode_rewards):.2f}")
    print(f"  Average reward: {np.mean(episode_rewards):.2f}")
    print(f"  Lines cleared: {info.get('lines_cleared', 0)}")
    print(f"  Final height: {info['max_height']}")
    print(f"  Final holes: {info['holes']}")
    
    env.close()
    
    print(f"\n✓ Single episode test passed!")
    return True


def test_batch_processing():
    """
    Test batch processing of multiple states
    """
    print("\n" + "="*70)
    print("TEST 2: Batch Processing")
    print("="*70)
    
    # Create environment and model
    env = TetrisEnv(render_mode=None)
    model = CNNPolicy(action_dim=env.action_space.n, hidden_dim=256)
    model.eval()
    
    # Collect batch of observations
    batch_size = 8
    observations = []
    
    print(f"\nCollecting {batch_size} observations...")
    obs, info = env.reset()
    
    for i in range(batch_size):
        observations.append(obs)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    # Preprocess batch
    print(f"Preprocessing batch...")
    batch_tensors = []
    for obs in observations:
        tensor = preprocess_observation(obs)
        batch_tensors.append(tensor)
    
    batch = torch.stack(batch_tensors, dim=0)
    print(f"  Batch shape: {batch.shape}")
    
    # Process batch through model
    print(f"Processing batch through CNN...")
    with torch.no_grad():
        action_logits, values = model(batch)
        actions, log_probs, values = model.get_action(batch, deterministic=False)
    
    print(f"  Action logits shape: {action_logits.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Log probs shape: {log_probs.shape}")
    
    # Verify shapes
    assert action_logits.shape == (batch_size, env.action_space.n), "Action logits shape mismatch"
    assert values.shape == (batch_size, 1), "Values shape mismatch"
    assert actions.shape == (batch_size,), "Actions shape mismatch"
    assert log_probs.shape == (batch_size,), "Log probs shape mismatch"
    
    print(f"\n  Sample actions: {actions.tolist()}")
    print(f"  Sample values: {[f'{v.item():.2f}' for v in values[:4]]}")
    
    env.close()
    
    print(f"\n✓ Batch processing test passed!")
    return True


def test_gradient_flow():
    """
    Test that gradients flow correctly through the network
    """
    print("\n" + "="*70)
    print("TEST 3: Gradient Flow")
    print("="*70)
    
    # Create model and dummy data
    model = CNNPolicy(action_dim=8, hidden_dim=256)
    model.train()  # Set to training mode
    
    print(f"\nCreating dummy batch...")
    batch_size = 4
    dummy_states = torch.rand(batch_size, 1, 20, 10)
    dummy_actions = torch.randint(0, 8, (batch_size,))
    
    print(f"  States shape: {dummy_states.shape}")
    print(f"  Actions shape: {dummy_actions.shape}")
    
    # Forward pass
    print(f"\nForward pass...")
    action_logits, values = model(dummy_states)
    
    # Compute simple loss (for testing gradients)
    print(f"Computing loss...")
    policy_loss = torch.nn.functional.cross_entropy(action_logits, dummy_actions)
    value_loss = values.mean()
    total_loss = policy_loss + value_loss
    
    print(f"  Policy loss: {policy_loss.item():.4f}")
    print(f"  Value loss: {value_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    
    # Backward pass
    print(f"\nBackward pass...")
    total_loss.backward()
    
    # Check gradients
    print(f"Checking gradients...")
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if 'conv' in name or 'fc' in name:
                print(f"  {name:30s}: grad_norm = {grad_norm:.6f}")
    
    avg_grad_norm = np.mean(grad_norms)
    print(f"\n  Average gradient norm: {avg_grad_norm:.6f}")
    
    # Verify gradients exist
    assert len(grad_norms) > 0, "No gradients computed!"
    assert avg_grad_norm > 0, "Gradients are zero!"
    
    print(f"\n✓ Gradient flow test passed!")
    return True


def test_action_distribution():
    """
    Test that action distributions are reasonable
    """
    print("\n" + "="*70)
    print("TEST 4: Action Distribution Analysis")
    print("="*70)
    
    # Create environment and model
    env = TetrisEnv(render_mode=None)
    model = CNNPolicy(action_dim=env.action_space.n, hidden_dim=256)
    model.eval()
    
    # Collect action distribution statistics
    num_samples = 100
    action_counts = np.zeros(env.action_space.n)
    
    print(f"\nSampling {num_samples} actions from random states...")
    
    obs, info = env.reset()
    for i in range(num_samples):
        # Preprocess and get action
        board_tensor = preprocess_observation(obs).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value = model.get_action(board_tensor, deterministic=False)
        
        action_int = action.item()
        action_counts[action_int] += 1
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action_int)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    # Analyze distribution
    action_probs = action_counts / num_samples
    
    print(f"\nAction distribution (from untrained network):")
    for i, prob in enumerate(action_probs):
        bar = '█' * int(prob * 50)
        print(f"  Action {i}: {prob:.3f} {bar}")
    
    # Check that distribution is not too skewed
    max_prob = action_probs.max()
    min_prob = action_probs.min()
    uniform_prob = 1.0 / env.action_space.n
    
    print(f"\nDistribution analysis:")
    print(f"  Max probability: {max_prob:.3f}")
    print(f"  Min probability: {min_prob:.3f}")
    print(f"  Uniform probability: {uniform_prob:.3f}")
    print(f"  Entropy: {-np.sum(action_probs * np.log(action_probs + 1e-10)):.3f}")
    print(f"  Max entropy: {np.log(env.action_space.n):.3f}")
    
    # For untrained network, we expect reasonable exploration
    # (not too deterministic, not too uniform)
    # With 40 actions, it's normal that some actions aren't sampled in 100 trials
    assert max_prob < 0.9, "Distribution too deterministic!"
    # Relax the min_prob check for larger action spaces
    non_zero_actions = np.sum(action_probs > 0)
    print(f"  Non-zero actions: {non_zero_actions}/{env.action_space.n}")
    assert non_zero_actions > env.action_space.n * 0.5, f"Too few actions explored! Only {non_zero_actions}/{env.action_space.n}"
    
    env.close()
    
    print(f"\n✓ Action distribution test passed!")
    return True


def test_value_prediction():
    """
    Test value prediction consistency
    """
    print("\n" + "="*70)
    print("TEST 5: Value Prediction Consistency")
    print("="*70)
    
    # Create environment and model
    env = TetrisEnv(render_mode=None)
    model = CNNPolicy(action_dim=env.action_space.n, hidden_dim=256)
    model.eval()
    
    print(f"\nCollecting value predictions across episode...")
    
    obs, info = env.reset()
    values = []
    heights = []
    holes = []
    
    for step in range(50):
        # Get value prediction
        board_tensor = preprocess_observation(obs).unsqueeze(0)
        
        with torch.no_grad():
            value = model.get_value(board_tensor)
        
        values.append(value.item())
        
        # Get board features
        features = get_board_features(obs)
        heights.append(features['max_height'])
        holes.append(features['holes'])
        
        # Take action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    # Analyze value predictions
    values = np.array(values)
    heights = np.array(heights)
    holes = np.array(holes)
    
    print(f"\nValue statistics (untrained network):")
    print(f"  Mean value: {values.mean():.3f}")
    print(f"  Std value: {values.std():.3f}")
    print(f"  Min value: {values.min():.3f}")
    print(f"  Max value: {values.max():.3f}")
    
    print(f"\nBoard statistics:")
    print(f"  Mean height: {heights.mean():.1f}")
    print(f"  Mean holes: {holes.mean():.1f}")
    
    # Check that values are not constant (some variance)
    assert values.std() > 0.01, "Values are too constant!"
    
    env.close()
    
    print(f"\n✓ Value prediction test passed!")
    return True


def main():
    """
    Run all integration tests
    """
    print("\n" + "="*70)
    print("PHASE 2 INTEGRATION TESTS: CNN Architecture")
    print("="*70)
    print("\nTesting complete pipeline:")
    print("  1. Environment observation → Preprocessing → Tensor")
    print("  2. Tensor → CNN Policy → Actions + Values")
    print("  3. Actions → Environment → New observation")
    print("\nThis validates that Phase 2 integrates correctly with Phase 1.")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run all tests
    tests = [
        ("Single Episode", test_single_episode),
        ("Batch Processing", test_batch_processing),
        ("Gradient Flow", test_gradient_flow),
        ("Action Distribution", test_action_distribution),
        ("Value Prediction", test_value_prediction),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED: {e}")
            results.append((test_name, False))
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name:30s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL INTEGRATION TESTS PASSED!")
        print("\nPhase 2 (CNN Architecture) is complete and integrated!")
        print("\nNext steps:")
        print("  - Phase 3: Implement PPO algorithm (ppo_agent.py)")
        print("  - Phase 4: Training loop and evaluation")
    else:
        print("✗ SOME TESTS FAILED - Please review errors above")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

