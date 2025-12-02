"""
Quick training script for composite action Tetris DQN.
"""

import torch
from tqdm import tqdm
from src.env import make_tetris_env
from src.models import DQNAgent, DQNConfig
from src.utils import preprocess_observation

def train_composite(steps=5000, eval_freq=2500):
    """Train DQN with composite actions."""
    
    # Create composite action environment
    env = make_tetris_env(render_mode=None, use_composite_actions=True)
    
    # Setup
    obs, _ = env.reset()
    board_shape = preprocess_observation(obs).shape
    n_actions = env.action_space.n
    
    warmup = min(1000, steps // 5)
    print('='*60)
    print('Training DQN with COMPOSITE ACTIONS')
    print('='*60)
    print(f'Board shape: {board_shape}')
    print(f'Action space: {n_actions} composite actions')
    print(f'Training steps: {steps}')
    print(f'Warmup steps: {warmup}')
    print(f'Evaluation frequency: every {eval_freq} steps')
    print('='*60)
    print()
    
    # Create agent
    warmup = min(1000, steps // 5)
    config = DQNConfig(
        buffer_size=20000,
        batch_size=64,
        epsilon_decay_steps=int(steps * 0.6),  # Decay over 60% of training
    )
    device = torch.device('cpu')
    agent = DQNAgent(board_shape, n_actions, device, config)
    
    # Training metrics
    episode_rewards = []
    episode_lines = []
    episode_lengths = []
    
    # Initialize
    obs, _ = env.reset()
    obs_proc = preprocess_observation(obs)
    episode_reward = 0.0
    episode_length = 0
    
    # Training loop
    pbar = tqdm(total=steps, desc="Training")
    
    for step in range(steps):
        # Select action
        action = agent.select_action(obs_proc, eval_mode=False)
        
        # Execute action (one composite action = one piece placement)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs_proc = preprocess_observation(next_obs)
        done = terminated or truncated
        
        # Store transition
        agent.store_transition(obs_proc, action, reward, next_obs_proc, done)
        
        # Update metrics
        episode_reward += reward
        episode_length += 1
        obs_proc = next_obs_proc
        
        # Train agent (after warmup)
        if step >= warmup:
            metrics = agent.train_step()
            if metrics:
                pbar.set_postfix({
                    'eps': f"{agent.epsilon:.3f}",
                    'loss': f"{metrics['loss']:.4f}",
                    'Q': f"{metrics['q_mean']:.2f}",
                })
        
        # Handle episode end
        if done:
            episode_rewards.append(episode_reward)
            episode_lines.append(info.get('lines_cleared', 0))
            episode_lengths.append(episode_length)
            
            # Reset
            obs, _ = env.reset()
            obs_proc = preprocess_observation(obs)
            episode_reward = 0.0
            episode_length = 0
        
        # Evaluation
        if (step + 1) % eval_freq == 0 or step == steps - 1:
            eval_reward, eval_lines, eval_pieces = evaluate(env, agent, episodes=5)
            print(f"\nStep {step + 1}: Eval Reward={eval_reward:.2f}, Lines={eval_lines:.1f}, Pieces={eval_pieces:.1f}")
        
        pbar.update(1)
    
    pbar.close()
    
    # Final stats
    print('\n' + '='*60)
    print('TRAINING COMPLETE')
    print('='*60)
    if episode_rewards:
        print(f'Episodes completed: {len(episode_rewards)}')
        print(f'Avg episode reward: {sum(episode_rewards)/len(episode_rewards):.2f}')
        print(f'Avg episode length: {sum(episode_lengths)/len(episode_lengths):.1f} pieces')
        print(f'Total lines cleared: {sum(episode_lines)}')
    
    env.close()
    return agent


def evaluate(env, agent, episodes=5):
    """Evaluate agent performance."""
    total_rewards = []
    total_lines = []
    total_pieces = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        obs_proc = preprocess_observation(obs)
        done = False
        ep_reward = 0.0
        pieces = 0
        
        while not done and pieces < 100:  # Cap at 100 pieces
            action = agent.select_action(obs_proc, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_proc = preprocess_observation(obs)
            ep_reward += reward
            pieces += 1
            done = terminated or truncated
        
        total_rewards.append(ep_reward)
        total_lines.append(info.get('lines_cleared', 0))
        total_pieces.append(pieces)
    
    import numpy as np
    return np.mean(total_rewards), np.mean(total_lines), np.mean(total_pieces)


if __name__ == '__main__':
    import sys
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    eval_freq = steps // 4  # Evaluate 4 times during training
    
    agent = train_composite(steps=steps, eval_freq=eval_freq)
    
    # Save the trained agent
    import os
    os.makedirs('models/composite', exist_ok=True)
    agent.save(f'models/composite/dqn_composite_{steps//1000}k.pt')
    print(f'\n✓ Model saved to models/composite/dqn_composite_{steps//1000}k.pt')

