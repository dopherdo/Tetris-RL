"""
Overnight training run for Tetris DQN with composite actions.
Optimized for maximum chance of discovering line clears.
"""

import os
import torch
from tqdm import tqdm
from pathlib import Path
from src.env import make_tetris_env
from src.models import DQNAgent, DQNConfig
from src.utils import preprocess_observation
import numpy as np


def evaluate(env, agent, episodes=10):
    """Quick evaluation."""
    total_rewards = []
    total_lines = []
    total_pieces = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        obs_proc = preprocess_observation(obs)
        done = False
        ep_reward = 0.0
        pieces = 0
        
        while not done and pieces < 100:
            action = agent.select_action(obs_proc, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_proc = preprocess_observation(obs)
            ep_reward += reward
            pieces += 1
            done = terminated or truncated
        
        total_rewards.append(ep_reward)
        total_lines.append(info.get('lines_cleared', 0))
        total_pieces.append(pieces)
    
    return np.mean(total_rewards), np.mean(total_lines), np.mean(total_pieces)


def train_overnight(total_steps=200000):
    """
    Overnight training with optimized settings.
    
    Key optimizations:
    - Longer epsilon decay for more exploration
    - Save checkpoints every 25K steps
    - Log results to file
    - Evaluate frequently to catch improvements
    """
    
    # Setup logging
    log_file = Path('training_overnight.log')
    
    def log(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
    
    # Create environment
    env = make_tetris_env(render_mode=None, use_composite_actions=True)
    obs, _ = env.reset()
    board_shape = preprocess_observation(obs).shape
    n_actions = env.action_space.n
    
    log('='*60)
    log('OVERNIGHT TRAINING: Optimized for Line Clearing Discovery')
    log('='*60)
    log(f'Board shape: {board_shape}')
    log(f'Action space: {n_actions} composite actions')
    log(f'Total steps: {total_steps:,}')
    log('')
    
    # Optimized DQN config for overnight run
    config = DQNConfig(
        gamma=0.99,
        learning_rate=1e-4,
        buffer_size=50000,  # Larger buffer
        batch_size=64,
        target_update_freq=1000,
        epsilon_start=1.0,
        epsilon_end=0.05,  # Higher min epsilon (more exploration)
        epsilon_decay_steps=int(total_steps * 0.7),  # Decay over 70% of training
        per_alpha=0.6,
        per_beta_start=0.4,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')
    log('')
    
    agent = DQNAgent(board_shape, n_actions, device, config)
    
    # Tracking
    episode_rewards = []
    episode_lines = []
    episode_lengths = []
    best_lines = 0
    
    checkpoint_freq = 25000
    eval_freq = 10000
    warmup = 5000
    
    # Initialize
    obs, _ = env.reset()
    obs_proc = preprocess_observation(obs)
    episode_reward = 0.0
    episode_length = 0
    
    # Training loop
    log('Starting training...')
    log('')
    pbar = tqdm(total=total_steps, desc="Training")
    
    for step in range(total_steps):
        # Select action
        action = agent.select_action(obs_proc, eval_mode=False)
        
        # Execute
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs_proc = preprocess_observation(next_obs)
        done = terminated or truncated
        
        # Store
        agent.store_transition(obs_proc, action, reward, next_obs_proc, done)
        
        # Update metrics
        episode_reward += reward
        episode_length += 1
        obs_proc = next_obs_proc
        
        # Train
        if step >= warmup:
            metrics = agent.train_step()
            if metrics:
                pbar.set_postfix({
                    'eps': f"{agent.epsilon:.3f}",
                    'loss': f"{metrics['loss']:.2f}",
                    'Q': f"{metrics['q_mean']:.1f}",
                })
        
        # Episode end
        if done:
            episode_rewards.append(episode_reward)
            episode_lines.append(info.get('lines_cleared', 0))
            episode_lengths.append(episode_length)
            
            # Check for best performance
            if info.get('lines_cleared', 0) > best_lines:
                best_lines = info.get('lines_cleared', 0)
                if best_lines > 0:
                    log(f'\\n🎉 NEW BEST! Step {step+1}: {best_lines} lines cleared!')
                    # Save best model
                    os.makedirs('models/best', exist_ok=True)
                    agent.save(f'models/best/best_{best_lines}lines.pt')
            
            obs, _ = env.reset()
            obs_proc = preprocess_observation(obs)
            episode_reward = 0.0
            episode_length = 0
        
        # Evaluation
        if (step + 1) % eval_freq == 0:
            eval_reward, eval_lines, eval_pieces = evaluate(env, agent, episodes=10)
            msg = f'\\nStep {step+1:,}: Reward={eval_reward:.1f}, Lines={eval_lines:.2f}, Pieces={eval_pieces:.1f}'
            log(msg)
            
            if eval_lines > 0:
                log(f'  🎉 LINES CLEARED IN EVALUATION!')
        
        # Checkpoint
        if (step + 1) % checkpoint_freq == 0:
            os.makedirs('models/overnight', exist_ok=True)
            agent.save(f'models/overnight/checkpoint_{(step+1)//1000}k.pt')
            log(f'\\n💾 Checkpoint saved at {step+1:,} steps')
        
        pbar.update(1)
    
    pbar.close()
    
    # Final save
    os.makedirs('models/overnight', exist_ok=True)
    agent.save('models/overnight/final.pt')
    
    # Final evaluation
    log('\\n' + '='*60)
    log('OVERNIGHT TRAINING COMPLETE')
    log('='*60)
    
    final_reward, final_lines, final_pieces = evaluate(env, agent, episodes=30)
    log(f'Final evaluation (30 episodes):')
    log(f'  Avg reward: {final_reward:.1f}')
    log(f'  Avg lines: {final_lines:.2f}')
    log(f'  Avg pieces: {final_pieces:.1f}')
    log(f'  Best lines in training: {best_lines}')
    
    if episode_lines:
        total_lines = sum(episode_lines)
        log(f'\\nTraining statistics:')
        log(f'  Total episodes: {len(episode_rewards)}')
        log(f'  Total lines cleared: {total_lines}')
        log(f'  Avg episode length: {np.mean(episode_lengths):.1f} pieces')
    
    if final_lines > 0 or best_lines > 0:
        log('\\n🎉🎉🎉 SUCCESS! Agent learned to clear lines! 🎉🎉🎉')
    else:
        log('\\n⚠️  No lines cleared. Recommend curriculum learning next.')
    
    log('='*60)
    log(f'\\nLogs saved to: {log_file}')
    log(f'Models saved to: models/overnight/')
    
    env.close()
    return agent


if __name__ == '__main__':
    import sys
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    
    print(f'Starting overnight run: {steps:,} steps')
    print('This will take several hours...')
    print('')
    
    train_overnight(steps)

