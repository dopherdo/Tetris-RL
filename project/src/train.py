"""Main DQN training loop for Tetris."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm, trange

from .env import make_tetris_env
from .models import DQNAgent, DQNConfig
from .utils import preprocess_observation
from .utils.visualization import plot_learning_curve


def infer_board_shape(env: gym.Env) -> Tuple[int, int]:
    """Infer the board shape from environment observation space."""
    obs, _ = env.reset()
    processed = preprocess_observation(obs)
    return processed.shape


def train_dqn(
    total_steps: int = 500_000,
    warmup_steps: int = 10_000,
    eval_frequency: int = 10_000,
    eval_episodes: int = 5,
    model_dir: str = "models/checkpoints",
    config: DQNConfig | None = None,
) -> None:
    """Train DQN agent on Tetris."""
    env = make_tetris_env(render_mode=None)
    n_actions = env.action_space.n
    board_shape = infer_board_shape(env)
    
    print(f"Board shape: {board_shape}")
    print(f"Action space: {n_actions}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = DQNAgent(
        board_shape=board_shape,
        n_actions=n_actions,
        device=device,
        config=config,
    )
    
    save_root = Path(model_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    episode_rewards = []
    episode_lengths = []
    training_losses = []
    step_rewards = []
    
    obs, _ = env.reset()
    obs_proc = preprocess_observation(obs)
    episode_reward = 0.0
    episode_length = 0
    
    pbar = tqdm(total=total_steps, desc="Training")
    
    for step in range(total_steps):
        action = agent.select_action(obs_proc, eval_mode=False)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs_proc = preprocess_observation(next_obs)
        done = terminated or truncated
        
        agent.store_transition(obs_proc, action, reward, next_obs_proc, done)
        
        episode_reward += reward
        episode_length += 1
        step_rewards.append(reward)
        
        obs_proc = next_obs_proc
        
        if step >= warmup_steps:
            metrics = agent.train_step()
            if metrics:
                training_losses.append(metrics['loss'])
                
                pbar.set_postfix({
                    'eps': f"{agent.epsilon:.3f}",
                    'loss': f"{metrics['loss']:.4f}",
                    'Q_mean': f"{metrics['q_mean']:.2f}",
                })
        
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            obs, _ = env.reset()
            obs_proc = preprocess_observation(obs)
            episode_reward = 0.0
            episode_length = 0
        
        if (step + 1) % eval_frequency == 0:
            eval_reward, eval_lines = evaluate_agent(env, agent, eval_episodes)
            print(f"\nStep {step + 1}: Eval Reward = {eval_reward:.2f}, Lines = {eval_lines:.1f}")
            
            ckpt_path = save_root / f"dqn_tetris_step{step+1}.pt"
            agent.save(str(ckpt_path))
            print(f"Saved checkpoint: {ckpt_path}")
        
        pbar.update(1)
    
    pbar.close()
    
    final_path = save_root / "dqn_tetris_final.pt"
    agent.save(str(final_path))
    print(f"\nTraining complete! Final model saved: {final_path}")
    
    if episode_rewards:
        plot_learning_curve(episode_rewards, save_path=plot_dir / "reward_curve.png")
        print(f"Saved reward curve: {plot_dir / 'reward_curve.png'}")
    
    env.close()


def evaluate_agent(
    env: gym.Env,
    agent: DQNAgent,
    num_episodes: int = 5,
) -> Tuple[float, float]:
    """Evaluate agent performance."""
    total_rewards = []
    total_lines = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        obs_proc = preprocess_observation(obs)
        done = False
        ep_reward = 0.0
        
        while not done:
            action = agent.select_action(obs_proc, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_proc = preprocess_observation(obs)
            ep_reward += reward
            done = terminated or truncated
        
        total_rewards.append(ep_reward)
        total_lines.append(info.get('lines_cleared', 0))
    
    return np.mean(total_rewards), np.mean(total_lines)


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DQN agent on Tetris")
    parser.add_argument("--steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--warmup", type=int, default=10_000, help="Warmup steps before training")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluation frequency")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size")
    parser.add_argument("--target-update", type=int, default=1_000, help="Target network update frequency")
    parser.add_argument("--epsilon-decay", type=int, default=10_000, help="Epsilon decay steps")
    parser.add_argument("--model-dir", type=str, default="models/checkpoints", help="Model save directory")
    
    args = parser.parse_args()
    
    # Create DQN config from arguments
    config = DQNConfig(
        gamma=args.gamma,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update,
        epsilon_decay_steps=args.epsilon_decay,
    )
    
    # Train
    train_dqn(
        total_steps=args.steps,
        warmup_steps=args.warmup,
        eval_frequency=args.eval_freq,
        model_dir=args.model_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
