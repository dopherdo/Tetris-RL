"""Evaluation script: load a trained DQN agent and watch it play Tetris."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .env import make_tetris_env
from .models import DQNAgent, DQNConfig
from .utils import preprocess_observation
from .train import infer_board_shape


def evaluate(
    checkpoint_path: str,
    episodes: int = 5,
    render_mode: str | None = "human",
    verbose: bool = True,
) -> dict:
    """Evaluate a trained DQN agent."""
    env = make_tetris_env(render_mode=render_mode)
    n_actions = env.action_space.n
    board_shape = infer_board_shape(env)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(
        board_shape=board_shape,
        n_actions=n_actions,
        device=device,
        config=DQNConfig(),
    )
    
    agent.load(checkpoint_path)
    agent.q_network.eval()
    
    if verbose:
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Board shape: {board_shape}")
        print(f"Action space: {n_actions}")
        print(f"Device: {device}")
        print(f"\nRunning {episodes} evaluation episodes...\n")
    
    episode_rewards = []
    episode_lengths = []
    episode_lines = []
    episode_scores = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        obs_proc = preprocess_observation(obs)
        done = False
        ep_reward = 0.0
        ep_length = 0
        
        while not done:
            action = agent.select_action(obs_proc, eval_mode=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            obs_proc = preprocess_observation(obs)
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_lines.append(info.get('lines_cleared', 0))
        episode_scores.append(info.get('score', 0))
        
        if verbose:
            print(
                f"Episode {ep + 1}/{episodes}: "
                f"Reward = {ep_reward:.2f}, "
                f"Length = {ep_length}, "
                f"Lines = {info.get('lines_cleared', 0)}, "
                f"Score = {info.get('score', 0)}"
            )
    
    env.close()
    
    stats = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'mean_lines': float(np.mean(episode_lines)),
        'mean_score': float(np.mean(episode_scores)),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_lines': episode_lines,
        'episode_scores': episode_scores,
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Mean Length: {stats['mean_length']:.1f} steps")
        print(f"Mean Lines Cleared: {stats['mean_lines']:.1f}")
        print(f"Mean Score: {stats['mean_score']:.1f}")
        print("=" * 60)
    
    return stats


def compare_with_random(
    checkpoint_path: str,
    episodes: int = 20,
) -> None:
    """Compare trained agent with random baseline."""
    print("\n" + "=" * 60)
    print("COMPARISON: Trained Agent vs. Random Baseline")
    print("=" * 60)
    
    print("\n1. TRAINED AGENT:")
    trained_stats = evaluate(checkpoint_path, episodes=episodes, render_mode=None, verbose=False)
    print(f"   Mean Reward: {trained_stats['mean_reward']:.2f} ± {trained_stats['std_reward']:.2f}")
    print(f"   Mean Lines: {trained_stats['mean_lines']:.1f}")
    print(f"   Mean Length: {trained_stats['mean_length']:.1f} steps")
    
    print("\n2. RANDOM BASELINE:")
    env = make_tetris_env(render_mode=None)
    random_rewards = []
    random_lines = []
    random_lengths = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated
        
        random_rewards.append(ep_reward)
        random_lines.append(info.get('lines_cleared', 0))
        random_lengths.append(ep_length)
    
    env.close()
    
    print(f"   Mean Reward: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    print(f"   Mean Lines: {np.mean(random_lines):.1f}")
    print(f"   Mean Length: {np.mean(random_lengths):.1f} steps")
    
    reward_improvement = (
        (trained_stats['mean_reward'] - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100
    )
    lines_improvement = (
        (trained_stats['mean_lines'] - np.mean(random_lines)) / max(np.mean(random_lines), 1) * 100
    )
    
    print("\n" + "=" * 60)
    print("IMPROVEMENT OVER RANDOM:")
    print(f"   Reward: {reward_improvement:+.1f}%")
    print(f"   Lines Cleared: {lines_improvement:+.1f}%")
    print("=" * 60 + "\n")


def main():
    """Main entry point for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent on Tetris")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render gameplay")
    parser.add_argument("--compare", action="store_true", help="Compare with random baseline")
    
    args = parser.parse_args()
    
    render_mode = "human" if args.render else None
    
    if args.compare:
        compare_with_random(args.checkpoint, episodes=max(args.episodes, 20))
    else:
        evaluate(args.checkpoint, episodes=args.episodes, render_mode=render_mode)


if __name__ == "__main__":
    main()
