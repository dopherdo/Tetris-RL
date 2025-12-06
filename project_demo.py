"""Standalone demo script for trained model."""
import torch
import numpy as np
from pathlib import Path

from src.env import TetrisEnv, CompositeActionWrapper
from src.models import DQNAgent, DQNConfig
from src.utils import preprocess_observation

print("="*70)
print("DEEP Q-LEARNING FOR TETRIS - LIVE DEMO")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\n" + "="*70)
print("LOADING TRAINED MODEL")
print("="*70)

base_env = TetrisEnv(render_mode=None)
env = CompositeActionWrapper(base_env)

obs, _ = env.reset()
board = preprocess_observation(obs)
board_shape = board.shape
n_actions = env.action_space.n

print(f"Board shape: {board_shape}")
print(f"Number of composite actions: {n_actions}")

config = DQNConfig()
agent = DQNAgent(
    board_shape=board_shape,
    n_actions=n_actions,
    device=device,
    config=config
)

checkpoint_path = Path('checkpoints/dqn_continued_final.pt')
if checkpoint_path.exists():
    agent.load(str(checkpoint_path))
    print(f"Loaded checkpoint: {checkpoint_path}")
else:
    print("No checkpoint found! Using untrained model.")

agent.q_network.eval()

print("\n" + "="*70)
print("RUNNING EVALUATION EPISODES")
print("="*70)

def run_episode(env, agent, use_agent=True):
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 1000:
        if use_agent:
            board = preprocess_observation(obs)
            action = agent.select_action(board, eval_mode=True)
        else:
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    
    return {
        'reward': total_reward,
        'steps': steps,
        'pieces': steps,
        'lines': info.get('lines_cleared', 0),
        'holes': info.get('holes', 0),
        'height': info.get('max_height', 0)
    }

print("Running 5 episodes with TRAINED agent...")
trained_results = [run_episode(env, agent, use_agent=True) for _ in range(5)]

print("Running 5 episodes with RANDOM agent...")
random_results = [run_episode(env, agent, use_agent=False) for _ in range(5)]

print("\n" + "="*70)
print("PERFORMANCE COMPARISON (5 episodes average)")
print("="*70)

def calc_avg(results, key):
    return np.mean([r[key] for r in results])

metrics = ['reward', 'steps', 'pieces', 'lines', 'holes', 'height']
labels = ['Avg Reward', 'Avg Steps', 'Avg Pieces', 'Avg Lines', 'Final Holes', 'Final Height']

print(f"{'Metric':<20} {'Trained':<15} {'Random':<15} {'Improvement'}")
print("-" * 70)

for metric, label in zip(metrics, labels):
    trained_val = calc_avg(trained_results, metric)
    random_val = calc_avg(random_results, metric)
    
    if random_val != 0:
        improvement = ((trained_val - random_val) / abs(random_val)) * 100
        print(f"{label:<20} {trained_val:>10.1f}    {random_val:>10.1f}    {improvement:>+8.1f}%")
    else:
        print(f"{label:<20} {trained_val:>10.1f}    {random_val:>10.1f}    {'N/A':>10}")

print("="*70)
improvement = ((calc_avg(trained_results, 'pieces') - calc_avg(random_results, 'pieces')) / 
               calc_avg(random_results, 'pieces')) * 100
print(f"\nDemo complete! Model performs {improvement:.1f}% better than random.")
