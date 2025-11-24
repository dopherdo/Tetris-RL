"""
Continue DQN training from checkpoint with optimizations from blurb

Key optimizations implemented:
1. Double DQN - Reduces Q-value overestimation bias
2. Experience Replay - Efficient use of experiences
3. Fixed Q-Target - Stable training with target network
4. Gamma = 0.999 - High discount for long-term rewards and delayed credit assignment
5. Pillar penalty - Board management reward component
6. Action masking - 40-50% action space reduction

Based on findings:
- Sparse rewards for better board management
- High discount factor (0.999) for delayed rewards
- Penalties for holes, bumpiness, height, and pillars
- Encouragement for moves that improve board stability
"""
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from src.env.tetris_env import TetrisEnv
from src.utils.preprocessing import TetrisPreprocessor
from src.models.dqn_agent import DQNAgent


def continue_training(
    checkpoint_path=None,  # Auto-detect latest checkpoint, or start fresh if none
    additional_steps=50_000,
    eval_freq=5_000,  # Recommended: Evaluate every 5K steps (good balance)
    eval_episodes=50,  # 50 episodes per evaluation (stable statistics)
    target_total_steps=None  # If set, will calculate additional_steps to reach this
):
    """
    Continue training from checkpoint
    """
    print("=" * 70)
    print(" " * 15 + "CONTINUE DQN TRAINING")
    print("=" * 70)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create environment with optimized reward configuration
    # Based on blurb findings: Sparse rewards for board management, pillar penalty added
    env = TetrisEnv(
        render_mode=None,
        reward_line_clear=120.0,        # Strong positive for clearing lines
        reward_hole_penalty=-2.0,       # Strongest penalty (prevents trapped empty spaces)
        reward_height_penalty=-0.1,     # Medium penalty (prevents stacking too high)
        reward_bumpiness_penalty=-0.1,  # Weak penalty (encourages smooth surfaces)
        reward_pillar_penalty=-0.15,    # NEW: Penalty for tall pillars (prevents difficult structures)
        reward_survival=0.5             # Small bonus for surviving
    )
    
    preprocessor = TetrisPreprocessor(use_active_piece=True, device=device)
    
    # Create agent with PER and optimized hyperparameters
    # Based on blurb findings: gamma=0.999 for long-term rewards and delayed credit assignment
    agent = DQNAgent(
        state_shape=(2, 20, 10),
        num_actions=40,
        device=device,
        learning_rate=1.5e-4,  # Optimized learning rate
        gamma=0.999,  # HIGH DISCOUNT: Handles delayed rewards (good moves may pay off many steps later)
        epsilon_start=0.15,  # Start with moderate exploration
        epsilon_end=0.05,  # Maintain minimum exploration (prevent collapse)
        epsilon_decay=0.99995,  # Slower decay
        batch_size=256,  # Increased batch size
        buffer_capacity=100_000,
        target_update_freq=750,  # Sweet spot: 750 steps
        learning_starts=1000,  # Start learning earlier
        use_per=True,  # Use Prioritized Experience Replay
        per_alpha=0.6,  # Priority exponent
        per_beta=0.4,  # Importance sampling exponent
        per_beta_increment=1e-6  # Beta annealing rate
    )
    
    # Handle checkpoint loading
    # If checkpoint_path is None, start fresh (don't auto-detect)
    # If checkpoint_path is provided but doesn't exist, try to find latest
    if checkpoint_path is None:
        print("Starting training from scratch (no checkpoint specified)")
        checkpoint_path = None
    elif not os.path.exists(checkpoint_path):
        # Try to find the latest checkpoint as fallback
        checkpoint_dir = 'checkpoints'
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoint_files:
                # Sort by modification time
                checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                print(f"Checkpoint not found, using latest: {checkpoint_path}")
            else:
                print(f"WARNING: No checkpoint found. Starting from scratch.")
                checkpoint_path = None
        else:
            print(f"WARNING: Checkpoint directory not found. Starting from scratch.")
            checkpoint_path = None
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        agent.load(checkpoint_path)
        start_steps = agent.total_steps
        print(f"  Loaded at step: {start_steps:,}")
        print(f"  Current epsilon: {agent.epsilon:.3f}")
        print(f"  Buffer size: {len(agent.replay_buffer):,}")
        print()
        
        # If continuing from checkpoint, apply optimized hyperparameters
        # These are already set in agent creation, but we'll ensure they're correct
        if agent.total_steps >= 10000:
            print("=" * 70)
            print("APPLYING OPTIMIZED HYPERPARAMETERS + TECHNIQUES FROM BLURB")
            print("=" * 70)
            print("NEW OPTIMIZATIONS:")
            print("  ✓ Action masking: 40-50% action space reduction!")
            print("    - O piece: 40 → 10 actions (75% reduction)")
            print("    - I piece: 40 → 20 actions (50% reduction)")
            print("    - S/Z pieces: 40 → 20 actions (50% reduction)")
            print("    - Column masking for wide pieces")
            print()
            print("  ✓ Double DQN: Reduces Q-value overestimation bias")
            print("    - Main network selects action, target network evaluates")
            print("    - More accurate Q-value estimates, better performance")
            print()
            print("  ✓ Gamma = 0.999: High discount for long-term rewards")
            print("    - Handles delayed credit assignment (good moves pay off later)")
            print("    - Critical for Tetris where setup moves matter")
            print()
            print("  ✓ Pillar penalty: NEW reward component for board management")
            print("    - Penalizes tall pillars (columns 3+ blocks taller than neighbors)")
            print("    - Prevents difficult-to-clear structures")
            print()
            print("HYPERPARAMETER UPDATES (OPTIMIZED FROM BLURB + PER):")
            print("  ✓ Prioritized Experience Replay (PER): ENABLED")
            print("    - Samples high TD-error experiences (biggest mistakes first)")
            print("    - 20-30% better sample efficiency")
            print("  Learning rate: 1.5e-4 (balanced for stability + learning speed)")
            print("  Epsilon: 0.15 → 0.05 (maintain exploration, prevent collapse)")
            print("  Batch size: 256 (better gradient estimates)")
            print("  Target update: 750 steps (sweet spot for stability)")
            print("  Learning starts: 1000 (start learning earlier)")
            print("  Epsilon decay: 0.99995 (slower decay, maintain exploration longer)")
            print("  Gamma: 0.999 (long-term reward focus from blurb)")
            print()
            print("Reason: Applying proven optimizations from successful Tetris RL implementation.")
            print("        PER is the #1 missing piece - focuses learning on biggest mistakes.")
            print()
            print("REWARD UPDATES (OPTIMIZED CONFIG):")
            print("  Line clear reward: 120.0 (strong positive for clearing lines)")
            print("  Survival reward: 0.5 (small bonus for surviving)")
            print("  Hole penalty: -2.0 (strongest penalty, prevents trapped spaces)")
            print("  Height penalty: -0.1 (prevents stacking too high)")
            print("  Bumpiness penalty: -0.1 (encourages smooth surfaces)")
            print("  Pillar penalty: -0.15 (NEW: prevents difficult structures)")
            print()
            print("Expected benefits:")
            print("  • Faster learning (fewer invalid actions to explore)")
            print("  • Better sample efficiency (focus on valid actions)")
            print("  • More accurate Q-values (Double DQN reduces overestimation)")
            print("  • Better long-term planning (gamma=0.999 for delayed rewards)")
            print("  • Improved board management (pillar penalty)")
            print("  • Stable training (balanced hyperparameters)")
            print()
            
            # Update hyperparameters - MORE CONSERVATIVE to prevent degradation
            # Performance degraded from 23.0 → 21.7 pieces, need more stable settings
            import torch.optim as optim
            agent.set_learning_rate(1.5e-4)  # Use new method for LR scheduling
            agent.set_epsilon(0.15)  # Use new method for epsilon scheduling
            agent.epsilon_decay = 0.99995  # Slower decay (maintain exploration longer)
            agent.batch_size = 256  # Keep moderate increase
            agent.target_update_freq = 250  # More frequent updates (stabilize Q-values faster)
            agent.learning_starts = 1000  # Start learning earlier but not immediately
            agent.gamma = 0.999  # High discount for long-term rewards (from blurb)
    else:
        # No checkpoint found - start from scratch
        start_steps = agent.total_steps
        print("Starting training from scratch (no checkpoint found)")
        print()
        print("Using optimized hyperparameters from blurb findings:")
        print("  • Gamma = 0.999 (long-term reward focus)")
        print("  • Double DQN (reduces overestimation)")
        print("  • Pillar penalty (board management)")
        print()
        # Apply optimized hyperparameters from blurb findings
        # Key improvements: Lower LR for stability, slower epsilon decay, LR scheduling
        import torch.optim as optim
        agent.set_learning_rate(1e-4)  # Reduced from 3e-4 for stability (loss was increasing)
        agent.set_epsilon(0.2)  # Higher initial epsilon for more exploration
        agent.epsilon_decay = 0.99995  # Slower decay (maintain exploration longer)
        agent.batch_size = 256
        agent.target_update_freq = 250  # More frequent target updates for stability
        agent.learning_starts = 1000
        agent.gamma = 0.999  # High discount for long-term rewards
    
    # Calculate additional_steps if target_total_steps is set
    if target_total_steps is not None:
        additional_steps = max(0, target_total_steps - agent.total_steps)
        print(f"  Target: {target_total_steps:,} steps")
        print(f"  Current: {agent.total_steps:,} steps")
        print(f"  Will train: {additional_steps:,} more steps")
        print()
    
    # Evaluate current performance
    print("CURRENT PERFORMANCE")
    print("-" * 70)
    current_stats = evaluate(agent, env, preprocessor, eval_episodes)
    print()
    
    print(f"Training plan:")
    print(f"  Additional steps: {additional_steps:,}")
    print(f"  Eval every: {eval_freq:,} steps")
    print(f"  Final target: {agent.total_steps + additional_steps:,} steps")
    print()
    print("=" * 70)
    print()
    
    # Training loop
    obs, info = env.reset()
    state = preprocessor(obs)
    
    losses = []
    history = {
        'steps': [agent.total_steps],
        'mean_pieces': [current_stats['mean_pieces']],
        'mean_reward': [current_stats['mean_reward']],
        'mean_lines': [current_stats['mean_lines']],
        'max_pieces': [current_stats['max_pieces']],
        'epsilon': [agent.epsilon],
        'loss': []
    }
    
    pbar = tqdm(range(additional_steps), desc="Continuing training")
    
    # Open log file for both progress (1K) and evaluations (5K)
    log_file = open('training.log', 'w')
    log_file.write(f"Training Log (1K updates + 5K evaluations)\n")
    log_file.write(f"Starting from: {agent.total_steps:,} steps\n")
    log_file.write(f"Target: {agent.total_steps + additional_steps:,} steps\n")
    log_file.write(f"Starting performance: {current_stats['mean_pieces']:.1f} pieces\n")
    log_file.write("=" * 70 + "\n\n")
    log_file.flush()
    
    for step in pbar:
        # Interact
        legal_mask = info.get('legal_actions_mask', None)
        action = agent.select_action(state, legal_actions_mask=legal_mask)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = preprocessor(next_obs)
        done = terminated or truncated
        
        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.train_step()
        
        if loss is not None:
            losses.append(loss)
        
        if done:
            obs, _ = env.reset()
            state = preprocessor(obs)
        else:
            state = next_state
        
        # Update progress
        if losses:
            pbar.set_postfix({
                'ε': f"{agent.epsilon:.3f}",
                'loss': f"{np.mean(losses[-100:]):.2f}",
                'LR': f"{agent.learning_rate:.0e}",
                'buf': f"{len(agent.replay_buffer)}"
            })
        
        # Learning rate scheduling: Halve LR every 10K steps (from blurb - alternating phases)
        if (step + 1) % 10000 == 0 and agent.total_steps > start_steps:
            new_lr = agent.learning_rate * 0.5
            agent.set_learning_rate(new_lr)
            print(f"\nLearning rate decayed to: {new_lr:.1e}")
            log_file.write(f"Learning rate decayed to: {new_lr:.1e}\n")
            log_file.flush()
        
        # Log progress every 1K steps (without evaluation for speed)
        if (step + 1) % 1000 == 0:
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            
            # Log to file (include learning rate for monitoring)
            log_msg = f"Step {agent.total_steps:,}: ε={agent.epsilon:.3f}, loss={avg_loss:.3f}, LR={agent.learning_rate:.0e}, buffer={len(agent.replay_buffer):,}\n"
            log_file.write(log_msg)
            log_file.flush()
            
            # Print to console (line by line progress)
            print(f"Step {agent.total_steps:,}: ε={agent.epsilon:.3f}, loss={avg_loss:.3f}, LR={agent.learning_rate:.0e}, buffer={len(agent.replay_buffer):,}")
        
        # Evaluation
        if (step + 1) % eval_freq == 0:
            print(f"\n\nEVAL @ {agent.total_steps:,} steps")
            print("-" * 70)
            eval_stats = evaluate(agent, env, preprocessor, eval_episodes)
            
            history['steps'].append(agent.total_steps)
            history['mean_pieces'].append(eval_stats['mean_pieces'])
            history['mean_reward'].append(eval_stats['mean_reward'])
            history['mean_lines'].append(eval_stats['mean_lines'])
            history['max_pieces'].append(eval_stats['max_pieces'])
            history['epsilon'].append(agent.epsilon)
            if losses:
                history['loss'].append(np.mean(losses[-1000:]))
            
            # Log evaluation to same log file
            log_file.write(f"\nEVAL @ {agent.total_steps:,} steps:\n")
            log_file.write(f"  Pieces: {eval_stats['mean_pieces']:.1f} ± {eval_stats['std_pieces']:.1f}\n")
            log_file.write(f"  Lines: {eval_stats['mean_lines']:.1f}\n")
            log_file.write(f"  Best: {eval_stats['max_pieces']} pieces, {eval_stats['max_lines']} lines\n")
            log_file.write(f"  Improvement: {eval_stats['mean_pieces'] / current_stats['mean_pieces']:.2f}x\n")
            log_file.write("-" * 70 + "\n")
            log_file.flush()
            
            # Save checkpoint
            agent.save(f'checkpoints/dqn_continued_{agent.total_steps}.pt')
            
            improvement = eval_stats['mean_pieces'] / current_stats['mean_pieces']
            print(f"  → Improvement: {improvement:.2f}x from start of session")
            print()
            
            obs, _ = env.reset()
            state = preprocessor(obs)
    
    # Close log file
    log_file.close()
    
    # Final save - also save with step number for easy restart
    agent.save('checkpoints/dqn_continued_final.pt')
    agent.save(f'checkpoints/dqn_continued_{agent.total_steps}.pt')
    
    # If we hit exactly 30K, ensure checkpoint exists for restart
    if agent.total_steps == 30000:
        agent.save('checkpoints/dqn_continued_30000.pt')
    
    # Plot
    plot_continued_training(history, current_stats['mean_pieces'])
    
    # Summary
    final_improvement = history['mean_pieces'][-1] / history['mean_pieces'][0]
    
    print("\n" + "=" * 70)
    print("SESSION COMPLETE!")
    print("=" * 70)
    print(f"  Start of session: {history['mean_pieces'][0]:.1f} pieces")
    print(f"  End of session:   {history['mean_pieces'][-1]:.1f} pieces")
    print(f"  Improvement: {final_improvement:.2f}x")
    print(f"  Total steps trained: {agent.total_steps:,}")
    print()
    
    env.close()


def evaluate(agent, env, preprocessor, num_episodes, verbose=True):
    """Evaluate agent"""
    rewards, pieces, lines = [], [], []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0
        
        while True:
            state = preprocessor(obs)
            legal_mask = info.get('legal_actions_mask', None)
            action = agent.select_action(state, legal_actions_mask=legal_mask, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            if terminated or truncated:
                rewards.append(ep_reward)
                pieces.append(info['pieces_placed'])
                lines.append(env.total_lines_cleared)
                break
    
    # Find the best run (episode with most pieces)
    best_idx = np.argmax(pieces)
    best_pieces = pieces[best_idx]
    best_lines = lines[best_idx]
    
    stats = {
        'mean_reward': np.mean(rewards),
        'mean_pieces': np.mean(pieces),
        'std_pieces': np.std(pieces),
        'mean_lines': np.mean(lines),
        'max_pieces': best_pieces,
        'max_lines': best_lines  # Lines for the best run (most pieces)
    }
    
    if verbose:
        print(f"  Pieces: {stats['mean_pieces']:.1f} ± {stats['std_pieces']:.1f}")
        print(f"  Lines:  {stats['mean_lines']:.1f}")
        print(f"  Best:   {stats['max_pieces']} pieces, {stats['max_lines']} lines")
    
    return stats


def plot_continued_training(history, starting_pieces):
    """Plot continued training"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    steps = np.array(history['steps']) / 1000
    
    # Pieces
    ax1.plot(steps, history['mean_pieces'], marker='o', linewidth=2, markersize=6)
    ax1.axhline(y=starting_pieces, color='r', linestyle='--', linewidth=2, 
                label=f'Session start ({starting_pieces:.1f})')
    ax1.plot(steps, history['max_pieces'], marker='x', linestyle='--', 
             alpha=0.5, color='green', label='Best')
    ax1.set_xlabel('Total Training Steps (1000s)')
    ax1.set_ylabel('Pieces Placed')
    ax1.set_title('Continued Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    if history['loss']:
        ax2.plot(steps[1:], history['loss'], linewidth=2, color='orange')
        ax2.set_xlabel('Total Training Steps (1000s)')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/dqn_continued.png', dpi=150)
    print(f"\n✓ Plot saved: results/dqn_continued.png")


if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING WITH ACTION MASKING + OPTIMIZED HYPERPARAMETERS")
    print("=" * 70)
    print("Optimizations:")
    print("  ✓ Action masking: 40-50% action space reduction")
    print("  ✓ Tuned rewards for longer games (higher survival, reduced penalties)")
    print("  ✓ Optimized hyperparameters for faster learning")
    print()
    
    # RECOMMENDED: START FRESH with Double DQN + all optimizations
    # 
    # Why start fresh?
    # - 35K checkpoint was trained with standard DQN (overestimation bias)
    # - Old gamma=0.99 vs new gamma=0.999 (Q-value mismatch)
    # - New pillar penalty changes reward structure
    # - Clean learning with all optimizations from the start
    #
    # The new optimizations (Double DQN, gamma=0.999, pillar penalty, action masking)
    # should help it learn faster and better than the old checkpoint.
    #
    # Training configuration:
    # - Evaluation: Every 5K steps (good balance of progress tracking vs speed)
    # - Eval episodes: 50 (stable statistics without being too slow)
    # - Training length: 50K-100K steps before tuning
    # 
    # After this run, evaluate performance and tune hyperparameters if needed
    
    continue_training(
        checkpoint_path='checkpoints/dqn_continued_70000.pt',  # Continue from 70K checkpoint (peak performance: 24.4 pieces)
        additional_steps=50_000,       # Train for 50K more steps (70K → 120K total)
        eval_freq=5_000,               # Evaluate every 5K steps (recommended)
        eval_episodes=50               # 50 episodes per evaluation (stable stats)
    )

