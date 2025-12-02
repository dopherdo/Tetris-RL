# Tetris-RL
Deep Q-Learning (DQN) for Tetris

A reinforcement learning project that trains an AI agent to play Tetris using the DQN algorithm with Double DQN, Prioritized Experience Replay, and custom reward engineering.

## Project Structure

```
Tetris-RL/
├── src/
│   ├── env/
│   │   ├── __init__.py
│   │   └── tetris_env.py          # Custom Gymnasium environment wrapper with reward engineering ✅
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dqn_network.py         # PyTorch CNN Q-Network architecture
│   │   └── dqn_agent.py           # DQN algorithm with PER and Double DQN
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── visualization.py       # Plotting and rendering helpers
│   │   └── preprocessing.py       # State conversion utilities
│   ├── train.py                   # Main DQN training loop
│   └── evaluate.py                # Model evaluation script
├── test_env.py                    # Environment testing and validation script
├── project.ipynb                  # Jupyter notebook demonstrating the project
├── requirements.txt               # Python dependencies
├── PROJECT_OUTLINE.md             # Detailed project plan and architecture
└── README.md                      # This file
```

## Setup Instructions

### 1. Create Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `gymnasium` - Standard RL environment interface
- `tetris-gymnasium` - Tetris game environment backend
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `torch` & `torchvision` - Deep learning framework
- `scipy` - Scientific computing
- `tqdm` - Progress bars
- `ipykernel` - Jupyter notebook support

### 3. Verify Environment Setup

Run the environment test:

```bash
source .venv/bin/activate
python -m src.env.tetris_env
```

This will:
- Create the custom Tetris environment
- Run 20 random actions to test functionality
- Display reward engineering metrics
- Show the final board state

## Algorithm Overview

### Deep Q-Network (DQN) with Enhancements

**Core Concept:** Learn a Q-function Q(s, a) that estimates the expected cumulative reward for taking action `a` in state `s`.

**Enhancements:**
1. **Fixed Q-Targets:** Separate target network updated periodically to stabilize learning
2. **Double DQN:** Decouples action selection and evaluation to reduce Q-value overestimation
3. **Prioritized Experience Replay (PER):** Prioritizes important transitions for faster learning

### Action Space

**Composite Actions:** Each action directly specifies piece placement:
- **Rotation:** 0-3 (four possible rotations)
- **Column:** 0-9 (ten possible landing columns)
- **Total Actions:** ~40 discrete composite actions per piece

This approach is more efficient than atomic actions (left, right, rotate, drop) because it reduces the action sequence length.

### Custom Reward Engineering

The environment implements sophisticated reward shaping:
- **Line Clear Bonus**: Exponential rewards for clearing multiple lines (2^n - 1)
- **Hole Penalty**: -0.1 per hole created (empty cell with filled cells above and below)
- **Bumpiness Penalty**: -0.01 × sum of height differences between adjacent columns
- **Height Penalty**: -0.01 × maximum column height
- **Survival Bonus**: +0.1 per step (encourages longer games)
- **Game Over Penalty**: -5.0 (strong negative signal)

### Metrics Tracking

The environment tracks comprehensive game statistics:
- Total lines cleared
- Holes count
- Column heights and bumpiness
- Episode steps and total score
- Custom reward breakdown

## Training Workflow

1. **State Observation:** Extract 20×10 board grid from Tetris-Gymnasium
2. **Preprocessing:** Convert board to tensor, normalize values
3. **CNN Processing:** Extract spatial features (holes, heights, patterns)
4. **Q-Value Estimation:** Output Q(s, a) for all composite actions
5. **Action Selection:** Epsilon-greedy (explore vs. exploit)
6. **Environment Step:** Execute action, receive reward and next state
7. **Store Transition:** Add (s, a, r, s', done) to prioritized replay buffer
8. **Training Update:** Sample batch, compute TD-error, update Q-network
9. **Target Network Sync:** Periodically copy weights to target network

## Current Status

✅ **Phase 1: Environment Setup** (COMPLETE)
- Custom Tetris environment with tetris-gymnasium backend
- Reward engineering implementation
- Comprehensive testing and validation
- Baseline: Random agent achieves ~-20 reward per 20 steps

🔄 **Phase 2: Neural Network Architecture** (IN PROGRESS)
- CNN Q-Network for feature extraction and Q-value estimation

🔄 **Phase 3: DQN Implementation** (TODO)
- Prioritized Experience Replay buffer
- Double DQN logic
- Training loop with epsilon-greedy exploration

🔄 **Phase 4: Evaluation** (TODO)
- Training metrics and visualization
- Agent performance evaluation vs. baselines

## Usage Example

### Environment Test

```python
from src.env.tetris_env import TetrisEnv

# Create environment
env = TetrisEnv(render_mode=None)

# Reset and play
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Reward: {reward:.2f}, Holes: {info['holes']}, Height: {info['max_height']}")
    
    if terminated or truncated:
        break

env.close()
```

### Training (Coming Soon)

```bash
python -m src.train --episodes 1000 --batch-size 64 --lr 1e-4
```

### Evaluation (Coming Soon)

```bash
python -m src.evaluate models/checkpoints/dqn_tetris_ep500.pt --episodes 5 --render
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1e-4 | Adam optimizer learning rate |
| Discount Factor (γ) | 0.99 | Future reward discount |
| Replay Buffer Size | 100,000 | Maximum transitions stored |
| Batch Size | 64 | Minibatch size for training |
| Target Update Freq | 1,000 steps | Target network sync interval |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon End | 0.01 | Minimum exploration rate |
| Epsilon Decay | 10,000 steps | Exploration decay period |

## Team Members

- **Caleb Chu** - Environment integration, reward engineering, preprocessing
- **Christopher Yeh** - CNN Q-Network architecture, model optimization
- **Edan Sasson** - DQN agent, replay buffer, training loop

## References

- [Tetris Gymnasium Documentation](https://max-we.github.io/Tetris-Gymnasium/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [DQN Paper (Mnih et al., 2015)](https://arxiv.org/abs/1312.5602)
- [Double DQN Paper](https://arxiv.org/abs/1509.06461)
- [Prioritized Experience Replay Paper](https://arxiv.org/abs/1511.05952)
