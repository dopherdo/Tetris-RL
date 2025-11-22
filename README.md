# Tetris-RL
Proximal Policy Optimization (PPO) for Tetris

A reinforcement learning project that trains an AI agent to play Tetris using the PPO algorithm with custom reward engineering.

## Project Structure

```
Tetris-RL/
├── src/
│   ├── env/
│   │   ├── __init__.py
│   │   └── tetris_env.py          # Custom Gymnasium environment wrapper with reward engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_policy.py          # PyTorch CNN architecture (TODO: Phase 2)
│   │   └── ppo_agent.py           # PPO algorithm implementation (TODO: Phase 3)
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py       # Plotting and rendering helpers (TODO: Phase 4)
│       └── preprocessing.py       # State conversion utilities (TODO: Phase 2)
├── test_env.py                    # Environment testing and validation script
├── project.ipynb                  # Jupyter notebook demonstrating the project
├── requirements.txt               # Python dependencies
├── PROJECT_OUTLINE.md             # Detailed project plan and architecture
└── README.md                      # This file
```

## Setup Instructions

### 1. Install Dependencies

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

### 2. Verify Environment Setup

Run the test script to verify Phase 1 is working correctly:

```bash
python3 test_env.py
```

This will:
- Create the custom Tetris environment
- Run random actions to test functionality
- Display reward engineering metrics
- Verify all components are working

### 3. Test CNN Integration (Phase 2)

Run the integration test to verify the CNN architecture:

```bash
python3 test_cnn_integration.py
```

This comprehensive test validates:
- State preprocessing (board → tensor conversion)
- CNN policy forward pass (tensor → actions + values)
- Batch processing capabilities
- Gradient flow through the network
- Action distribution analysis
- Value prediction consistency

### 4. Run the Jupyter Notebook

```bash
jupyter notebook project.ipynb
```

The notebook demonstrates:
- Environment initialization
- Random agent gameplay
- Reward metrics visualization
- Board state visualization

## Workflow

1. **Preprocessing:** Board is preprocessed into a usable tensor
2. **CNN Analysis:** CNN processes the tensor into a feature vector
3. **Decision:** PPO takes the feature vector and determines the best action
4. **Action:** The selected action is sent to the environment
5. **Update:** Environment processes the move and updates the board state
6. **Loop:** Repeats from **Step 1** with the new board state until Game Over

## Features

### Macro-Actions for Better RL Learning

The environment uses **macro-actions** (direct piece placements) instead of atomic actions:
- **Macro-action**: One action = place piece at specific (column, rotation)
- **Action space**: Discrete(40) = 10 columns × 4 rotations
- **Benefits**:
  - ✅ **Immediate credit assignment**: One action → one outcome → one reward
  - ✅ **Faster learning**: Shorter episodes, clearer cause-effect
  - ✅ **Better for PPO**: Denser rewards, more stable training
  - ✅ **Human-intuitive**: Matches strategic thinking

Compare to atomic actions (left, right, rotate, drop):
- Atomic requires multiple steps per placement
- Harder credit assignment (which action caused the result?)
- Longer episodes, sparser rewards

### Custom Reward Engineering

The environment implements sophisticated reward shaping:
- **Line Clear Bonus**: Exponential rewards for clearing multiple lines (2^n - 1)
- **Hole Penalty**: -0.5 per hole created (empty cell with filled cell above)
- **Bumpiness Penalty**: -0.1 × sum of height differences between adjacent columns
- **Height Penalty**: -0.05 × maximum column height
- **Survival Bonus**: +0.01 per step (encourages longer games)
- **Game Over Penalty**: -5.0 (strong negative signal)

### Metrics Tracking

The environment tracks comprehensive game statistics:
- Total lines cleared
- Holes count
- Column heights and bumpiness
- Episode steps and total score
- Custom reward breakdown

## Current Status

✅ **Phase 1: Environment Setup** (COMPLETED)
- Custom Tetris environment with tetris-gymnasium backend
- Reward engineering implementation
- Comprehensive testing and validation

✅ **Phase 2: Neural Network Architecture** (COMPLETED)
- CNN feature extractor with 3 convolutional layers
- Policy and value network heads (Actor-Critic)
- State preprocessing utilities
- Full integration testing with environment

🔄 **Phase 3: PPO Implementation** (TODO)
- Rollout buffer and GAE
- PPO clip loss and training loop

🔄 **Phase 4: Evaluation** (TODO)
- Training metrics and visualization
- Agent performance evaluation

## Usage Example

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

## References

- [Tetris Gymnasium Documentation](https://max-we.github.io/Tetris-Gymnasium/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)