# Tetris RL with DQN - Project Outline

## 1. Project Architecture
*   **Objective:** Train an agent to play Tetris using Deep Q-Learning (DQN) with improvements (Double DQN, Prioritized Experience Replay) to maximize score and line clears.
*   **Input (State Space):**
    *   Board State: \(20 \times 10\) grid (binary or integer representation).
    *   Current Piece: Encoded within composite action space.
*   **Output (Action Space):**
    *   Composite discrete actions representing: **Rotation (0-3) + Column Position (0-9)** = ~40 possible placements per piece.
    *   Each action directly specifies where and how to place the current piece.
*   **Algorithm:** Deep Q-Network (DQN) with enhancements:
    *   **Fixed Q-Targets:** Separate target network updated periodically for stability.
    *   **Double DQN:** Reduces Q-value overestimation by decoupling action selection and evaluation.
    *   **Prioritized Experience Replay (PER):** Samples transitions based on TD-error for efficient learning.

## 2. Development Phases

### Phase 1: Environment Setup & Wrapper ✅ COMPLETE
*   **Goal:** Establish a Gym-compatible Tetris environment.
*   **Tasks:**
    *   ✅ Wrap Tetris-Gymnasium environment with custom reward shaping.
    *   ✅ Define `observation_space` (grid + piece) and `action_space`.
    *   ✅ Implement `step()`, `reset()`, and `render()` functions.
    *   ✅ **Reward Engineering:** Exponential line clear bonus, hole penalty, bumpiness penalty, height penalty, survival bonus, game over penalty.

### Phase 2: Neural Network Architecture (CNN Q-Network)
*   **Goal:** Build the Q-network for value estimation.
*   **Components:**
    *   **Input Layer:** Accepts \((20, 10, 1)\) board tensor.
    *   **CNN Backbone:** 
        *   Conv2d layers to detect local features (holes, flat surfaces, wells, column heights).
        *   ReLU activations.
        *   Flatten layer.
    *   **Fully Connected Layers:** Process extracted features.
    *   **Output Head:**
        *   *Q-Value Head:* Outputs Q(s, a) for each composite action (rotation + column).
*   **Responsible:** Chris (model architecture)

### Phase 3: DQN Implementation & Training Loop
*   **Goal:** Implement the DQN algorithm with improvements.
*   **Key Components:**
    *   **Replay Buffer (PER):** Store transitions \((s, a, r, s', done)\) with TD-error priorities.
        *   Sample high-priority transitions more frequently.
        *   Update priorities after each training step.
    *   **Fixed Q-Targets:** 
        *   Maintain `q_network` (online) and `target_network`.
        *   Update target network every N steps: `target_network.load_state_dict(q_network.state_dict())`.
    *   **Double DQN:**
        *   Select action using online network: `a' = argmax_a Q_online(s', a)`.
        *   Evaluate using target network: `y = r + γ * Q_target(s', a')`.
    *   **Training Loop:**
        *   Epsilon-greedy exploration (ε decays over time).
        *   Collect transitions and store in replay buffer.
        *   Sample minibatch from PER buffer.
        *   Compute TD-error and update Q-network.
        *   Update priorities in buffer.
*   **Responsible:** Edan (DQN agent, replay buffer, training loop)

### Phase 4: Evaluation & Visualization
*   **Goal:** Measure performance and visualize results.
*   **Metrics:** Average score, lines cleared, game duration, holes per move, average Q-values.
*   **Baselines:** Compare against:
    *   Random agent (baseline established: ~-20 reward per 20 steps).
    *   Simple heuristic agent (optional: Dellacherie features).
*   **Deliverables:**
    *   Training curves (Reward vs. Episode, Epsilon decay, Loss).
    *   GIF/Video of the best agent playing.
    *   `project.ipynb` demonstration.
*   **Responsible:** All team members (joint evaluation and analysis)

## 3. File Structure

```text
src/
├── env/
│   ├── __init__.py
│   └── tetris_env.py         # Custom Gymnasium environment (✅ COMPLETE)
├── models/
│   ├── __init__.py
│   ├── dqn_network.py        # PyTorch CNN Q-Network architecture
│   └── dqn_agent.py          # DQN Algorithm (replay buffer, Double DQN, PER)
├── utils/
│   ├── __init__.py
│   ├── visualization.py      # Plotting and rendering helpers
│   └── preprocessing.py      # State conversion (board to tensor)
├── train.py                  # Main DQN training loop
└── evaluate.py               # Script to load model and test
```

## 4. Tools & Libraries
*   **Python ecosystem:** NumPy, SciPy
*   **Machine learning:** PyTorch
*   **RL Environment:** Gymnasium, Tetris-Gymnasium
*   **Visualization:** Matplotlib

## 5. Hyperparameters (Initial Values)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1e-4 | Adam optimizer learning rate |
| Discount Factor (γ) | 0.99 | Future reward discount |
| Replay Buffer Size | 100,000 | Maximum transitions stored |
| Batch Size | 64 | Minibatch size for training |
| Target Update Frequency | 1,000 steps | How often to sync target network |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon End | 0.01 | Minimum exploration rate |
| Epsilon Decay | 10,000 steps | Steps to decay epsilon |
| PER Alpha | 0.6 | Priority exponent |
| PER Beta | 0.4 → 1.0 | Importance sampling (annealed) |

## 6. Team Responsibilities

**Caleb:**
- ✅ Tetris-Gymnasium integration
- ✅ Custom environment wrapper
- ✅ Reward shaping implementation
- ✅ State preprocessing

**Chris:**
- CNN Q-Network architecture
- Forward pass implementation
- Model optimization and debugging

**Edan:**
- DQN agent implementation
- Prioritized Experience Replay buffer
- Double DQN logic
- Target network updates
- Main training loop

**All Members (Joint):**
- Hyperparameter tuning
- Agent evaluation
- Results analysis
- Report writing
