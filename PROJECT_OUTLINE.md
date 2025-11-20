# Tetris RL with PPO - Project Outline

## 1. Project Architecture
*   **Objective:** Train an agent to play Tetris using Proximal Policy Optimization (PPO) to maximize score and line clears.
*   **Input (State Space):**
    *   Board State: $20 \times 10$ grid (binary or integer representation).
    *   Current Piece: One-hot encoding or integer ID of the 7 tetrominoes.
*   **Output (Action Space):**
    *   Discrete actions representing specific moves (e.g., Rotate Left/Right + Drop Column Index).
    *   *Alternative:* Discrete set of atomic actions (Left, Right, Rotate, Soft Drop, Hard Drop).
*   **Algorithm:** Proximal Policy Optimization (PPO) (Actor-Critic).
    *   **Actor:** Outputs probability distribution over actions.
    *   **Critic:** Estimates value function $V(s)$ for the current state.

## 2. Development Phases

### Phase 1: Environment Setup & Wrapper
*   **Goal:** Establish a Gym-compatible Tetris environment.
*   **Tasks:**
    *   Implement or wrap a Tetris engine (custom or based on libraries like `gym-tetris`).
    *   Define `observation_space` (grid + piece) and `action_space`.
    *   Implement `step()`, `reset()`, and `render()` functions.
    *   **Crucial:** Implement "Reward Engineering" to guide learning (e.g., $+1$ per line, $-0.1$ per hole created, $+0.01$ for surviving).

### Phase 2: Neural Network Architecture (CNN)
*   **Goal:** Build the feature extractor and PPO heads.
*   **Components:**
    *   **Input Layer:** Accepts $(20, 10, 1)$ board tensor.
    *   **CNN Backbone:** Convolutional layers to detect local features (holes, flat surfaces, wells).
    *   **Fully Connected Layers:** Process extracted features.
    *   **Heads:**
        *   *Policy Head (Actor):* Output logits for actions.
        *   *Value Head (Critic):* Scalar output for state value.

### Phase 3: PPO Implementation & Training Loop
*   **Goal:** Implement the training logic.
*   **Key Logic:**
    *   **Rollout Buffer:** Store transitions $(s, a, r, s', \log \pi)$.
    *   **GAE (Generalized Advantage Estimation):** Calculate advantages for stable updates.
    *   **PPO Clip Loss:** Implement the clipped objective function to prevent drastic policy shifts.
    *   **Updates:** Batched stochastic gradient descent on collected trajectories.

### Phase 4: Evaluation & Visualization
*   **Goal:** Measure performance and visualize results.
*   **Metrics:** Average score, lines cleared, game duration, holes per move.
*   **Baselines:** Compare against a random agent and a simple heuristic agent.
*   **Deliverables:**
    *   Training curves (Reward vs. Episode).
    *   GIF/Video of the best agent playing.
    *   `project.ipynb` demonstration.

## 3. File Structure

```text
src/
├── env/
│   ├── __init__.py
│   └── tetris_env.py       # Custom Gymnasium environment
├── models/
│   ├── __init__.py
│   ├── cnn_policy.py       # PyTorch CNN architecture
│   └── ppo_agent.py        # PPO Algorithm logic (update step, buffer)
├── utils/
│   ├── visualization.py    # Plotting and rendering helpers
│   └── preprocessing.py    # State conversion (board to tensor)
├── train.py                # Main training loop
└── evaluate.py             # Script to load model and test
```

## 4. Tools & Libraries
*   **Python ecosystem:** NumPy, SciPy
*   **Machine learning:** PyTorch
*   **RL Environment:** Gymnasium
*   **Visualization:** Matplotlib

