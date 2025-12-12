# Tetris-RL

Deep Q-Learning (DQN) implementation for playing Tetris using reinforcement learning.

## Description

This project implements a DQN agent to play Tetris. The agent uses a CNN to process the game board and learns to play through trial and error.

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python -m src.train
```

### Evaluation
```bash
python -m src.evaluate checkpoints/dqn_continued_final.pt
```

### Demo
```bash
python project_demo.py
```

## Project Structure

```
src/
├── env/           # Tetris environment
├── models/        # DQN network and agent
├── utils/         # Helper functions
├── train.py       # Training script
└── evaluate.py    # Evaluation script
```

## File Descriptions

- `project_demo.py`: CLI demo to run the trained Tetris agent.
- `requirements.txt`: Python dependencies for training and evaluation.
- `checkpoints/`: Saved DQN models (various training steps and final).
- `figures/`: Plots for learning curves, reward breakdown, and comparisons.
- `src/`: Package with environment, models, utilities, and training/eval entrypoints.
  - `src/train.py`: Trains the DQN agent.
  - `src/evaluate.py`: Loads a checkpoint and evaluates the agent.
  - `src/main.py`: Alternative entrypoint/wrapper for running the project.
  - `src/env/`: Tetris game environment and wrappers.
  - `src/models/`: Network architecture and DQN agent logic.
  - `src/utils/`: Preprocessing helpers and visualization utilities.

## Features

- DQN with Double DQN
- Prioritized Experience Replay
- Custom reward shaping
- CNN-based Q-network

## Team

- Caleb Chu
- Christopher Yeh
- Edan Sasson
