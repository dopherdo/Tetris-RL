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

## Features

- DQN with Double DQN
- Prioritized Experience Replay
- Custom reward shaping
- CNN-based Q-network

## Team

- Caleb Chu
- Christopher Yeh
- Edan Sasson
