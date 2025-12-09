# Tetris-RL Project Files

This directory contains all files necessary to run the project demonstration notebook.

## Main Files

- **project.ipynb**: Jupyter notebook demonstrating the trained DQN agent with evaluation and visualization
- **checkpoint_500k.pt**: Trained DQN model checkpoint from 500,000 training steps
- **requirements.txt**: Python package dependencies for the project

## Source Code (src/)

### Environment Module (src/env/)
- **__init__.py**: Module initialization and factory functions for creating Tetris environments
- **tetris_env.py**: Custom Tetris environment wrapper with reward shaping (line clears, holes, bumpiness, height penalties)
- **composite_wrapper.py**: Wrapper that converts 40 composite actions (rotation + column) into atomic Tetris actions
- **curriculum_wrapper.py**: Experimental curriculum learning wrapper (not used in final model)

### Models Module (src/models/)
- **__init__.py**: Module initialization for DQN agent and network classes
- **dqn_agent.py**: DQN agent implementation with Double DQN, Prioritized Experience Replay, and fixed Q-targets
- **dqn_network.py**: CNN-based Q-network architecture (3 conv layers → 512 → 256 → 40 outputs)

### Utilities (src/utils/)
- **__init__.py**: Module initialization for utility functions
- **preprocessing.py**: Observation preprocessing functions to convert board state to neural network input
- **visualization.py**: Plotting utilities for learning curves and performance metrics

### Training and Evaluation (src/)
- **__init__.py**: Package initialization
- **train.py**: Main training script for DQN agent with configurable hyperparameters
- **evaluate.py**: Evaluation script to test trained models and compare with baselines
- **main.py**: Entry point script (currently empty)

