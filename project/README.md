# Tetris-RL Project Files

One-line descriptions for every file in `project/`.

- `checkpoint_500k.pt`: Trained DQN checkpoint (500k steps) kept locally for demo.
- `project.ipynb`: Demo notebook that loads the checkpoint and compares agents.
- `project.html`: HTML export of the demo notebook with outputs saved.
- `README.md`: This file.
- `src/__init__.py`: Marks `src` as a package.
- `src/main.py`: Placeholder entry point (unused).
- `src/train.py`: Training loop for the DQN agent with configurable flags.
- `src/evaluate.py`: Evaluation script to run trained agents vs. baselines.
- `src/env/__init__.py`: Environment package initializer.
- `src/env/tetris_env.py`: Tetris environment wrapper with shaped rewards.
- `src/env/composite_wrapper.py`: Maps 40 composite actions to atomic env steps.
- `src/env/curriculum_wrapper.py`: Experimental curriculum wrapper (not used).
- `src/models/__init__.py`: Models package initializer.
- `src/models/dqn_agent.py`: DQN agent with Double DQN and PER.
- `src/models/dqn_network.py`: CNN Q-network (3 conv layers + 2 FC heads).
- `src/utils/__init__.py`: Utils package initializer.
- `src/utils/preprocessing.py`: Observation and board preprocessing helpers.
- `src/utils/visualization.py`: Plotting helpers for metrics and learning curves.

