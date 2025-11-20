# Tetris-RL
Proximal Policy Optimization (PPO) for Tetris

# A README file that contains a 1 line description of each file in project/

# Workflow
1. **Preprocessing:** Board is preprocessed into a usable tensor
2. **CNN Analysis:** CNN processes the tensor into a feature vector
3. **Decision:** PPO takes the feature vector and determines the best action
4. **Action:** The selected action is sent to the environment
5. **Update:** Environment processes the move and updates the board state
6. **Loop:** Repeats from **Step 1** with the new board state until Game Over