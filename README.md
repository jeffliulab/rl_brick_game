# Breakout

This project is continuing in V2, implement value-based methods including Q-learning and DQN, to try to find a final solution. 

This is one of two parts of my Tufts University's Reinforcement Learning course's final project. This project focus on value-based methods, while another one focus on policy-based methods: [Walker2d](https://github.com/jeffliulab/walker2d)

## Prototype to evaluate Q-learning

This part is a simplified prototype of the Breakout game environment designed to evaluate the performance of Q-learning in a discrete setting. The game is implemented on a 10x10 grid, featuring a paddle, ball, and bricks. The paddle is fixed at the bottom, the ball moves with a fixed vertical velocity and a random horizontal direction, and the bricks are arranged in a single row at the top.

The demo's running graph is:
- <img src="docs/readme/simple_demo/simple_1.png" width="400" />

After learning, the agent can successfully play the game and eliminate all the bricks:
- <img src="docs/readme/simple_demo/simple_2.png" width="400" />

The learning curve is shown below:
- <img src="docs/readme/simple_demo/simple_3.png" width="800" />

### Environment Details
- **Grid**: 10x10 discrete grid.
- **Paddle**: Fixed on the bottom row with a width of 3 cells; can move left or right.
- **Ball**:
  - Starts directly above the paddle at a random horizontal position within the paddle's range.
  - Has an initial upward vertical velocity (-1) and a random horizontal velocity (-1 or 1).
- **Bricks**: Placed in the top row; represented by a 10-bit binary number where each bit indicates the presence of a brick.

### Q-Learning Agent
The Q-learning agent learns to play the game by interacting with the environment and updating its Q-table based on the rewards received:
- **Actions**:
  - `0`: No movement.
  - `1`: Move paddle left.
  - `2`: Move paddle right.
- **Reward Structure**:
  - **Step penalty**: -0.1 per move.
  - **Brick hit**: +5 reward (brick is removed and the ball's direction is reversed).
  - **Paddle hit**: +1 reward when the ball is successfully returned.
  - **Game loss**: -10 reward if the ball falls below the paddle.
  - **Game win**: +10 reward when all bricks are removed.

### Project Structure
- **BreakoutEnv**: Implements the game environment, handling paddle, ball, and brick initialization, state updates, collision detection, and reward computation.
- **BreakoutRenderer**: Uses Pygame to render the game state graphically, providing visual feedback during training or demonstration.
- **QLearningAgent**: Implements the Q-learning algorithm with methods to choose actions (using an epsilon-greedy strategy), update Q-values, and train the agent over multiple episodes.
- **Learning Curves**: Uses Matplotlib to plot rewards and success rates per episode, helping visualize the learning progress over time.
- **Main Function**: Integrates all components:
  1. Initializes the game environment and Q-learning agent.
  2. Trains the agent on the environment.
  3. Demonstrates the trained policy using a graphical interface.
  4. Plots the training progress.

### Dependencies
Please ensure that you have the following Python libraries installed:
- [Pygame](https://www.pygame.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

Install the dependencies using **`pip install pygame numpy matplotlib`**.

### Running the Prototype
Run the main script with **`python main.py`**. This command will:
1. Train the Q-learning agent in the Breakout environment.
2. Launch the Pygame graphical demo to showcase the learned policy.
3. Display learning curves after training.

### Parameter Configuration
- **Training Episodes**: Adjust the number of episodes by modifying `num_episodes` in `q_agent.train(env, num_episodes=100000)`.
- **Learning Rate, Discount Factor, and Initial Epsilon**: These parameters can be configured in the `QLearningAgent` constructor (`alpha`, `gamma`, `initial_epsilon`).
- **Rendering Settings**: Change the grid size and frame rate in the `BreakoutRenderer` to tailor the visual experience.

### Conclusion
This prototype provides a practical example for evaluating Q-learning in a simplified gaming environment. It is a useful foundation for further experiments and can be extended or modified for additional reinforcement learning research and applications.

### License
This project is licensed under the MIT License.
