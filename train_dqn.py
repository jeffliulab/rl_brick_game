import gym
import numpy as np
import torch
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from dqn_agent import DQNAgent

# Preprocess game frames to simplify the input
def preprocess_frame(frame):
    gray_frame = rgb2gray(frame)
    resized_frame = resize(gray_frame, (84, 84))
    return resized_frame

# Stack frames for state representation
def stack_frames(frames, frame, is_new_episode):
    if is_new_episode:
        frames = deque([frame] * 4, maxlen=4)
    else:
        frames.append(frame)
    return np.stack(frames, axis=0), frames

# Main training loop
# env = gym.make('Breakout-v4', render_mode='human')  # Rendering disabled for faster training
env = gym.make('Breakout-v4')
agent = DQNAgent(env.action_space.n)
num_episodes = 2000
batch_size = 32
update_target_every = 5
min_replay_size = 5000  # Minimum memory size before replay starts
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995

for episode in range(num_episodes):
    # Reset the environment and get only the initial observation
    state, _ = env.reset()
    state = preprocess_frame(state)  # Convert to grayscale and resize
    state, frames = stack_frames([], state, True)
    done = False
    total_reward = 0

    # Force the first action to be "fire" (usually action `1`) to start the game
    action = 1
    next_state, reward, terminated, truncated, _ = env.step(action)
    next_state = preprocess_frame(next_state)
    next_state, frames = stack_frames(frames, next_state, False)
    total_reward += reward

    state = next_state  # Update state after firing to begin the game

    while not done:
        # Select action with epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.choose_action(state)

        # Step the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Preprocess and stack the next state
        next_state = preprocess_frame(next_state)
        next_state, frames = stack_frames(frames, next_state, False)

        # Store transition in memory and update state
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Replay if memory is sufficient
        if len(agent.memory) > min_replay_size:
            agent.replay(batch_size)

    # Update target network
    if episode % update_target_every == 0:
        agent.target_model.load_state_dict(agent.model.state_dict())

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {episode}, Total Reward: {total_reward}")

# Save the model
torch.save(agent.model.state_dict(), "dqn_breakout_model.pth")
print("模型已保存为 dqn_breakout_model.pth")

env.close()