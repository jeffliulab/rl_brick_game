# evaluate_dqn.py
import gym
import torch
import numpy as np
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
import time
from dqn_model import DQN  # Import the DQN class

class BreakoutDQNAgent:
    def __init__(self, action_space=4, frame_stack_size=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(action_space).to(self.device)
        self.frame_stack_size = frame_stack_size
        self.frames = deque(maxlen=frame_stack_size)
        
    def preprocess_frame(self, frame):
        """Convert frame to grayscale and resize"""
        # Normalize pixel values to [0, 1]
        gray_frame = rgb2gray(frame)
        resized_frame = resize(gray_frame, (84, 84), anti_aliasing=True)
        return np.array(resized_frame, dtype=np.float32)
    
    def stack_frames(self, frame, is_new_episode=False):
        """Stack frames for temporal information"""
        if is_new_episode:
            self.frames.clear()
            for _ in range(self.frame_stack_size):
                self.frames.append(frame)
        else:
            self.frames.append(frame)
            
        # Stack frames along first dimension
        stacked_frames = np.stack(self.frames, axis=0)
        return stacked_frames
    
    def get_action(self, state):
        """Get action from model using current state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def load_model(self, path):
        """Load pretrained model weights"""
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            print("Model loaded successfully")
        except FileNotFoundError:
            print(f"Error: Model file '{path}' not found.")
            raise
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

def main():
    try:
        # Initialize environment and agent
        env = gym.make('ALE/Breakout-v5', render_mode='human')
        agent = BreakoutDQNAgent()
        agent.load_model("dqn_breakout_model_2000.pth")
        
        # Game loop
        episode_reward = 0
        lives = env.ale.lives()
        
        # Initial reset
        state, _ = env.reset()
        state = agent.preprocess_frame(state)
        stacked_state = agent.stack_frames(state, is_new_episode=True)
        
        # Fire to start the game
        env.step(1)  # Fire action
        time.sleep(0.5)  # Wait for ball to appear
        
        while True:
            # Get action from agent
            action = agent.get_action(stacked_state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Check for life lost
            current_lives = env.ale.lives()
            life_lost = current_lives < lives
            lives = current_lives
            
            if life_lost:
                # Fire to restart after losing a life
                time.sleep(0.5)
                env.step(1)
                
            # Preprocess next state
            next_state = agent.preprocess_frame(next_state)
            stacked_state = agent.stack_frames(next_state, is_new_episode=life_lost)
            
            episode_reward += reward
            
            # Check for game over
            if terminated or truncated:
                print(f"Game Over! Total Score: {episode_reward}")
                break
                
            env.render()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        env.close()

if __name__ == "__main__":
    main()