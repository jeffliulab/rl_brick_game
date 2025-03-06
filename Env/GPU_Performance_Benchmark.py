import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical

# Configuration
ENV_NAME = "CartPole-v1"
HIDDEN_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPISODES = 100
MAX_STEPS = 500
GAMMA = 0.99
EPSILON = 0.2  # PPO clipping parameter
UPDATE_INTERVAL = 5
BENCHMARK_ITERATIONS = 5

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU()
        )
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) network
        self.critic = nn.Linear(HIDDEN_SIZE, 1)
    
    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)
    
    def get_action(self, state, training=True):
        state = torch.FloatTensor(state).to(device)
        probs, _ = self.forward(state)
        dist = Categorical(probs)
        
        if training:
            action = dist.sample()
        else:
            action = torch.argmax(probs)
            
        return action.item(), dist.log_prob(action)

# PPO Algorithm
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def select_action(self, state, training=True):
        action, log_prob = self.policy.get_action(state, training)
        state_tensor = torch.FloatTensor(state).to(device)
        _, value = self.policy(state_tensor)
        
        if training:
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value.item())
        
        return action
    
    def update(self):
        returns = self._compute_returns()
        advantages = self._compute_advantages()
        
        batch_states = torch.FloatTensor(np.array(self.states)).to(device)
        batch_actions = torch.LongTensor(np.array(self.actions)).to(device)
        batch_log_probs = torch.stack(self.log_probs).to(device)
        batch_returns = torch.FloatTensor(returns).to(device)
        batch_advantages = torch.FloatTensor(advantages).to(device)
        
        # Get current policy outputs
        probs, values = self.policy(batch_states)
        dist = Categorical(probs)
        curr_log_probs = dist.log_prob(batch_actions)
        
        # Calculate ratios
        ratios = torch.exp(curr_log_probs - batch_log_probs.detach())
        
        # Calculate surrogate losses
        surr1 = ratios * batch_advantages
        surr2 = torch.clamp(ratios, 1-EPSILON, 1+EPSILON) * batch_advantages
        
        # Calculate losses
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)
        
        # Update policy
        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        return loss.item()
    
    def store_transition(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    def _compute_returns(self):
        returns = []
        discounted_reward = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + GAMMA * discounted_reward
            returns.insert(0, discounted_reward)
            
        return returns
    
    def _compute_advantages(self):
        returns = self._compute_returns()
        advantages = []
        
        for ret, value in zip(returns, self.values):
            advantages.append(ret - value)
            
        return advantages

# Run training
def train():
    start_time = time.time()
    
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPO(state_dim, action_dim)
    
    # Training loop
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(reward, done)
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        # Update policy
        if episode % UPDATE_INTERVAL == 0:
            loss = agent.update()
            
        # Print episode information
        if episode % 10 == 0:
            print(f"Episode {episode}/{NUM_EPISODES}, Reward: {episode_reward:.2f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return training_time

# Run evaluation
def evaluate():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPO(state_dim, action_dim)
    
    state, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        episode_reward += reward
    
    return episode_reward

# GPU benchmark function
def benchmark_gpu():
    print("Starting GPU benchmark with PPO reinforcement learning...")
    
    # Run multiple training iterations to get an average
    training_times = []
    for i in range(BENCHMARK_ITERATIONS):
        print(f"\nBenchmark Iteration {i+1}/{BENCHMARK_ITERATIONS}")
        training_time = train()
        training_times.append(training_time)
    
    avg_training_time = sum(training_times) / len(training_times)
    
    # Calculate matrix multiplication performance
    matrix_size = 5000
    print(f"\nTesting matrix multiplication performance with {matrix_size}x{matrix_size} matrices...")
    
    matrix_times = []
    for i in range(3):
        matrix_a = torch.rand(matrix_size, matrix_size, device=device)
        matrix_b = torch.rand(matrix_size, matrix_size, device=device)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        result = torch.matmul(matrix_a, matrix_b)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()
        
        matrix_times.append(end_time - start_time)
    
    avg_matrix_time = sum(matrix_times) / len(matrix_times)
    
    # Print benchmark results
    print("\n===== GPU Benchmark Results =====")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / (1024**2):.2f} MB")
    
    print(f"\nPPO Training Time (avg of {BENCHMARK_ITERATIONS} runs): {avg_training_time:.2f} seconds")
    print(f"Matrix Multiplication ({matrix_size}x{matrix_size}): {avg_matrix_time:.4f} seconds")
    
    if device.type == "cuda":
        performance_score = 10000 / (avg_training_time * avg_matrix_time)
        print(f"\nPerformance Score: {performance_score:.2f}")
        
        if performance_score > 100:
            print("Rating: Excellent - Your GPU is very powerful!")
        elif performance_score > 50:
            print("Rating: Good - Your GPU has solid performance")
        elif performance_score > 20:
            print("Rating: Average - Decent performance for most RL tasks")
        else:
            print("Rating: Below average - Consider using a more powerful GPU for intensive RL tasks")
    else:
        print("\nYou're running on CPU. For better performance, consider using a CUDA-enabled GPU.")

if __name__ == "__main__":
    benchmark_gpu()