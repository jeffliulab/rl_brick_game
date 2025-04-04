#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep Q-Network (DQN) to play Breakout (ALE/Breakout-v5).

All printed messages and figure texts are in English.
All code comments are in Chinese.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======== Gym & ALE ========
import gymnasium as gym
import ale_py

# ======== PyTorch ========
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ======== Other utils ========
import cv2
from tqdm import trange
from collections import deque
from pandas import DataFrame

# ========== Part 1: Global configurations, random seeds, device setup ==========

# 以下注释均为中文，为了方便理解：
# 设置随机种子，保证实验可复现。并选择CPU或GPU作为计算设备。
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 用于绘制曲线时的指数加权平均函数 (moving average)
def moving_average(x, span, **kw):
    return (
        DataFrame({"x": np.asarray(x)})
        .x.ewm(span=span, **kw)
        .mean()
        .values
    )

# ========== Part 2: Environment initialization ==========

def initialize_environment(render_mode="rgb_array", frameskip=4):
    """
    Initialize an Atari Breakout environment with given frameskip.
    Prints in English. If environment fails to load, exit.
    """
    try:
        env = gym.make(
            "ALE/Breakout-v5",
            render_mode=render_mode,
            frameskip=frameskip
        )
        print("Environment loaded successfully!")
        return env
    except Exception as e:
        print(f"Failed to load environment: {e}")
        sys.exit(1)

# 在Breakout里，我们常常需要在开局进行一次FIRE动作，以避免训练期间学会“开局发球”。
class AutoFireResetEnv(gym.Wrapper):
    """
    After reset(), automatically fire once.
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, reward, done, truncated, info = self.env.step(1)  # action=1 => FIRE
        if done or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

# ========== Part 3: Preprocessing wrappers ==========

class PreprocessAtari(gym.ObservationWrapper):
    """
    Convert color Atari image to grayscale [84x84], normalized to [0,1].
    """
    def __init__(self, env):
        super().__init__(env)
        self.img_size = (84, 84)

        # 修改观察空间为 (84, 84, 1) 浮点类型
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.img_size[0], self.img_size[1], 1),
            dtype=np.float32
        )

    def observation(self, obs):
        # 裁剪掉画面上下无用区域
        obs = obs[34:-16, :, :]
        # 缩放至 (84, 84)
        obs = cv2.resize(obs, self.img_size)
        # 转为灰度图 (单通道)
        obs = obs.mean(axis=-1, keepdims=True)
        # 归一化至 [0,1]
        obs = obs.astype(np.float32) / 255.0
        return obs


class FrameBuffer(gym.Wrapper):
    """
    Maintain a buffer of n_frames, stacking them along channel dimension.
    dim_order='pytorch' => (C, H, W).
    """
    def __init__(self, env, n_frames=4, dim_order='pytorch'):
        super().__init__(env)
        self.dim_order = dim_order
        height, width, channels = env.observation_space.shape

        if dim_order == 'pytorch':
            # PyTorch shape: (C, H, W)
            self.frame_shape = (channels, height, width)
            obs_shape = (channels * n_frames, height, width)
        else:
            raise ValueError('dim_order must be "pytorch" for this code.')

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_shape,
            dtype=np.float32
        )
        self.n_frames = n_frames
        self.framebuffer = np.zeros(obs_shape, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(obs)
        return self.framebuffer, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        done = done or truncated
        self.update_buffer(obs)
        return self.framebuffer, reward, done, info

    def update_buffer(self, obs):
        # 将[H,W,C]的图像转为[C,H,W]并拼接到framebuffer最前面，移除最旧的一帧
        obs = np.transpose(obs, (2, 0, 1))
        old = self.framebuffer[:-self.frame_shape[0], :, :]
        self.framebuffer = np.concatenate([obs, old])

# ========== Part 4: DQN Network, Replay Buffer, Agent ==========

class DQN(nn.Module):
    """
    A Convolutional Neural Network for feature extraction from stacked frames.
    Output: Q(s, a) for each action.
    """
    def __init__(self, input_shape, n_actions):
        super().__init__()
        # ========== 卷积层结构 ==========
        # 通常参考DeepMind的Nature DQN配置:
        # Conv2D(32, 8x8, stride=4) => Conv2D(64, 4x4, stride=2) => Conv2D(64, 3x3, stride=1)
        # 之后展平再接全连接层512维, 最后输出n_actions维度
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # 计算卷积输出大小
        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        # 这里通过构造一个shape一致的全零张量，前向传播一次来获得输出维度
        test_inp = torch.zeros(1, *shape)
        out = self.conv(test_inp)
        return int(np.prod(out.size()))

    def forward(self, x):
        # 前向传播: 卷积 -> 展平 -> 全连接
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)

class ReplayBuffer:
    """
    A simple replay buffer storing (s, a, r, s_next, done).
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        # 从buffer中随机取出batch_size条数据
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)

        s = np.array(s, dtype=np.float32)
        a = np.array(a, dtype=np.int64)
        r = np.array(r, dtype=np.float32)
        s_next = np.array(s_next, dtype=np.float32)
        done = np.array(done, dtype=np.float32)

        # 转为torch张量并放到同一个device上
        s_t = torch.from_numpy(s).to(device)
        a_t = torch.from_numpy(a).to(device)
        r_t = torch.from_numpy(r).to(device)
        s_next_t = torch.from_numpy(s_next).to(device)
        done_t = torch.from_numpy(done).to(device)

        return s_t, a_t, r_t, s_next_t, done_t


class DQNAgent:
    """
    DQN agent with a policy network and a target network, plus epsilon-greedy.
    """
    def __init__(self, state_shape, n_actions, epsilon=1.0):
        self.epsilon = epsilon
        self.n_actions = n_actions

        # 策略网络 (policy_net)
        self.policy_net = DQN(state_shape, n_actions).to(device)
        # 目标网络 (target_net)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.update_target_network()

        # target_net参数冻结，不参加梯度计算
        for p in self.target_net.parameters():
            p.requires_grad = False

    def update_target_network(self):
        # 复制policy_net的参数到target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_qvalues(self, states):
        with torch.no_grad():
            inp = torch.from_numpy(states).float().to(device)
            qvals = self.policy_net(inp)
            return qvals.cpu().numpy()

    def sample_actions(self, qvalues):
        # \epsilon-greedy: 以epsilon概率随机动作，否则选Q值最大的动作
        batch_size = qvalues.shape[0]
        random_actions = np.random.choice(self.n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=1)
        explore = np.random.rand(batch_size) < self.epsilon
        return np.where(explore, random_actions, best_actions)

# ========== Part 5: Training & Evaluation functions ==========

def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """
    Let the agent play 'n_games' in the environment, return average total reward.
    If 'greedy=True', use argmax(Q); else use epsilon-greedy.
    """
    rewards = []
    for _ in range(n_games):
        obs, info = env.reset()
        ep_reward = 0
        for _ in range(t_max):
            state_batch = np.array([obs], dtype=np.float32)
            qvals = agent.get_qvalues(state_batch)
            if greedy:
                action = qvals.argmax(axis=1)[0]
            else:
                action = agent.sample_actions(qvals)[0]
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
    return np.mean(rewards)


def play_and_record(agent, env, replay_buffer, n_steps=1):
    """
    Interact with the environment for n_steps. Store transitions into the replay buffer.
    Returns the sum of rewards over these n_steps.
    """
    s = env.framebuffer
    reward_sum = 0.0

    for _ in range(n_steps):
        state_batch = np.array([s], dtype=np.float32)
        qvals = agent.get_qvalues(state_batch)
        action = agent.sample_actions(qvals)[0]

        s_next, reward, done, info = env.step(action)
        replay_buffer.add(s, action, reward, s_next, float(done))

        reward_sum += reward

        if done:
            s_reset, _ = env.reset()
            s = s_reset
        else:
            s = s_next

    return reward_sum

def train_on_batch(agent, optimizer, replay_buffer, batch_size=64, gamma=0.99):
    """
    Sample a batch from replay buffer and do one gradient step.

    这里我们采用的是时序差分 (Temporal Difference) 学习:
    在Q-learning的更新公式中, 目标值 y 可写为:
        y = r + gamma * max_{a'} Q_target(s', a')
    在DQN中, 我们用 target_net 来计算 max_{a'} Q_target(s', a'), 以减少自举偏差.

    损失函数 (loss) 通常用 HuberLoss (smooth_l1_loss), 表达形式大致是:
        L = Huber( Q_policy(s,a) - y )

    其中 Q_policy(s,a) 是当前策略网络的输出, y 是上述TD目标值.
    """
    if len(replay_buffer) < batch_size:
        return 0.0

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Q_policy(s,a)
    qvals = agent.policy_net(states)
    current_q = qvals.gather(1, actions.unsqueeze(1)).squeeze(1)

    # 计算TD目标 y = r + gamma * max_{a'} Q_target(s', a') * (1 - done)
    with torch.no_grad():
        qvals_next = agent.target_net(next_states)
        max_q_next = qvals_next.max(dim=1)[0]
        target_q = rewards + gamma * max_q_next * (1 - dones)

    # 使用smooth_l1_loss(Huber Loss)来衡量 current_q 与 target_q 的差异
    loss = F.smooth_l1_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()

    # 可选：梯度裁剪，避免梯度爆炸
    for param in agent.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()
    return loss.item()


# ========== Part 6: Main training script ==========

def main():
    # 1) 初始化环境
    raw_env = initialize_environment(render_mode="rgb_array", frameskip=4)
    raw_env = AutoFireResetEnv(raw_env)
    env = PreprocessAtari(raw_env)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')

    # 2) 准备DQNAgent, 经验回放等
    n_actions = env.action_space.n
    print("Number of actions:", n_actions)
    state_shape = env.observation_space.shape

    agent = DQNAgent(state_shape, n_actions, epsilon=1.0)
    optimizer = optim.Adam(agent.policy_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(capacity=70000)

    obs, info = env.reset()

    # 先预填充一些回放缓冲区, 让网络能sample到足够多样的经验
    print("Filling replay buffer, please wait...")
    play_and_record(agent, env, replay_buffer, n_steps=10000)

    # 下面这些列表用于记录训练/评估指标, 以便可视化
    mean_rw_history = []       # 评估时的平均reward
    loss_history = []          # 训练过程中的TD损失
    train_reward_history = []  # 每次迭代play_and_record返回的奖励
    epsilon_history = []       # epsilon随迭代变化的曲线

    # 超参数设定
    gamma = 0.99
    batch_size = 64
    target_update_freq = 500
    epsilon_decay = 0.999
    min_epsilon = 0.01

    eval_freq = 5000        # 每多少迭代做一次评估
    num_iterations = 200000 # 总迭代次数

    print("Start DQN training...")
    for i in trange(num_iterations):
        # 每次迭代先与环境交互10步
        train_reward = play_and_record(agent, env, replay_buffer, n_steps=10)
        train_reward_history.append(train_reward)

        # 然后sample一个batch做一次训练
        loss = train_on_batch(agent, optimizer, replay_buffer, batch_size, gamma)
        loss_history.append(loss)

        # 隔一定迭代频率更新target_net, 并衰减epsilon
        if i % target_update_freq == 0:
            agent.update_target_network()
            agent.epsilon = max(agent.epsilon * epsilon_decay, min_epsilon)

        # 记录当前epsilon
        epsilon_history.append(agent.epsilon)

        # 定期评估
        if i % eval_freq == 0:
            eval_env = initialize_environment(render_mode="rgb_array", frameskip=4)
            eval_env = AutoFireResetEnv(eval_env)
            eval_env = PreprocessAtari(eval_env)
            eval_env = FrameBuffer(eval_env, n_frames=4, dim_order='pytorch')

            mean_reward = evaluate(eval_env, agent, n_games=3, greedy=True)
            mean_rw_history.append(mean_reward)

            print(f"Iteration {i}, Eval average reward: {mean_reward:.2f}, "
                  f"Replay size: {len(replay_buffer)}, "
                  f"Epsilon: {agent.epsilon:.3f}")

        # 每10000次迭代保存一次模型并测试一局
        save_freq = 10000
        if i > 0 and i % save_freq == 0:
            os.makedirs("output/models", exist_ok=True)
            model_path = f"output/models/dqn_model_atari_{i}.pt"
            torch.save(agent.policy_net.state_dict(), model_path)
            print(f"Model saved to {model_path}")

            # 演示一局
            demo_env = initialize_environment(render_mode="rgb_array", frameskip=4)
            demo_env = AutoFireResetEnv(demo_env)
            demo_env = PreprocessAtari(demo_env)
            demo_env = FrameBuffer(demo_env, n_frames=4, dim_order='pytorch')

            demo_reward = evaluate(demo_env, agent, n_games=1, greedy=True)
            print(f"[Demo] iteration {i}, single-episode reward: {demo_reward:.2f}")
            demo_env.close()

    # ========== 训练结束, 保存最终模型 ==========
    os.makedirs("output/models", exist_ok=True)
    torch.save(agent.policy_net.state_dict(), "output/models/dqn_model_atari_final.pt")
    print("Final model saved.")

    # ========== 录制一段评估视频 ==========

    agent.epsilon = 0  # 测试时设为贪心
    os.makedirs("output/videos", exist_ok=True)

    video_env = gym.wrappers.RecordVideo(
        initialize_environment(render_mode="rgb_array"),
        video_folder="output/videos",
        episode_trigger=lambda e: True
    )
    video_env = AutoFireResetEnv(video_env)
    video_env = PreprocessAtari(video_env)
    video_env = FrameBuffer(video_env, n_frames=4, dim_order='pytorch')

    final_rewards = [evaluate(video_env, agent, n_games=1, greedy=True)]
    video_env.close()
    print(f"Final reward (one episode): {np.mean(final_rewards):.2f}")

    # ========== 绘制一张包含4个子图的大图, 并保存 ==========

    os.makedirs("output", exist_ok=True)
    plt.figure(figsize=(14, 10))  # 画布更大, 便于看清
    # 为了让图像更清晰, 我们在savefig时还会指定 dpi=200

    # (1) 评估平均回报
    plt.subplot(2, 2, 1)
    plt.title("Evaluation reward (average of 3 games)")
    plt.plot(mean_rw_history)
    plt.grid()

    # (2) 训练期间的奖励 (moving average)
    plt.subplot(2, 2, 2)
    plt.title("Training reward (moving average)")
    plt.plot(moving_average(train_reward_history, span=100, min_periods=100))
    plt.grid()

    # (3) 训练损失曲线 (moving average)
    plt.subplot(2, 2, 3)
    plt.title("Training loss (smooth L1) - moving average")
    plt.plot(moving_average(loss_history, span=100, min_periods=100))
    plt.grid()

    # (4) Epsilon 随迭代的变化
    plt.subplot(2, 2, 4)
    plt.title("Epsilon over iterations")
    plt.plot(epsilon_history)
    plt.grid()

    plt.tight_layout()
    plt.savefig("output/dqn_all_curves.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
