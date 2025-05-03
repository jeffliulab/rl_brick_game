#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep Q-Network (DQN) to play Breakout (ALE/Breakout-v5).

All printed messages and figure texts are in English.
All code comments are in Chinese.

使用方法：
先确定好checkpoint位置和FINAL EPSILON的数值, 然后确定Target Iteration Times后, 执行参数运行脚本:
python src/dqn_train.py --resume [Checkpoint File Path] --iterations [Target Iteration Times]

example:
python .\src\dqn_train.py --resume .\src\dqn_checkpoint_1950000.pt --iterations 3000000
"""

# 全局训练设置
EPSILON_FINAL_SETTING = 0.04  # 最终epsilon值

# CHECKPOINT恢复设置
CONTINUE_TRAINING = True
CHECKPOINT = "./output/checkpoints/dqn_checkpoint_1950000.pt"
TARGET_EPISODES = 3000000

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import datetime

# ======== PyTorch ========
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ======== Other utils ========
from tqdm import tqdm, trange

# ======== 自定义模块 ========
# 内存管理模块
from memory_manage import (
    limit_memory_usage, get_memory_usage, print_memory_usage, force_gc,
    analyze_gpu_tensors, setup_memory_leak_detection, setup_logger,
    MemoryEfficientReplayBuffer, check_and_manage_memory
)

# Atari环境模块
from atari_environment import (
    initialize_environment, make_evaluation_env, make_recording_env
)

# 检查点管理模块
from checkpoint_manage import (
    save_checkpoint, save_model, load_checkpoint, resume_from_checkpoint,
    continue_training_from_checkpoint
)

# 导入绘图和保存相关函数
try:
    from plot_utils import (
        save_training_plots, save_training_history, moving_average
    )
except ImportError:
    print("Warning: plot_utils module not found. Training will continue but plots won't be generated.")
    
    # 创建空的函数避免错误
    def save_training_plots(*args, **kwargs): pass
    def save_training_history(*args, **kwargs): pass
    def moving_average(*args, **kwargs): return []

TOTAL_ITERATION_TIMES = 2000000  # 默认总迭代次数，可以通过参数调整

# ========== Part 1: Global configurations, random seeds, device setup ==========

# 设置随机种子，保证实验可复现。并选择CPU或GPU作为计算设备。
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== Part 2: DQN Network and Agent ==========

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

# ========== Part 3: Training & Evaluation functions ==========

def evaluate(env, agent, n_games=1, greedy=False, t_max=10000, logger=None):
    """
    让智能体在环境中玩n_games次游戏，返回平均总奖励。
    如果greedy=True，使用argmax(Q)；否则使用epsilon-greedy策略。
    
    Args:
        env: 游戏环境
        agent: DQN智能体
        n_games: 游戏次数
        greedy: 是否使用贪心策略
        t_max: 每局游戏的最大步数
        logger: 日志记录器
        
    Returns:
        平均奖励
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
    
    avg_reward = np.mean(rewards)
    if logger:
        logger.info(f"Evaluation: average reward over {n_games} games: {avg_reward:.2f}")
    return avg_reward


def play_and_record(agent, env, replay_buffer, n_steps=1, logger=None):
    """
    与环境交互n_steps步，并将转换存储到回放缓冲区中。
    返回这n_steps步骤中的奖励总和。
    
    Args:
        agent: DQN智能体
        env: 游戏环境
        replay_buffer: 经验回放缓冲区
        n_steps: 交互步数
        logger: 日志记录器
        
    Returns:
        总奖励
    """
    # 确保有正确的初始状态
    try:
        s = env.framebuffer
    except AttributeError:  # 如果framebuffer尚未初始化
        s, _ = env.reset()
    
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

def train_on_batch(agent, optimizer, replay_buffer, batch_size=64, gamma=0.99, logger=None):
    """
    从回放缓冲区中采样一批数据并进行一次梯度步骤。
    包含内存问题的错误处理。
    
    Args:
        agent: DQN智能体
        optimizer: 优化器
        replay_buffer: 经验回放缓冲区
        batch_size: 批量大小
        gamma: 折扣因子
        logger: 日志记录器
        
    Returns:
        损失值，如果训练失败则返回0
    """
    if len(replay_buffer) < batch_size:
        if logger:
            logger.debug("Replay buffer too small for sampling")
        return 0.0

    # 获取一个批量的数据，如果内存不足会自动减小批量大小
    batch_data = replay_buffer.sample(batch_size)
    
    # 如果采样失败，返回0损失
    if batch_data is None:
        message = "Batch sampling returned None, skipping training step"
        if logger:
            logger.warning(message)
        else:
            print(message)
        return 0.0
        
    states, actions, rewards, next_states, dones = batch_data

    try:
        # Q_policy(s,a) - 当前状态动作值
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

        # 梯度裁剪，避免梯度爆炸
        for param in agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        optimizer.step()
        
        # 显式清理中间张量
        del states, actions, rewards, next_states, dones, qvals, current_q, qvals_next, max_q_next, target_q
        
        # 返回损失值
        loss_value = loss.item()
        del loss
        
        # 强制同步GPU操作，确保所有计算完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        return loss_value
        
    except (MemoryError, RuntimeError) as e:
        message = f"Training error: {e}"
        if logger:
            logger.error(message)
        else:
            print(message)
        force_gc(logger)  # 强制垃圾回收
        return 0.0

# ========== Part 4: Main training functions ==========

def main():
    """
    主训练函数，从头开始训练DQN。
    """
    # 检查是否需要从断点恢复训练，但避免重复调用
    global main_has_continue_called
    if CONTINUE_TRAINING and os.path.exists(CHECKPOINT) and not main_has_continue_called:
        print(f"Continue training mode enabled, loading checkpoint: {CHECKPOINT}")
        # 设置标记避免重复调用
        main_has_continue_called = True
        # 使用设置的目标迭代次数
        continue_training_from_checkpoint(
            checkpoint_path=CHECKPOINT,
            num_iterations=TARGET_EPISODES,
            DQNAgent=DQNAgent,
            initialize_environment=initialize_environment,
            make_evaluation_env=make_evaluation_env,
            make_recording_env=make_recording_env,
            play_and_record=play_and_record,
            train_on_batch=train_on_batch,
            evaluate=evaluate,
            device=device,
            MemoryBufferClass=MemoryEfficientReplayBuffer,
            setup_logger=setup_logger,
            setup_memory_leak_detection=setup_memory_leak_detection,
            check_and_manage_memory=check_and_manage_memory,
            force_gc=force_gc,
            print_memory_usage=print_memory_usage,
            analyze_gpu_tensors=analyze_gpu_tensors,
            save_checkpoint_fn=save_checkpoint,
            save_model_fn=save_model,
            save_training_plots_fn=save_training_plots,
            save_training_history_fn=save_training_history,
            epsilon_final=EPSILON_FINAL_SETTING,
            total_iteration_times=TOTAL_ITERATION_TIMES,
            main_function=main,
            main_has_continue_called=main_has_continue_called
        )
        return
    
    # 限制最大内存使用
    limit_memory_usage(28)
    
    # 设置日志
    logger = setup_logger()
    logger.info("Starting DQN training script")
    
    # 启动内存泄漏检测
    leak_detector = setup_memory_leak_detection(interval=50000)
    
    # 1) 初始化环境
    env = initialize_environment(game_name="ALE/Breakout-v5", render_mode="rgb_array", frameskip=4)

    # 2) 准备DQNAgent, 经验回放等
    n_actions = env.action_space.n
    logger.info(f"Number of actions: {n_actions}")
    state_shape = env.observation_space.shape

    # 打印初始内存使用情况
    print_memory_usage("Initial", logger)

    # 更合理的回放缓冲区大小，防止内存溢出
    replay_buffer_size = 100000  # 从500,000减小到100,000
    
    agent = DQNAgent(state_shape, n_actions, epsilon=1.0)
    optimizer = optim.Adam(agent.policy_net.parameters(), lr=1e-4)
    # 添加学习率调度器 - 每100万步学习率减半
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.5)
    replay_buffer = MemoryEfficientReplayBuffer(capacity=replay_buffer_size, logger=logger, device=device)

    obs, info = env.reset()

    # 先预填充一些回放缓冲区, 让网络能sample到足够多样的经验
    logger.info("Filling replay buffer with 50000 steps, please wait...")
    pbar = tqdm(total=50000, desc="Buffer Filling")
    steps_done = 0
    while steps_done < 50000:
        steps_to_do = min(1000, 50000 - steps_done)
        reward = play_and_record(agent, env, replay_buffer, n_steps=steps_to_do, logger=logger)
        steps_done += steps_to_do
        pbar.update(steps_to_do)
        
        # 每10000步检查一次内存
        if steps_done % 10000 == 0:
            print_memory_usage(f"Buffer fill {steps_done} steps", logger)
            force_gc(logger)  # 强制垃圾回收
            
    pbar.close()

    # 下面这些列表用于记录训练/评估指标, 以便可视化
    mean_rw_history = []       # 评估时的平均reward
    loss_history = []          # 训练过程中的TD损失
    train_reward_history = []  # 每次迭代play_and_record返回的奖励
    epsilon_history = []       # epsilon随迭代变化的曲线
    memory_usage_history = []  # 内存使用历史

    # 超参数设定
    gamma = 0.99
    batch_size = 16  # 从32减小到16
    target_update_freq = 10000
    
    # 线性衰减的epsilon参数
    epsilon_start = 1.0
    epsilon_final = EPSILON_FINAL_SETTING
    epsilon_decay_steps = 1000000  # 前100万步线性衰减

    eval_freq = 50000        # 每多少迭代做一次评估
    memory_check_freq = 5000  # 每多少迭代检查一次内存 (从10000改为5000，更频繁检查)
    num_iterations = TOTAL_ITERATION_TIMES # 总迭代次数

    # 添加自动保存点，用于训练中断后恢复
    checkpoint_dir = "output/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 修改保存频率设置
    early_save_points = [50, 100]  # 添加早期保存点，用于测试保存机制
    regular_save_freq = 50000      # 将常规保存频率改为50000

    # 内存管理阈值
    MEMORY_WARNING_THRESHOLD = 80.0  # 80%内存使用就开始警告
    MEMORY_CRITICAL_THRESHOLD = 85.0  # 85%就采取行动

    logger.info("Start DQN training...")
    for i in trange(num_iterations):
        # 检查并管理内存
        batch_size = check_and_manage_memory(
            i, logger, replay_buffer, batch_size, 
            MEMORY_WARNING_THRESHOLD, MEMORY_CRITICAL_THRESHOLD
        )
                
        # 每次迭代先与环境交互10步
        try:
            train_reward = play_and_record(agent, env, replay_buffer, n_steps=10, logger=logger)
            train_reward_history.append(train_reward)
        except RuntimeError as e:
            logger.error(f"Error during play_and_record: {e}")
            force_gc(logger)
            continue

        # 然后sample一个batch做一次训练
        try:
            loss = train_on_batch(agent, optimizer, replay_buffer, batch_size, gamma, logger=logger)
            loss_history.append(loss)
        except RuntimeError as e:
            logger.error(f"Error during train_on_batch: {e}")
            force_gc(logger)
            continue

        # 线性衰减epsilon
        if i < epsilon_decay_steps:
            agent.epsilon = epsilon_start - (epsilon_start - epsilon_final) * (i / epsilon_decay_steps)
        else:
            agent.epsilon = epsilon_final

        # 隔一定迭代频率更新target_net
        if i % target_update_freq == 0:
            agent.update_target_network()
            logger.info(f"Iteration {i}: Target network updated")
            
        # 每1000步调整一次学习率
        if i % 1000 == 0:
            scheduler.step()

        # 记录当前epsilon
        epsilon_history.append(agent.epsilon)
        
        # 定期监控内存使用
        if i % memory_check_freq == 0:
            memory_usage = print_memory_usage(f"Iteration {i}", logger)
            memory_usage_history.append(memory_usage)
            logger.info(f"Iteration {i}, Buffer size: {len(replay_buffer)}, Epsilon: {agent.epsilon:.3f}")
            
            # 检查内存泄漏
            leak_detector(i, logger)
            
            # 强制垃圾回收
            force_gc(logger)

        # 定期评估
        if i % eval_freq == 0:
            # 先进行垃圾回收以确保评估有足够内存
            force_gc(logger)
            
            try:
                # 创建独立的评估环境
                eval_env = make_evaluation_env(game_name="ALE/Breakout-v5", render_mode="rgb_array", frameskip=4)

                mean_reward = evaluate(eval_env, agent, n_games=3, greedy=True, logger=logger)
                mean_rw_history.append(mean_reward)

                logger.info(f"Iteration {i}, Eval average reward: {mean_reward:.2f}, "
                          f"Replay size: {len(replay_buffer)}, "
                          f"Epsilon: {agent.epsilon:.3f}, "
                          f"LR: {scheduler.get_last_lr()[0]:.6f}")
                          
                # 关闭评估环境以释放资源
                eval_env.close()
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                
            # 评估后再次进行垃圾回收
            force_gc(logger)
        
        # 检查是否是早期保存点或常规保存点
        should_save = (i in early_save_points) or (i > 0 and i % regular_save_freq == 0)
        
        if should_save:
            # 强制垃圾回收以确保有足够内存
            force_gc(logger)
            
            # 调用绘图和保存函数
            try:
                save_checkpoint(i, agent, optimizer, scheduler, 
                              mean_rw_history, loss_history, train_reward_history, 
                              epsilon_history, memory_usage_history, 
                              checkpoint_dir, logger)
                              
                save_model(i, agent, logger)
                
                save_training_plots(i, mean_rw_history, train_reward_history, loss_history, 
                                  epsilon_history, memory_usage_history, memory_check_freq, 
                                  logger)
                                  
                save_training_history(i, mean_rw_history, train_reward_history, loss_history, 
                                    epsilon_history, memory_usage_history, logger)
            except Exception as e:
                logger.error(f"Failed to save at iteration {i}: {e}")

    # ========== 训练结束, 保存最终模型 ==========
    force_gc(logger)  # 确保有足够内存保存模型
    
    try:
        # 保存最终模型
        save_model("final", agent, logger)
        
        # 保存最终训练曲线
        save_training_plots("final", mean_rw_history, train_reward_history, loss_history, 
                           epsilon_history, memory_usage_history, memory_check_freq, logger)
                           
        # 保存最终训练历史数据
        save_training_history("final", mean_rw_history, train_reward_history, loss_history, 
                             epsilon_history, memory_usage_history, logger)
    except Exception as e:
        logger.error(f"Failed to save final results: {e}")

    # ========== 录制一段评估视频 ==========
    try:
        agent.epsilon = 0  # 测试时设为贪心
        os.makedirs("output/videos", exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_folder = f"output/videos/run_{timestamp}"

        # 使用视频录制环境
        video_env = make_recording_env(
            video_folder=video_folder,
            game_name="ALE/Breakout-v5", 
            frameskip=4
        )

        final_rewards = [evaluate(video_env, agent, n_games=1, greedy=True, logger=logger)]
        video_env.close()
        logger.info(f"Final reward (one episode): {np.mean(final_rewards):.2f}")
    except Exception as e:
        logger.error(f"Video recording failed: {e}")

# 全局变量，用于防止重复调用恢复训练
main_has_continue_called = False

# 处理断点续训的入口函数
def main_with_recovery():
    """主入口函数，支持从检查点恢复训练"""
    # 初始化标记变量
    global main_has_continue_called
    main_has_continue_called = False
    
    # 如果设置了全局变量，优先使用全局变量
    if CONTINUE_TRAINING and os.path.exists(CHECKPOINT):
        try:
            print(f"Using global settings: CHECKPOINT={CHECKPOINT}, TARGET_EPISODES={TARGET_EPISODES}")
            continue_training_from_checkpoint(
                checkpoint_path=CHECKPOINT,
                num_iterations=TARGET_EPISODES,
                DQNAgent=DQNAgent,
                initialize_environment=initialize_environment,
                make_evaluation_env=make_evaluation_env,
                make_recording_env=make_recording_env,
                play_and_record=play_and_record,
                train_on_batch=train_on_batch,
                evaluate=evaluate,
                device=device,
                MemoryBufferClass=MemoryEfficientReplayBuffer,
                setup_logger=setup_logger,
                setup_memory_leak_detection=setup_memory_leak_detection,
                check_and_manage_memory=check_and_manage_memory,
                force_gc=force_gc,
                print_memory_usage=print_memory_usage,
                analyze_gpu_tensors=analyze_gpu_tensors,
                save_checkpoint_fn=save_checkpoint,
                save_model_fn=save_model,
                save_training_plots_fn=save_training_plots,
                save_training_history_fn=save_training_history,
                epsilon_final=EPSILON_FINAL_SETTING,
                total_iteration_times=TOTAL_ITERATION_TIMES,
                main_function=main,
                main_has_continue_called=main_has_continue_called
            )
            return
        except Exception as e:
            print(f"Failed to resume from global checkpoint: {e}")
            import traceback
            traceback.print_exc()
            # 如果全局参数恢复失败，则设置标记，确保不会再次尝试恢复
            main_has_continue_called = True
    
    # 否则解析命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description='DQN Breakout Training')
    parser.add_argument('--resume', type=str, help='Path to checkpoint file to resume from')
    parser.add_argument('--iterations', type=int, help='Number of iterations to train for', default=TOTAL_ITERATION_TIMES)
    parser.add_argument('--epsilon', type=float, help='Final epsilon value', default=EPSILON_FINAL_SETTING)
    args = parser.parse_args()
    
    if args.resume and os.path.exists(args.resume):
        try:
            continue_training_from_checkpoint(
                checkpoint_path=args.resume,
                num_iterations=args.iterations,
                DQNAgent=DQNAgent,
                initialize_environment=initialize_environment,
                make_evaluation_env=make_evaluation_env,
                make_recording_env=make_recording_env,
                play_and_record=play_and_record,
                train_on_batch=train_on_batch,
                evaluate=evaluate,
                device=device,
                MemoryBufferClass=MemoryEfficientReplayBuffer,
                setup_logger=setup_logger,
                setup_memory_leak_detection=setup_memory_leak_detection,
                check_and_manage_memory=check_and_manage_memory,
                force_gc=force_gc,
                print_memory_usage=print_memory_usage,
                analyze_gpu_tensors=analyze_gpu_tensors,
                save_checkpoint_fn=save_checkpoint,
                save_model_fn=save_model,
                save_training_plots_fn=save_training_plots,
                save_training_history_fn=save_training_history,
                epsilon_final=args.epsilon,
                total_iteration_times=TOTAL_ITERATION_TIMES,
                main_function=main,
                main_has_continue_called=main_has_continue_called
            )
        except Exception as e:
            print(f"Failed to resume from checkpoint: {e}")
            print("Starting fresh training...")
            # 设置标记，确保不会再次尝试恢复
            main_has_continue_called = True
            main()
    else:
        main()

if __name__ == "__main__":
    try:
        main_with_recovery()
    except Exception as e:
        print(f"Fatal error: {e}")
        # 打印完整的异常堆栈
        import traceback
        traceback.print_exc()