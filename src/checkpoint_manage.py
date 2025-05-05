#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DQN训练的检查点管理模块

包含检查点的保存、加载和恢复训练状态等功能
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from tqdm import trange

from memory_manage import (
    force_gc, print_memory_usage, optimize_checkpoint_for_loading
)

def save_checkpoint(iteration, agent, optimizer, scheduler, 
                  mean_rw_history, loss_history, train_reward_history, 
                  epsilon_history, memory_usage_history, 
                  checkpoint_dir, logger=None):
    """
    保存训练检查点，包含所有需要的状态信息，用于断点恢复。
    
    Args:
        iteration: 当前迭代次数
        agent: DQN智能体
        optimizer: 优化器
        scheduler: 学习率调度器
        mean_rw_history: 平均奖励历史记录
        loss_history: 损失值历史记录
        train_reward_history: 训练奖励历史记录
        epsilon_history: epsilon值历史记录
        memory_usage_history: 内存使用历史记录
        checkpoint_dir: 检查点保存目录
        logger: 日志记录器
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"dqn_checkpoint_{iteration}.pt")
    
    # 创建检查点字典
    checkpoint = {
        'iteration': iteration,
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epsilon': agent.epsilon,
        'mean_rw_history': mean_rw_history,
        'loss_history': loss_history,
        'train_reward_history': train_reward_history,
        'epsilon_history': epsilon_history,
        'memory_usage_history': memory_usage_history,
    }
    
    # 保存检查点
    torch.save(checkpoint, checkpoint_path)
    
    if logger:
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    else:
        print(f"Checkpoint saved to {checkpoint_path}")

def save_model(iteration, agent, logger=None):
    """
    只保存模型参数，不包含其他训练状态。
    
    Args:
        iteration: 当前迭代次数或标识符
        agent: DQN智能体
        logger: 日志记录器
    """
    os.makedirs("output/models", exist_ok=True)
    
    # 保存策略网络
    policy_net_path = f"output/models/dqn_policy_net_{iteration}.pt"
    torch.save(agent.policy_net.state_dict(), policy_net_path)
    
    # 保存目标网络
    target_net_path = f"output/models/dqn_target_net_{iteration}.pt"
    torch.save(agent.target_net.state_dict(), target_net_path)
    
    if logger:
        logger.info(f"Model saved to {policy_net_path} and {target_net_path}")
    else:
        print(f"Model saved to {policy_net_path} and {target_net_path}")

def load_checkpoint(checkpoint_path, map_location='cpu'):
    """
    加载检查点文件并返回其内容。
    
    Args:
        checkpoint_path: 检查点文件路径
        map_location: 模型权重加载位置
        
    Returns:
        优化后的检查点内容，如果加载失败则返回None
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} does not exist!")
        return None
    
    try:
        # 1. 尝试使用新版PyTorch加载方式
        try:
            # 使用最新PyTorch功能：先添加安全全局变量
            import torch.serialization
            torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
            # 然后加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
            print(f"Checkpoint loaded successfully with weights_only=True from {checkpoint_path}")
        except Exception as e1:
            # 方法1失败，尝试方法2：使用weights_only=False
            print(f"First loading method failed: {e1}")
            print("Trying alternate loading method (weights_only=False)...")
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            print(f"Checkpoint loaded successfully with weights_only=False from {checkpoint_path}")
            
        print(f"Checkpoint contains iteration: {checkpoint.get('iteration', 'Not found')}")
        
        # 优化检查点数据以减少内存使用
        checkpoint = optimize_checkpoint_for_loading(checkpoint)
        return checkpoint
        
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None
    
def resume_from_checkpoint(checkpoint_path, DQNAgent, initialize_environment, 
                          play_and_record, device, ReplayBufferClass, 
                          logger=None, epsilon_final=None):
    """
    从检查点文件恢复训练状态，确保正确加载和初始化。
    
    Args:
        checkpoint_path: 检查点文件路径
        DQNAgent: DQNAgent类
        initialize_environment: 初始化环境的函数
        play_and_record: 收集经验的函数
        device: 训练设备 (CPU/GPU)
        ReplayBufferClass: 回放缓冲区类
        logger: 日志记录器
        epsilon_final: 最终epsilon值
        
    Returns:
        恢复的训练状态元组，如果恢复失败则返回None
    """
    if logger:
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        print(f"Resuming from checkpoint: {checkpoint_path}")
    
    # 1) 加载检查点
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        return None
    
    # 2) 初始化环境
    env = initialize_environment(game_name="ALE/Breakout-v5", render_mode="rgb_array", frameskip=4)

    # 确保环境已经初始化
    obs, info = env.reset() 
    # 额外的重置步骤，确保环境状态干净
    for _ in range(3):
        obs, _, _, _ = env.step(0)  # 执行几次无操作动作确保干净状态
    
    # 3) 获取环境信息
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    
    # 4) 初始化Agent - 使用任意epsilon初始化，之后会重置
    agent = DQNAgent(state_shape, n_actions, epsilon=1.0)
    
    # 加载处理过的权重
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    
    # 清理不需要的检查点状态字典
    del checkpoint['policy_net_state_dict']
    del checkpoint['target_net_state_dict']
    
    # 5) 初始化优化器
    optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    del checkpoint['optimizer_state_dict']
    
    # 对于旧格式的检查点兼容性
    if 'scheduler_state_dict' in checkpoint:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.5)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        del checkpoint['scheduler_state_dict']
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.5)
    
    # 6) 重新创建回放缓冲区 (减小容量降低内存压力)
    replay_buffer = ReplayBufferClass(capacity=100000, logger=logger, device=device)
    
    # 7) 重新填充回放缓冲区
    if logger:
        logger.info("Refilling replay buffer...")
    else:
        print("Refilling replay buffer...")
        
    pbar = tqdm(total=50000, desc="Buffer Refilling")
    steps_done = 0
    while steps_done < 50000:
        steps_to_do = min(1000, 50000 - steps_done)
        reward = play_and_record(agent, env, replay_buffer, n_steps=steps_to_do, logger=logger)
        steps_done += steps_to_do
        pbar.update(steps_to_do)
        
        # 每10000步检查一次内存
        if steps_done % 10000 == 0:
            force_gc(logger)  # 强制垃圾回收
            
    pbar.close()
    
    # 8) 恢复保存的历史数据
    mean_rw_history = checkpoint.get('mean_rw_history', [])
    loss_history = checkpoint.get('loss_history', [])
    train_reward_history = checkpoint.get('train_reward_history', [])
    epsilon_history = checkpoint.get('epsilon_history', [])
    memory_usage_history = checkpoint.get('memory_usage_history', [])
    
    # 保存迭代次数并清理剩余的检查点内容
    iteration = checkpoint.get('iteration', 0)
    del checkpoint
    
    # 强制垃圾回收
    force_gc(logger)
    
    # 9) 返回恢复的状态
    return (env, agent, optimizer, scheduler, replay_buffer, 
            iteration, mean_rw_history, loss_history, 
            train_reward_history, epsilon_history, memory_usage_history)


def continue_training_from_checkpoint(checkpoint_path, num_iterations, 
                                     DQNAgent, initialize_environment, 
                                     make_evaluation_env, make_recording_env, 
                                     play_and_record, train_on_batch,
                                     evaluate, device, MemoryBufferClass,
                                     setup_logger, setup_memory_leak_detection,
                                     check_and_manage_memory, force_gc,
                                     print_memory_usage, analyze_gpu_tensors,
                                     save_checkpoint_fn=None, save_model_fn=None,
                                     save_training_plots_fn=None, save_training_history_fn=None,
                                     epsilon_start=None, epsilon_final=None,
                                     decay_start_iter=None, decay_end_iter=None,
                                     total_iteration_times=None,
                                     main_function=None, main_has_continue_called=False):
    """
    实际执行从检查点恢复训练的主函数，包含完整的训练循环
    
    Args:
        checkpoint_path: 检查点文件路径
        num_iterations: 目标迭代次数
        DQNAgent: DQNAgent类
        initialize_environment: 初始化环境的函数
        make_evaluation_env: 创建评估环境的函数
        make_recording_env: 创建录制视频环境的函数
        play_and_record: 收集经验的函数
        train_on_batch: 训练批次的函数
        evaluate: 评估函数
        device: 训练设备
        MemoryBufferClass: 回放缓冲区类
        setup_logger, setup_memory_leak_detection, check_and_manage_memory, 
        force_gc, print_memory_usage, analyze_gpu_tensors: 内存管理相关函数
        save_checkpoint_fn, save_model_fn, save_training_plots_fn, 
        save_training_history_fn: 保存相关函数
        epsilon_start: 起始epsilon值，来自dqn_train.py的EPSILON_START_SETTING
        epsilon_final: 最终epsilon值，来自dqn_train.py的EPSILON_FINAL_SETTING
        decay_start_iter: 开始衰减的迭代次数，来自dqn_train.py的DECAY_START_ITER
        decay_end_iter: 结束衰减的迭代次数，来自dqn_train.py的DECAY_END_ITER
        total_iteration_times: 总迭代次数，来自dqn_train.py的TOTAL_ITERATION_TIMES
        main_function: 主训练函数，用于恢复失败时调用
        main_has_continue_called: 避免递归调用的标记变量
        
    Returns:
        None
    """
    # 设置递归调用标记
    global_var_name = 'main_has_continue_called'
    if global_var_name in globals():
        globals()[global_var_name] = True

    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} does not exist!")
        print("Starting fresh training...")
        if main_function is not None:
            main_function()
        return
        
    print(f"Checkpoint file found: {checkpoint_path}")
    print(f"Target iterations: {num_iterations}")
    
    # 设置日志
    logger = setup_logger(log_file='output/dqn_resumed_training.log')
    logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
    logger.info(f"Training will continue until iteration {num_iterations}")
    
    # 启动内存泄漏检测
    leak_detector = setup_memory_leak_detection(interval=50000)
    
    # 从检查点恢复状态
    try:
        checkpoint_result = resume_from_checkpoint(
            checkpoint_path=checkpoint_path,
            DQNAgent=DQNAgent,
            initialize_environment=initialize_environment,
            play_and_record=play_and_record,
            device=device,
            ReplayBufferClass=MemoryBufferClass,
            logger=logger,
            epsilon_final=epsilon_final
        )
        
        if checkpoint_result is None:
            logger.error("Failed to resume from checkpoint")
            logger.info("Starting fresh training...")
            if main_function is not None:
                main_function()
            return
            
        (env, agent, optimizer, scheduler, replay_buffer, 
         start_iteration, mean_rw_history, loss_history, 
         train_reward_history, epsilon_history, memory_usage_history) = checkpoint_result
        
        logger.info(f"Successfully restored training state from iteration {start_iteration}")
        
        # 打印当前内存使用情况和GPU使用情况
        memory_usage = print_memory_usage("After checkpoint load", logger)
        analyze_gpu_tensors(logger)
        
        # 强制设置新的epsilon值
        if epsilon_start is not None:
            agent.epsilon = epsilon_start
            logger.info(f"Epsilon value explicitly overridden to {agent.epsilon}")
        
    except Exception as e:
        logger.error(f"Failed to resume from checkpoint: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Starting fresh training...")
        if main_function is not None:
            main_function()
        return
    
    # 设置迭代次数
    if num_iterations is None and total_iteration_times is not None:
        num_iterations = total_iteration_times
    elif num_iterations is None:
        # 默认值，仅在没有提供任何参数时使用
        num_iterations = 2000000
    
    # 确认超参数均已设置
    if epsilon_start is None:
        logger.warning("epsilon_start not provided, using a default value of 0.05")
        epsilon_start = 0.05
        
    if epsilon_final is None:
        logger.warning("epsilon_final not provided, using a default value of 0.005")
        epsilon_final = 0.005
        
    if decay_start_iter is None:
        logger.warning("decay_start_iter not provided, using a default value of 2000000")
        decay_start_iter = 2000000
        
    if decay_end_iter is None:
        logger.warning("decay_end_iter not provided, using a default value of 5000000")
        decay_end_iter = 5000000
    
    # 超参数设定
    gamma = 0.99
    batch_size = 16  # 从32减小到16，减轻内存负担
    target_update_freq = 10000
    
    eval_freq = 50000        # 每多少迭代做一次评估
    memory_check_freq = 5000  # 每多少迭代检查一次内存
    
    # 添加自动保存点，用于训练中断后恢复
    checkpoint_dir = "output/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 修改保存频率设置
    regular_save_freq = 50000  # 每50000次迭代保存一次
    
    # 内存管理阈值
    MEMORY_WARNING_THRESHOLD = 80.0  # 80%内存使用就开始警告
    MEMORY_CRITICAL_THRESHOLD = 85.0  # 85%就采取行动
    
    logger.info(f"Continuing DQN training from iteration {start_iteration}...")
    logger.info(f"Using epsilon decay: start={epsilon_start}, final={epsilon_final}, " 
               f"decay_start={decay_start_iter}, decay_end={decay_end_iter}")
    
    for i in trange(start_iteration, num_iterations, desc="Resumed training"):
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
        
        # 根据迭代次数决定epsilon值
        if i < decay_start_iter:
            agent.epsilon = epsilon_start
        elif i < decay_end_iter:
            # 从decay_start_iter到decay_end_iter之间线性衰减
            decay_progress = (i - decay_start_iter) / (decay_end_iter - decay_start_iter)
            agent.epsilon = epsilon_start - (epsilon_start - epsilon_final) * decay_progress
        else:
            agent.epsilon = epsilon_final
            
        # 记录当前epsilon
        epsilon_history.append(agent.epsilon)
        
        # 隔一定迭代频率更新target_net
        if i % target_update_freq == 0:
            agent.update_target_network()
            logger.info(f"Iteration {i}: Target network updated")
            
        # 每1000步调整一次学习率
        if i % 1000 == 0:
            scheduler.step()
            
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
            
        # 检查是否应该保存
        if i > start_iteration and i % regular_save_freq == 0:
            # 强制垃圾回收以确保有足够内存
            force_gc(logger)
            
            # 调用绘图和保存函数
            try:
                if save_checkpoint_fn:
                    save_checkpoint_fn(i, agent, optimizer, scheduler, 
                                     mean_rw_history, loss_history, train_reward_history, 
                                     epsilon_history, memory_usage_history, 
                                     checkpoint_dir, logger)
                              
                if save_model_fn:              
                    save_model_fn(i, agent, logger)
                
                if save_training_plots_fn:
                    save_training_plots_fn(i, mean_rw_history, train_reward_history, loss_history, 
                                         epsilon_history, memory_usage_history, memory_check_freq, 
                                         logger)
                                  
                if save_training_history_fn:
                    save_training_history_fn(i, mean_rw_history, train_reward_history, loss_history, 
                                           epsilon_history, memory_usage_history, logger)
            except Exception as e:
                logger.error(f"Failed to save at iteration {i}: {e}")
                
    # 保存最终结果
    force_gc(logger)
    try:
        # 保存最终模型
        if save_model_fn:
            save_model_fn("final_resumed", agent, logger)
        
        # 保存最终训练曲线
        if save_training_plots_fn:
            save_training_plots_fn("final_resumed", mean_rw_history, train_reward_history, loss_history, 
                                 epsilon_history, memory_usage_history, memory_check_freq, logger)
                           
        # 保存最终训练历史数据
        if save_training_history_fn:
            save_training_history_fn("final_resumed", mean_rw_history, train_reward_history, loss_history, 
                                   epsilon_history, memory_usage_history, logger)
    except Exception as e:
        logger.error(f"Failed to save final results: {e}")
        
    # 录制最终评估视频
    try:
        import datetime
        
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