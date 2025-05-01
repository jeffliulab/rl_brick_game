"""
保存和绘图实用工具模块，用于DQN训练过程中的模型保存和可视化。
该脚本是3_dqn_5M.py使用的封装函数。
"""

import os
# 在导入matplotlib前设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
import torch
from pandas import DataFrame
import logging

# 移动平均计算函数
def moving_average(x, span, **kw):
    """计算序列的指数加权移动平均值"""
    return (
        DataFrame({"x": np.asarray(x)})
        .x.ewm(span=span, **kw)
        .mean()
        .values
    )

# 保存检查点
def save_checkpoint(iteration, agent, optimizer, scheduler, 
                  mean_rw_history, loss_history, train_reward_history, 
                  epsilon_history, memory_usage_history, 
                  checkpoint_dir="output/checkpoints", logger=None):
    """保存训练检查点，包含完整训练状态"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/dqn_checkpoint_{iteration}.pt"
    
    try:
        torch.save({
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
        }, checkpoint_path)
        
        if logger:
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        else:
            print(f"Checkpoint saved to {checkpoint_path}")
            
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to save checkpoint: {e}")
        else:
            print(f"Failed to save checkpoint: {e}")
        return False

# 保存模型
def save_model(iteration, agent, logger=None):
    """保存模型权重"""
    os.makedirs("output/models", exist_ok=True)
    model_path = f"output/models/dqn_model_atari_{iteration}.pt"
    
    try:
        torch.save(agent.policy_net.state_dict(), model_path)
        
        if logger:
            logger.info(f"Model saved to {model_path}")
        else:
            print(f"Model saved to {model_path}")
            
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to save model: {e}")
        else:
            print(f"Failed to save model: {e}")
        return False

# 保存训练图表
def save_training_plots(iteration, mean_rw_history, train_reward_history, loss_history, 
                      epsilon_history, memory_usage_history, memory_check_freq, 
                      logger=None):
    """生成并保存完整的训练图表"""
    try:
        os.makedirs("output/plots", exist_ok=True)
        plt.figure(figsize=(16, 12))
        
        # (1) 评估平均回报
        plt.subplot(2, 3, 1)
        plt.title("Evaluation reward (average of 3 games)")
        plt.plot(mean_rw_history)
        plt.grid()
        
        # (2) 训练期间的奖励 (moving average)
        plt.subplot(2, 3, 2)
        plt.title("Training reward (moving average)")
        if len(train_reward_history) >= 100:
            plt.plot(moving_average(train_reward_history, span=100, min_periods=1))
        else:
            plt.plot(train_reward_history)  # 数据少时直接绘制
        plt.grid()
        
        # (3) 训练损失曲线 (moving average)
        plt.subplot(2, 3, 3)
        plt.title("Training loss (smooth L1) - moving average")
        if len(loss_history) >= 100:
            plt.plot(moving_average(loss_history, span=100, min_periods=1))
        else:
            plt.plot(loss_history)  # 数据少时直接绘制
        plt.grid()
        
        # (4) Epsilon 随迭代的变化
        plt.subplot(2, 3, 4)
        plt.title("Epsilon over iterations")
        plt.plot(epsilon_history)
        plt.grid()
        
        # (5) 内存使用情况随时间变化
        plt.subplot(2, 3, 5)
        plt.title("Memory usage (MB)")
        if memory_usage_history:
            x_vals = range(0, len(memory_usage_history) * memory_check_freq, memory_check_freq)
            plt.plot(x_vals[:len(memory_usage_history)], memory_usage_history)
        plt.grid()
        
        plt.tight_layout()
        plt.savefig(f"output/plots/dqn_curves_iter_{iteration}.png", dpi=200)
        plt.close()  # 关闭图表释放内存
        
        if logger:
            logger.info(f"Training curves at iteration {iteration} saved.")
        else:
            print(f"Training curves at iteration {iteration} saved.")
            
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to generate plots: {e}")
        else:
            print(f"Failed to generate plots: {e}")
        return False

# 保存训练历史数据
def save_training_history(iteration, mean_rw_history, train_reward_history, loss_history, 
                         epsilon_history, memory_usage_history, logger=None):
    """保存训练历史数据到.npz文件"""
    try:
        os.makedirs("output/history", exist_ok=True)
        history_data = {
            'mean_rewards': mean_rw_history,
            'train_rewards': train_reward_history,
            'losses': loss_history,
            'epsilons': epsilon_history,
            'memory_usage': memory_usage_history
        }
        
        history_path = f'output/history/training_history_iter_{iteration}.npz'
        np.savez(history_path, **history_data)
        
        if logger:
            logger.info(f"Training history at iteration {iteration} saved to {history_path}")
        else:
            print(f"Training history at iteration {iteration} saved to {history_path}")
            
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to save training history: {e}")
        else:
            print(f"Failed to save training history: {e}")
        return False