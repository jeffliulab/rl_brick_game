#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DQN训练的内存管理模块

包含内存监控、垃圾回收、内存泄漏检测等功能
"""

import os
import gc
import time
import psutil
import tracemalloc
import logging
import torch
import numpy as np
from collections import deque
import random

# ===== 内存监控和限制函数 =====

def limit_memory_usage(max_memory_gb=28):
    """限制程序最大内存使用"""
    try:
        import resource
        # 将软限制设为最大内存的90%
        soft_limit = int(max_memory_gb * 0.9 * 1024 * 1024 * 1024)
        # 将硬限制设为最大内存
        hard_limit = int(max_memory_gb * 1024 * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))
        print(f"Memory usage limited to {max_memory_gb}GB")
    except ImportError:
        print("resource module not available, cannot limit memory usage")
    except Exception as e:
        print(f"Failed to set memory limit: {e}")

def get_memory_usage():
    """获取当前进程的内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024
    return memory_usage_mb

def print_memory_usage(prefix="", logger=None):
    """打印当前内存使用情况，可选添加前缀和记录到日志"""
    memory_usage = get_memory_usage()
    message = f"{prefix} Memory usage: {memory_usage:.2f} MB"
    if logger:
        logger.info(message)
    else:
        print(message)
    return memory_usage

def force_gc(logger=None):
    """强制进行垃圾回收，包括CUDA内存"""
    if logger:
        logger.debug("执行垃圾回收...")
    
    # 强制进行完整垃圾回收
    gc.collect(generation=2)  
    
    # 清理CUDA内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # 确保所有CUDA操作完成
        
    # 记录剩余张量（如果启用了DEBUG级别日志）
    if logger and logger.isEnabledFor(logging.DEBUG):
        try:
            remaining = gc.get_objects()
            tensors = [obj for obj in remaining if isinstance(obj, torch.Tensor)]
            logger.debug(f"剩余张量数量: {len(tensors)}")
        except:
            pass

def analyze_gpu_tensors(logger, detailed=False):
    """分析GPU内存中的张量使用情况"""
    if not torch.cuda.is_available():
        return
    
    # 将中文改为英文，避免编码问题
    logger.info("====== GPU Tensor Analysis ======")
    logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    
    if detailed:
        try:
            # 尝试使用较新版本的PyTorch内存工具
            torch.cuda.memory_summary(abbreviated=True)
        except:
            pass

# ===== 内存泄漏检测 =====

def setup_memory_leak_detection(interval=50000):
    """初始化内存泄漏检测，每隔interval次迭代检查一次"""
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    def check_for_leaks(iteration, logger=None):
        if iteration % interval == 0 and iteration > 0:
            snapshot2 = tracemalloc.take_snapshot()
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            message = "[ Top 5 memory usage differences ]"
            if logger:
                logger.info(message)
            else:
                print(message)
            
            for stat in top_stats[:5]:
                if logger:
                    logger.info(str(stat))
                else:
                    print(stat)
            return top_stats
        return None
    
    return check_for_leaks

# ===== 回放缓冲区内存管理 =====

class MemoryEfficientReplayBuffer:
    """
    A memory-efficient replay buffer storing (s, a, r, s_next, done).
    Uses uint8 for storing frames to save memory.
    """
    def __init__(self, capacity, logger=None, device="cuda"):
        self.buffer = deque(maxlen=capacity)
        self.memory_warning_threshold = 0.8  # 内存使用率阈值降低到80%
        self.gc_counter = 0  # 添加计数器来减少垃圾回收频率
        self.total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
        self.logger = logger
        self.device = device
        
        if logger:
            logger.info(f"Total system memory: {self.total_memory:.2f} GB")
        else:
            print(f"Total system memory: {self.total_memory:.2f} GB")
            
        # 记录最后一次紧急内存处理的时间
        self.last_emergency_cleanup = 0

    def __len__(self):
        return len(self.buffer)

    def _check_memory(self):
        """检查内存使用情况，使用多级内存管理策略"""
        # 增加计数器，只有每5次调用才真正检查内存 (从10改为5，更频繁检查)
        self.gc_counter += 1
        if self.gc_counter % 5 != 0:
            return False
            
        memory = psutil.virtual_memory()
        memory_used_ratio = memory.percent / 100.0
        
        # 多级内存管理策略
        if memory_used_ratio > 0.95:  # 内存使用率超过95%，紧急情况
            # 确保紧急清理操作不会太频繁（至少间隔1000次添加操作）
            current_time = len(self.buffer)
            if current_time - self.last_emergency_cleanup > 1000:
                self.last_emergency_cleanup = current_time
                message = f"EMERGENCY: Critical memory usage ({memory_used_ratio*100:.1f}%), removing old experiences"
                if self.logger:
                    self.logger.warning(message)
                else:
                    print(message)
                
                # 计算要移除的经验数量（移除25%的缓冲区）
                if len(self.buffer) > 10000:  # 确保有足够的样本
                    remove_count = len(self.buffer) // 4
                    for _ in range(remove_count):
                        self.buffer.popleft()  # 移除最旧的经验
                    
                    message = f"Removed {remove_count} old experiences, buffer size now: {len(self.buffer)}"
                    if self.logger:
                        self.logger.warning(message)
                    else:
                        print(message)
                
                # 强制进行多次垃圾回收
                message = "Performing aggressive garbage collection..."
                if self.logger:
                    self.logger.info(message)
                else:
                    print(message)
                    
                for _ in range(3):  # 连续进行3次垃圾回收
                    force_gc(self.logger)
                    
                return True
                
        elif memory_used_ratio > 0.85:  # 内存使用率超过85%，常规垃圾回收
            message = f"Warning: High memory usage ({memory_used_ratio*100:.1f}%), triggering garbage collection"
            if self.logger:
                self.logger.info(message)
            else:
                print(message)
                
            force_gc(self.logger)  # 正常垃圾回收
            return True
            
        return False

    def add(self, s, a, r, s_next, done):
        """
        添加经验到缓冲区，将浮点状态转为uint8节省内存
        """
        # 检查并转换状态为uint8以节省内存
        if isinstance(s, np.ndarray) and s.dtype == np.float32:
            s = (s * 255).astype(np.uint8)
        if isinstance(s_next, np.ndarray) and s_next.dtype == np.float32:
            s_next = (s_next * 255).astype(np.uint8)
            
        self.buffer.append((s, a, r, s_next, done))
        
        # 减少内存检查频率，只在添加较多样本后才检查
        if len(self.buffer) % 1000 == 0:  # 从10000改为1000
            self._check_memory()

    def sample(self, batch_size, max_attempts=3):
        """
        带有内存安全机制的采样函数，会在内存不足时自动减小批量大小
        显式分离张量并移除中间变量以防止内存泄漏
        """
        if len(self.buffer) < batch_size:
            return None
            
        current_batch_size = batch_size
        attempts = 0
        
        while attempts < max_attempts:
            try:
                batch = random.sample(self.buffer, current_batch_size)
                
                # 批量处理转换，分步进行以减少峰值内存
                # 1. 先处理状态
                s_batch = []
                for s, _, _, _, _ in batch:
                    s_batch.append(s)
                s = np.array(s_batch, dtype=np.uint8).astype(np.float32) / 255.0
                s_t = torch.from_numpy(s).to(self.device)
                del s_batch, s  # 显式删除中间变量
                
                # 2. 处理动作
                a = np.array([item[1] for item in batch], dtype=np.int64)
                a_t = torch.from_numpy(a).to(self.device)
                del a  # 显式删除
                
                # 3. 处理奖励
                r = np.array([item[2] for item in batch], dtype=np.float32)
                r_t = torch.from_numpy(r).to(self.device)
                del r  # 显式删除
                
                # 4. 处理下一状态
                s_next_batch = []
                for _, _, _, s_next, _ in batch:
                    s_next_batch.append(s_next)
                s_next = np.array(s_next_batch, dtype=np.uint8).astype(np.float32) / 255.0
                s_next_t = torch.from_numpy(s_next).to(self.device)
                del s_next_batch, s_next  # 显式删除
                
                # 5. 处理结束标志
                done = np.array([item[4] for item in batch], dtype=np.float32)
                done_t = torch.from_numpy(done).to(self.device)
                del done  # 显式删除
                del batch  # 显式删除原始批次
                
                # 返回分离的张量以防止内存泄漏
                return s_t.detach(), a_t.detach(), r_t.detach(), s_next_t.detach(), done_t.detach()
                
            except (MemoryError, RuntimeError, np.core._exceptions._ArrayMemoryError) as e:
                # 内存不足，减小批量大小并重试
                attempts += 1
                force_gc(self.logger)  # 强制垃圾回收
                current_batch_size = current_batch_size // 2
                
                message = f"Memory error: {e}, reducing batch size to {current_batch_size}"
                if self.logger:
                    self.logger.warning(message)
                else:
                    print(message)
                
        # 如果多次尝试后仍然失败，返回一个非常小的批次
        message = "Multiple sampling attempts failed, using minimal batch size"
        if self.logger:
            self.logger.warning(message)
        else:
            print(message)
            
        return self.sample(8, max_attempts=1)  # 尝试用极小的批量大小

# ===== 日志设置 =====

def setup_logger(log_file='output/dqn_training.log'):
    """设置日志记录器"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('dqn_training')
    logger.setLevel(logging.INFO)
    
    # 清理现有的处理器，避免重复添加
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ===== 动态内存管理函数 =====

def check_and_manage_memory(iteration, logger, replay_buffer, batch_size, 
                           memory_warning_threshold=80.0, memory_critical_threshold=85.0):
    """
    每隔一定迭代次数检查内存使用情况，并在必要时进行主动内存管理
    
    参数:
        iteration: 当前迭代次数
        logger: 日志记录器
        replay_buffer: 回放缓冲区
        batch_size: 当前批量大小
        memory_warning_threshold: 内存警告阈值（百分比）
        memory_critical_threshold: 内存危急阈值（百分比）
        
    返回:
        调整后的批量大小
    """
    if iteration % 500 != 0:  # 每500次迭代检查一次
        return batch_size
        
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    if memory_percent > memory_critical_threshold:
        # 紧急内存管理
        logger.warning(f"CRITICAL MEMORY: {memory_percent:.1f}%, reducing buffer size!")
        
        # 立即减少回放缓冲区大小
        if len(replay_buffer.buffer) > 20000:
            to_remove = len(replay_buffer.buffer) // 3  # 删除三分之一
            for _ in range(to_remove):
                replay_buffer.buffer.popleft()
            
            # 强制垃圾回收
            force_gc(logger)
            
            # 等待系统释放内存
            time.sleep(0.5)
        
        # 临时减小批量大小
        adjusted_batch_size = max(4, batch_size // 2)
        logger.info(f"Temporarily reducing batch size to {adjusted_batch_size}")
        return adjusted_batch_size
        
    elif memory_percent > memory_warning_threshold:
        logger.info(f"High memory usage: {memory_percent:.1f}%, running GC")
        force_gc(logger)
    
    return batch_size  # 如果内存使用正常，返回原批量大小

# ===== 检查点加载优化函数 =====

def optimize_checkpoint_for_loading(checkpoint, device='cpu'):
    """优化检查点数据，减少内存使用"""
    # 创建新的优化后的模型权重字典
    policy_dict = {}
    for k, v in checkpoint['policy_net_state_dict'].items():
        if isinstance(v, torch.Tensor):
            policy_dict[k] = v.detach().clone().to(device)
        else:
            policy_dict[k] = v
            
    target_dict = {}
    for k, v in checkpoint['target_net_state_dict'].items():
        if isinstance(v, torch.Tensor):
            target_dict[k] = v.detach().clone().to(device)
        else:
            target_dict[k] = v
    
    # 替换原检查点中的权重字典
    checkpoint['policy_net_state_dict'] = policy_dict
    checkpoint['target_net_state_dict'] = target_dict
    
    # 清理其他大型数据以节省内存
    keys_to_keep = ['policy_net_state_dict', 'target_net_state_dict', 
                   'optimizer_state_dict', 'scheduler_state_dict', 
                   'iteration', 'epsilon']
    
    for key in list(checkpoint.keys()):
        if key not in keys_to_keep:
            del checkpoint[key]
    
    return checkpoint