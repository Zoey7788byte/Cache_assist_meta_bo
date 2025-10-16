import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
import numpy as np
import torch
import math
import copy
import os
import time
import csv
from datetime import datetime
from collections import deque
import threading
from parameters import args_parser
args = args_parser()

class MetaPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 合并为单一输出层
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # 初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.01)
        nn.init.zeros_(self.fc_out.bias)

        # 层归一化
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # 单一可学习的缩放参数
        self.logit_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
       
        # 单一输出
        output = self.fc_out(x) * torch.sigmoid(self.logit_scale)
        return output

class MetaLearner:
    def __init__(self, env, all_requests, request_records, end_time, num_epochs=100, 
                 task_length=100, save_dir='./saved_models', save_every=10, shared_flag=True):
        self.env = env
        self.all_requests = all_requests
        self.num_epochs = num_epochs
        self.task_length = task_length
        self.save_dir = save_dir
        self.save_every = save_every
        self.end_time = end_time
        self.num_tasks = int(self.end_time / self.task_length)
        self.meta_lr = args.meta_lr
        self.inner_lr = args.inner_lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # === 核心新增：shared_flag ===
        self.shared_flag = args.Shared_flag
        print(f"{'='*60}")
        print(f"Shared Mode: {'ENABLED' if shared_flag else 'DISABLED'}")
        if not shared_flag:
            print("Note: Models are treated as complete but size includes shared components")
        print(f"{'='*60}")
        
        # 基本维度
        self.N, self.M = self.env.NUM_EDGE_NODES, self.env.NUM_MODELS
        self.NUM_SHARED_MODELS = self.env.NUM_SHARED_MODELS
        self.NUM_SPECIFIC_MODELS = self.env.NUM_SPECIFIC_MODELS
        self.switch_lambda  = args.switch_lambda

        # 初始化网络和优化器
        input_dim = self.calculate_input_dim()
        output_dim = self.N * self.M * 2  # deployment logits + update logits concatenated
        self.meta_policy = MetaPolicyNetwork(input_dim, output_dim, args.HIDDEN_DIM).to(self.device)
        self.meta_optimizer = torch.optim.Adam(self.meta_policy.parameters(), lr=self.meta_lr)
        
        # 预计算常用张量
        self._precompute_static_tensors()
    
        # 批处理配置
        self.batch_size = min(10, self.task_length // 4)  # 批量处理时间步
        self.gradient_accumulation_steps = 4  # 梯度累积步数
        
        # 状态缓存
        self.state_cache = StateCache(max_size=200)
        self.tensor_buffers = TensorBufferPool(self.device, self.N, self.M)
        
        # 数据预加载
        self._setup_optimized_data_loader()
        
        # 损失记录
        self.loss_history = {
            'epoch': [], 'meta_loss': [], 'accuracy_cost': [], 'update_cost': [],
            'switching_cost': [], 'communication_cost': [], 'inference_cost': [],
            'hit_rate': [], 'timestamp': []
        }
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        shared_mode_str = "Shared" if self.shared_flag else "NonShared"
        self.loss_log_path = os.path.join(save_dir, f'loss_history_{shared_mode_str}_Innerlr{args.inner_lr}_Melr{args.meta_lr}_HIDDEN_DIM{args.HIDDEN_DIM}_train_NumTask{args.train_task_num}.csv')

    def meta_policy_forward_with_weights(self, state, fast_weights):
        """使用fast_weights进行前向传播 - 增强错误处理"""
        try:
            return functional_call(self.meta_policy, fast_weights, (state,))
        except Exception as e:
            print(f"前向传播失败: {e}")
            # 返回默认输出
            output_dim = self.N * self.M * 2
            return torch.zeros(output_dim, device=self.device, requires_grad=True)

    def _setup_optimized_data_loader(self):
        """设置优化的数据加载器"""
        try:
            # 预加载所有数据到GPU
            all_requests_array = np.array(self.all_requests, dtype=np.float32)
            self.all_requests_tensor = torch.from_numpy(all_requests_array).to(self.device)
            
            if self.all_requests_tensor.dim() == 2:
                self.all_requests_tensor = self.all_requests_tensor.view(-1, self.N, self.M)
            
            print(f"预加载数据形状: {self.all_requests_tensor.shape}")
        except Exception as e:
            print(f"数据预加载失败: {e}")
            # 创建默认的零张量
            self.all_requests_tensor = torch.zeros(
                (self.end_time, self.N, self.M), device=self.device, dtype=torch.float32
            )
      
    def _precompute_static_tensors(self):
        """预计算所有静态张量"""
        try:
            device = self.device
            
            # 模型相关张量
            if self.shared_flag:
                # 考虑共享：使用原始模型大小
                self.model_sizes_tensor = torch.tensor(
                    self.env.cloud_manager.model_sizes, dtype=torch.float32, device=device
                )
            else:
                # 不考虑共享：特定模型大小 = 特定部分 + 所有依赖的共享部分
                adjusted_sizes = self.env.cloud_manager.model_sizes.copy()
                for specific_idx in range(self.NUM_SPECIFIC_MODELS):
                    global_specific_idx = self.NUM_SHARED_MODELS + specific_idx
                    # 获取依赖的共享模型
                    if hasattr(self.env, 'cloud_manager') and hasattr(self.env.cloud_manager, 'dependency_matrix'):
                        dependency_row = self.env.cloud_manager.dependency_matrix[specific_idx]
                        shared_size_sum = sum(
                            self.env.cloud_manager.model_sizes[shared_idx] 
                            for shared_idx in range(self.NUM_SHARED_MODELS) 
                            if dependency_row[shared_idx] > 0
                        )
                        adjusted_sizes[global_specific_idx] += shared_size_sum
                
                self.model_sizes_tensor = torch.tensor(
                    adjusted_sizes, dtype=torch.float32, device=device
                )
            
            self.accuracy_min_tensor = torch.tensor(
                self.env.accuracy_min, dtype=torch.float32, device=device
            )
            
            # 成本相关张量
            self.model_switch_costs = torch.tensor(
                [self.env.get_switch_cost(mid) for mid in range(self.M)],
                device=device, dtype=torch.float32
            ).view(1, -1)
            
            self.model_update_costs = torch.tensor(
                [self.env.get_update_cost(mid) for mid in range(self.M)],
                device=device, dtype=torch.float32
            ).view(1, -1)
            
            # 依赖关系矩阵
            if hasattr(self.env, 'cloud_manager') and hasattr(self.env.cloud_manager, 'dependency_matrix'):
                self.dependency_matrix_tensor = torch.tensor(
                    self.env.cloud_manager.dependency_matrix, dtype=torch.float32, device=device
                )
            else:
                self.dependency_matrix_tensor = torch.zeros(
                    (self.NUM_SPECIFIC_MODELS, self.NUM_SHARED_MODELS), device=device
                )
                
            # 预计算初始精度张量（所有模型）
            self.initial_accuracies_tensor = torch.tensor(
                self.env.initial_acc, dtype=torch.float32, device=device
            )
            
            # 预计算每个特定模型的依赖共享模型索引
            self.dependency_indices_cache = []
            for specific_idx in range(self.NUM_SPECIFIC_MODELS):
                if self.shared_flag:
                    # 考虑共享：保留依赖关系
                    dependencies = self.dependency_matrix_tensor[specific_idx]
                    indices = torch.nonzero(dependencies > 0).squeeze(1).tolist()
                else:
                    # 不考虑共享：无依赖关系
                    indices = []
                self.dependency_indices_cache.append(indices)

            # 其他常量
            self.edge_cache_storage = args.edge_cache_storage
            self.age_decay_factor = args.age_decay_factor
            self.delay_local = torch.tensor(args.DELAY_LOCAL, device=device)
            self.delay_cloud = torch.tensor(args.DELAY_CLOUD, device=device)
            self.model_inference_cost = torch.tensor(args.INFERENCE_COST_PER_MB, device=device)
            
            print("静态张量预计算完成")
        except Exception as e:
            print(f"静态张量预计算失败: {e}")
            # 创建默认张量以避免程序崩溃
            self._create_default_tensors()
    
    def _create_default_tensors(self):
        """创建默认张量（用于错误恢复）"""
        device = self.device
        
        self.model_sizes_tensor = torch.ones(self.M, dtype=torch.float32, device=device)
        self.accuracy_min_tensor = torch.full((self.M,), 0.5, dtype=torch.float32, device=device)
        self.model_switch_costs = torch.ones((1, self.M), dtype=torch.float32, device=device)
        self.model_update_costs = torch.ones((1, self.M), dtype=torch.float32, device=device)
        self.dependency_matrix_tensor = torch.zeros(
            (self.NUM_SPECIFIC_MODELS, self.NUM_SHARED_MODELS), device=device
        )
        self.initial_accuracies_tensor = torch.full((self.M,), 0.8, dtype=torch.float32, device=device)
        self.dependency_indices_cache = [[] for _ in range(self.NUM_SPECIFIC_MODELS)]
        
        # 默认常量
        self.edge_cache_storage = 1000
        self.age_decay_factor = args.age_decay_factor
        self.delay_local = torch.tensor(1.0, device=device)
        self.delay_cloud = torch.tensor(10.0, device=device)
        self.model_inference_cost = torch.tensor(0.1, device=device)

    def _get_state_cache_key(self, t, isolated_state):
        """生成状态缓存键 - 改进的哈希函数"""
        try:
            # 使用更稳定的哈希方法
            cache_bytes = isolated_state['cache_state'].tobytes()
            age_bytes = isolated_state['age_state'].tobytes()
            
            # 使用简单的哈希组合
            cache_hash = hash(cache_bytes) & 0x7FFFFFFF  # 确保非负
            age_hash = hash(age_bytes) & 0x7FFFFFFF
            
            return (t, cache_hash, age_hash)
        except Exception as e:
            print(f"状态缓存键生成失败: {e}")
            # 返回简单的时间戳键
            return (t, 0, 0)

    def _get_state_from_isolated(self, t, isolated_state):
        """从隔离状态获取状态向量 - 增强错误处理"""
        # 保存原始状态
        original_cache = getattr(self.env, 'cache_state', None)
        original_age = getattr(self.env, 'age_state', None)
        original_accuracies = getattr(self.env, 'accuracies', None)
        
        try:
            # 安全地设置环境状态
            if isolated_state['cache_state'] is not None:
                self.env.cache_state = isolated_state['cache_state'].flatten()
            if isolated_state['age_state'] is not None:
                self.env.age_state = isolated_state['age_state'].flatten()
            if isolated_state['accuracies'] is not None:
                self.env.accuracies = isolated_state['accuracies']
            
            # 获取状态向量
            state = self.env.get_state_vector(t).to(device=self.device, non_blocking=True)
            return state
            
        except Exception as e:
            print(f"状态向量获取失败 (t={t}): {e}")
            # 返回默认状态向量
            default_dim = self.calculate_input_dim()
            return torch.zeros(default_dim, device=self.device, dtype=torch.float32)
            
        finally:
            # 始终恢复原始状态
            try:
                if original_cache is not None:
                    self.env.cache_state = original_cache
                if original_age is not None:
                    self.env.age_state = original_age
                if original_accuracies is not None:
                    self.env.accuracies = original_accuracies
            except Exception as restore_e:
                print(f"状态恢复失败: {restore_e}")

    def calculate_input_dim(self):
        return (self.N * self.M) * 3 + self.M + self.N

    def lightweight_snapshot(self):
        snapshot = {}
        try:
            # 确保从全局状态获取正确的值
            if hasattr(self.env, 'cache_state') and self.env.cache_state is not None:
                snapshot['cache_state'] = np.array(self.env.cache_state).reshape(self.N, self.M).copy()
            else:
                snapshot['cache_state'] = np.zeros((self.N, self.M), dtype=np.float32)
                
            if hasattr(self.env, 'age_state') and self.env.age_state is not None:
                snapshot['age_state'] = np.array(self.env.age_state).reshape(self.N, self.M).copy()
            else:
                snapshot['age_state'] = np.zeros((self.N, self.M), dtype=np.float32)
                
            if hasattr(self.env, 'accuracies') and self.env.accuracies is not None:
                snapshot['accuracies'] = np.array(self.env.accuracies).copy()
            else:
                # 使用初始精度作为默认值
                snapshot['accuracies'] = np.array(self.env.initial_acc).copy()
        except Exception as e:
            print(f"快照创建失败: {e}")
            # 创建默认快照
            snapshot = {
                'cache_state': np.zeros((self.N, self.M), dtype=np.float32),
                'age_state': np.zeros((self.N, self.M), dtype=np.float32),
                'accuracies': np.array(self.env.initial_acc).copy() if hasattr(self.env, 'initial_acc') 
                            else np.zeros(self.M, dtype=np.float32)
            }
            
        return snapshot

    def restore_lightweight_snapshot(self, snapshot):
        try:
            if snapshot is None:
                raise ValueError("快照不能为空")  
            if snapshot.get('cache_state') is not None:
                self.env.cache_state = snapshot['cache_state'].flatten().copy()
            if snapshot.get('age_state') is not None:
                self.env.age_state = snapshot['age_state'].flatten().copy()
            if snapshot.get('accuracies') is not None:
                self.env.accuracies = snapshot['accuracies'].copy()
        except Exception as e:
            print(f"快照恢复失败: {e}")
            # 初始化为默认状态
            self.env.cache_state = np.zeros(self.N * self.M, dtype=np.float32)
            self.env.age_state = np.zeros(self.N * self.M, dtype=np.float32)
            self.env.accuracies = np.zeros(self.M, dtype=np.float32)

    def create_isolated_env_state(self, base_snapshot):
        """创建隔离的环境状态 - 增强错误处理"""
        try:
            if base_snapshot is None:
                raise ValueError("基础快照不能为空")
                
            return {
                'cache_state': base_snapshot['cache_state'].copy(),
                'age_state': base_snapshot['age_state'].copy(),
                'accuracies': base_snapshot['accuracies'].copy()
            }
        except Exception as e:
            print(f"隔离状态创建失败: {e}")
            # 返回默认隔离状态
            return {
                'cache_state': np.zeros((self.N, self.M), dtype=np.float32),
                'age_state': np.zeros((self.N, self.M), dtype=np.float32),
                'accuracies': np.zeros(self.M, dtype=np.float32)
            }

    def get_requests_tensor(self, t):
        """获取请求张量 - 增强边界检查"""
        try:
            if t < 0 or t >= self.all_requests_tensor.size(0):
                return torch.zeros((self.N, self.M), device=self.device, dtype=torch.float32)
            
            tensor = self.all_requests_tensor[t]
            if tensor.dim() == 1:
                tensor = tensor.reshape(self.N, self.M)
            return tensor
        except Exception as e:
            print(f"请求张量获取失败 (t={t}): {e}")
            return torch.zeros((self.N, self.M), device=self.device, dtype=torch.float32)

    def enforce_storage_constraints(self, deployment_logits_tensor, temperature=5.0, eps=1e-3):
        try:
            N, M = self.N, self.M
            
            # 创建新张量而不是使用 reshape，避免潜在的原地操作问题
            logits = deployment_logits_tensor.view(N, M).clone()

            # 温度 sigmoid，避免 0/1 边界
            probs_soft = torch.sigmoid(logits / temperature).clamp(eps, 1 - eps)  # [N, M]

            # 使用 torch 排序：按概率降序
            sorted_vals, sorted_idx = torch.sort(probs_soft, dim=1, descending=True)  # [N, M]
            
            # 将 model sizes 按每行的 sorted_idx 排列 -> sizes_sorted [N, M]
            sizes_sorted = self.model_sizes_tensor[sorted_idx]  # 利用广播索引

            # 计算累计和，判断哪些被选中（每行）
            cumsum_sizes = torch.cumsum(sizes_sorted, dim=1)  # [N, M]
            capacity = float(self.edge_cache_storage)
            feasible_mask_sorted = cumsum_sizes <= capacity + 1e-9  # [N, M] 布尔

            # 将 sorted mask 转回原始模型顺序 - 使用非原地操作
            hard = torch.zeros_like(probs_soft, requires_grad=False)
            hard = hard.scatter(1, sorted_idx, feasible_mask_sorted.float())  # 非原地操作

            # 修复直通估计器，避免原地操作
            surrogate = probs_soft + (hard - probs_soft).detach()
            return surrogate, hard.detach()
            
        except Exception as e:
            print(f"存储约束执行失败: {e}")
            # 返回默认值
            default_surrogate = torch.zeros((N, M), device=self.device, requires_grad=True)
            default_hard = torch.zeros((N, M), device=self.device)
            return default_surrogate, default_hard

    def enforce_update_constraints(self, update_logits_tensor, deployment_surrogate, 
                                  temperature=3.0, eps=1e-4):
        """改进的更新约束实施 - 移除硬阈值，增强错误处理"""
        try:
            N, M = self.N, self.M
            
            update_logits = update_logits_tensor.view(N, M).clone()
            
            # 软更新概率
            update_probs_soft = torch.sigmoid(update_logits / temperature)
            update_probs_soft = torch.clamp(update_probs_soft, eps, 1 - eps)
            
            # 只能更新已部署的模型 - 使用软乘法
            update_probs_soft = update_probs_soft * deployment_surrogate
            
            # 硬决策（仅用于状态更新）
            update_hard = (update_probs_soft > 0.5).float().detach()
            
            # Surrogate with gradient flow
            update_surrogate = update_probs_soft + (update_hard - update_probs_soft).detach()
            
            return update_surrogate, update_hard
            
        except Exception as e:
            print(f"更新约束执行失败: {e}")
            # 返回默认值
            N, M = self.N, self.M
            default_surrogate = torch.zeros((N, M), device=self.device, requires_grad=True)
            default_hard = torch.zeros((N, M), device=self.device)
            return default_surrogate, default_hard


    def _record_epoch_metrics(self, epoch, metrics):
        """记录epoch指标 - 增强错误处理"""
        try:
            self.loss_history['epoch'].append(epoch + 1)
            for key, value in metrics.items():
                if key in self.loss_history:
                    self.loss_history[key].append(value)
            
            # 保存到CSV
            with open(self.loss_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if epoch == 0:  # 写入表头
                    writer.writerow(['epoch', 'meta_loss', 'accuracy_cost', 'update_cost', 
                                   'switching_cost', 'communication_cost', 'inference_cost',
                                   'hit_rate', 'timestamp'])
                
                writer.writerow([
                    epoch + 1, metrics['meta_loss'], metrics['accuracy_cost'], 
                    metrics['update_cost'], metrics['switching_cost'], 
                    metrics['communication_cost'], metrics['inference_cost'],
                    metrics['hit_rate'], datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])
        except Exception as e:
            print(f"指标记录失败: {e}")

    def _outer_evaluation(self, query_time_steps, fast_weights, isolated_state):
        device = self.device
        total_meta_loss = torch.tensor(0.0, device=device)
        total_hit = torch.tensor(0.0, device=device)
        total_switch = torch.tensor(0.0, device=device)
        total_comm = torch.tensor(0.0, device=device)
        total_update = torch.tensor(0.0, device=device)
        total_infer = torch.tensor(0.0, device=device)
        total_accuracy_cost = torch.tensor(0.0, device=device)

        try:
            iso_state = copy.deepcopy(isolated_state)
        except Exception as e:
            print(f"深拷贝隔离状态失败: {e}")
            iso_state = self.create_isolated_env_state({
                'cache_state': np.zeros((self.N, self.M), dtype=np.float32),
                'age_state': np.zeros((self.N, self.M), dtype=np.float32),
                'accuracies': np.zeros(self.M, dtype=np.float32)
            })

        try:
            prev_dep_tensor = torch.as_tensor(iso_state['cache_state'], device=device, dtype=torch.float32)
        except Exception as e:
            print(f"前一部署张量创建失败: {e}")
            prev_dep_tensor = torch.zeros((self.N, self.M), device=device, dtype=torch.float32)

        valid_evaluations = 0

        for idx, t in enumerate(query_time_steps):
            try:
                state_t = self._get_state_from_isolated(t, iso_state)
                if state_t.dim() == 1:
                    state_t = state_t.unsqueeze(0)  # [1, input_dim]

                requests_t = self.get_requests_tensor(t)
                if requests_t.dim() == 1:
                    requests_t = requests_t.reshape(self.N, self.M)

                # 前向传播 -> 联合输出
                with torch.no_grad():  # 外循环评估不需要梯度
                    network_outputs = self.meta_policy_forward_with_weights(state_t, fast_weights)

                    # 切分 deploy / update
                    mid_point = self.N * self.M
                    deploy_logits = network_outputs[..., :mid_point].view(self.N, self.M)
                    update_logits = network_outputs[..., mid_point:].view(self.N, self.M)

                    # 约束 - 直接获取硬决策
                    _, hard_deploy = self.enforce_storage_constraints(deploy_logits)
                    _, hard_update = self.enforce_update_constraints(update_logits, hard_deploy)

                    # 模拟状态更新 - 使用硬决策
                    hard_deployment = hard_deploy.cpu().numpy()
                    hard_update_np = hard_update.cpu().numpy()
                    iso_state = self._update_isolated_state(iso_state, hard_deployment, hard_update_np)

                    updated_age_tensor = torch.as_tensor(iso_state['age_state'], dtype=torch.float32, device=device)
                    updated_accuracies_tensor = torch.as_tensor(iso_state['accuracies'], dtype=torch.float32, device=device)

                # 成本计算 - 使用硬决策而不是代理变量
                cost, hit_rate, switching_cost, communication_cost, update_cost, inference_cost, accuracy_cost = \
                    self.calculate_cost_with_gradients(
                        hard_deploy, hard_update, requests_t, t,
                        prev_dep_tensor, updated_age_tensor, updated_accuracies_tensor
                    )

                if torch.isnan(cost) or torch.isinf(cost):
                    print(f"Warning: Invalid cost at query time {t}, skipping...")
                    continue

                # 累积指标
                total_meta_loss = total_meta_loss + cost
                total_hit = total_hit + hit_rate
                total_switch = total_switch + switching_cost
                total_comm = total_comm + communication_cost
                total_update = total_update + update_cost
                total_infer = total_infer + inference_cost
                total_accuracy_cost = total_accuracy_cost + accuracy_cost

                # 更新prev_dep_tensor
                prev_dep_tensor = hard_deploy.detach()
                valid_evaluations += 1

            except Exception as e:
                print(f"查询时间步 {t} 评估失败: {e}")
                continue

        # 平均化指标
        if valid_evaluations == 0:
            print("Warning: No valid evaluations in outer loop")
            zero_cost = torch.tensor(0.0, device=device)
            return (zero_cost, zero_cost, zero_cost, zero_cost, zero_cost, zero_cost, zero_cost)

        meta_loss = total_meta_loss / valid_evaluations
        hit_rate = total_hit / valid_evaluations
        switching_cost = total_switch / valid_evaluations
        communication_cost = total_comm / valid_evaluations
        update_cost = total_update / valid_evaluations
        inference_cost = total_infer / valid_evaluations
        accuracy_cost = total_accuracy_cost / valid_evaluations
        return meta_loss, hit_rate, switching_cost, communication_cost, update_cost, inference_cost, accuracy_cost

    def calculate_cost_with_gradients(self, deployed_surrogate, update_surrogate, requests, t,
                                 prev_deployment_surrogate, updated_age_state, updated_accuracies,
                                 lambda_smooth=0.1):
        try:
            device = self.device
            N, M = self.N, self.M
            NUM_SHARED_MODELS = self.NUM_SHARED_MODELS
            NUM_SPECIFIC_MODELS = self.NUM_SPECIFIC_MODELS

            # 初始化成本项
            costs = {
                'switching': torch.tensor(0.0, device=device, requires_grad=True),
                'update': torch.tensor(0.0, device=device, requires_grad=True),
                'inference': torch.tensor(0.0, device=device, requires_grad=True),
                'communication': torch.tensor(0.0, device=device, requires_grad=True),
                'accuracy': torch.tensor(0.0, device=device, requires_grad=True)
            }
            
            metrics = {
                'total_fragments_needed': torch.tensor(0.0, device=device),
                'total_fragments_available': torch.tensor(0.0, device=device)
            }

            # 1. 改进的切换成本（使用平滑正则化）
            if prev_deployment_surrogate is not None:
                # 1.1 原有的基础激活切换成本
                activation_switches = deployed_surrogate * (1 - prev_deployment_surrogate)
                base_switching_cost = torch.sum(activation_switches * self.model_switch_costs)
                
                # # 1.2 平滑正则化 - 惩罚部署决策的变化，提高稳定性和减少频繁切换
                # deployment_change = torch.abs(deployed_surrogate - prev_deployment_surrogate)
                # smoothness_penalty = lambda_smooth * torch.sum(deployment_change ** 2)
                # 使用示例
                smoothness_penalty = lambda_smooth * self.pseudo_huber_regularizer(
                    deployed_surrogate, prev_deployment_surrogate, delta=0.1
                )
                
                # 组合切换成本
                costs['switching'] = base_switching_cost + smoothness_penalty

            # 2. 更新成本-更新成本不需要区分共享还是非共享，因为共享的模型参数不更新。
            # if self.shared_flag:
            #     # 共享模式：考虑依赖关系，共享模型需要级联更新
            #     extended_update_mask = update_surrogate.clone()
                
            #     for specific_idx in range(NUM_SPECIFIC_MODELS):
            #         global_specific_idx = NUM_SHARED_MODELS + specific_idx
            #         nodes_with_update = torch.nonzero(update_surrogate[:, global_specific_idx] > 0.5)
                    
            #         if len(nodes_with_update) > 0:
            #             shared_dependencies = self.dependency_indices_cache[specific_idx]
            #             for shared_idx in shared_dependencies:
            #                 extended_update_mask[nodes_with_update, shared_idx] = 1.0
                
            #     costs['update'] = costs['update'] + torch.sum(
            #         extended_update_mask * deployed_surrogate * self.model_update_costs
            #     )
            # else:
            # 非共享模式：每个模型独立更新，无级联效应
            costs['update'] = costs['update'] + torch.sum(
                update_surrogate * deployed_surrogate * self.model_update_costs
            )

            # 3. 精度和通信成本计算 - 保持原有逻辑
            specific_requests = requests[:, NUM_SHARED_MODELS:]
            specific_deployed = deployed_surrogate[:, NUM_SHARED_MODELS:]
            
            request_mask = specific_requests > 1e-6

            for specific_idx in range(NUM_SPECIFIC_MODELS):
                model_global_id = specific_idx + NUM_SHARED_MODELS
                req_rates = specific_requests[:, specific_idx]
                mask = request_mask[:, specific_idx]

                if not torch.any(mask):
                    continue
                
                #测试是二维，训练时是一维
                model_accuracies = updated_accuracies[0, model_global_id] 
                accuracy_min = self.accuracy_min_tensor[model_global_id]
                accuracy_deficit = torch.clamp(accuracy_min - model_accuracies, min=0.0)

                # 对所有有请求的节点应用相同的全局精度成本
                accuracy_cost = torch.sum(mask * req_rates) * accuracy_deficit
                
                # # 只对有请求的节点计算精度成本
                costs['accuracy'] = costs['accuracy'] + accuracy_cost

                if self.shared_flag:
                    # 共享模式：考虑分片依赖
                    fragments_needed = 1 + len(self.dependency_indices_cache[specific_idx])
                    metrics['total_fragments_needed'] += torch.sum(mask * req_rates) * fragments_needed

                    availability_probs = self._calculate_fragment_availability(
                        specific_idx, specific_deployed[:, specific_idx], deployed_surrogate
                    )
                    
                    metrics['total_fragments_available'] += torch.sum(
                        mask * req_rates * availability_probs['expected_fragments']
                    )

                    # 推理成本基于完全可用性
                    costs['inference'] = costs['inference'] + torch.sum(
                        mask * req_rates * self.model_inference_cost * 
                        availability_probs['full_availability']
                    )

                    # 通信成本考虑分片情况
                    comm_costs = self._calculate_communication_costs(
                        mask, req_rates, availability_probs
                    )
                    costs['communication'] = costs['communication'] + comm_costs
                else:
                    # 非共享模式：模型整体部署，无分片概念
                    fragments_needed = 1  # 只需要特定模型本身
                    metrics['total_fragments_needed'] += torch.sum(mask * req_rates) * fragments_needed
                    
                    # 可用性就是模型是否部署
                    model_available = specific_deployed[:, specific_idx]
                    metrics['total_fragments_available'] += torch.sum(mask * req_rates * model_available)
                    
                    # 推理成本
                    costs['inference'] = costs['inference'] + torch.sum(
                        mask * req_rates * self.model_inference_cost * model_available
                    )
                    
                    # 通信成本：命中用本地延迟，未命中用云端延迟
                    local_cost = mask * req_rates * self.delay_local * model_available
                    cloud_cost = mask * req_rates * self.delay_cloud * (1 - model_available)
                    costs['communication'] = costs['communication'] + torch.sum(local_cost + cloud_cost)

            # 计算命中率
            hit_rate = torch.where(
                metrics['total_fragments_needed'] > 1e-5,
                metrics['total_fragments_available'] / metrics['total_fragments_needed'],
                torch.tensor(0.0, device=device)
            )

            # 总成本
            total_cost = sum(costs.values())
            return (total_cost, hit_rate, costs['switching'], costs['communication'],
                    costs['update'], costs['inference'], costs['accuracy'])

        except Exception as e:
            print(f"成本计算错误 (t={t}): {e}")
            # 返回一个具有梯度的零成本
            zero_cost = torch.tensor(0.0, device=device, requires_grad=True)
            return (zero_cost, zero_cost, zero_cost, zero_cost, zero_cost, zero_cost, zero_cost)

    def _calculate_fragment_availability(self, specific_idx, specific_deployed, all_deployed):
        """计算分片可用性概率 - 仅在共享模式下有效"""
        if not self.shared_flag:
            # 非共享模式：返回简单的可用性
            return {
                'full_availability': specific_deployed,
                'partial_availability': torch.zeros_like(specific_deployed),
                'expected_fragments': specific_deployed
            }
        """计算分片可用性概率"""
        dependency_indices = self.dependency_indices_cache[specific_idx]
        
        if not dependency_indices:
            # 无依赖情况
            return {
                'full_availability': specific_deployed,
                'partial_availability': torch.zeros_like(specific_deployed),
                'expected_fragments': specific_deployed
            }
        
        # 有依赖情况 - 使用更精确的概率模型
        shared_deployed = all_deployed[:, dependency_indices]
        
        # 完全可用概率（特定模型 AND 所有依赖都可用）
        shared_all_prob = torch.prod(shared_deployed, dim=1)
        full_availability = specific_deployed * shared_all_prob
        
        # 部分可用概率（至少一个共享模型可用但不是全部）
        shared_any_prob = 1 - torch.prod(1 - shared_deployed, dim=1)
        partial_availability = specific_deployed * (shared_any_prob - shared_all_prob)
        
        # 期望可用分片数
        expected_shared_fragments = torch.sum(shared_deployed, dim=1)
        expected_fragments = specific_deployed * (1 + expected_shared_fragments)
        
        return {
            'full_availability': full_availability,
            'partial_availability': partial_availability,
            'expected_fragments': expected_fragments
        }

    def _update_isolated_state(self, isolated_state, hard_deployment, hard_update):
        hard_deployment = np.array(hard_deployment).reshape(self.N, self.M)
        hard_update = np.array(hard_update).reshape(self.N, self.M)
        
        previous_deployment = isolated_state['cache_state'].copy()
        isolated_state['cache_state'] = hard_deployment.copy()
        
        # age_state保持(N, M)维度
        if isolated_state['age_state'].ndim == 1:
            isolated_state['age_state'] = isolated_state['age_state'].reshape(self.N, self.M)
        
        # accuracies应该是(M,)维度 - 全局模型精度
        if isolated_state['accuracies'].ndim != 1:
            isolated_state['accuracies'] = isolated_state['accuracies'].flatten()[:self.M]
        
        new_age_state = isolated_state['age_state'].copy()
        new_acc_state = isolated_state['accuracies'].copy()  # (M,)维度
        
        # 对每个模型，基于所有节点的最小age来更新全局精度
        for j in range(self.M):
            updated_nodes = []
            for i in range(self.N):
                if hard_update[i, j] > 0:
                    new_age_state[i, j] = 0
                    updated_nodes.append(i)
                elif hard_deployment[i, j] > 0:
                    if previous_deployment[i, j] > 0:
                        new_age_state[i, j] += 1
                    else:
                        new_age_state[i, j] = 0
                else:
                    new_age_state[i, j] = 0
            
            # 全局精度基于该模型在所有节点中的最小age
            if len(updated_nodes) > 0:
                # 至少有一个节点更新了该模型
                new_acc_state[j] = self.env.initial_acc[j]
            else:
                # 使用最小age（最新的版本）来计算精度
                deployed_ages = [new_age_state[i, j] for i in range(self.N) 
                            if hard_deployment[i, j] > 0]
                if deployed_ages:
                    min_age = min(deployed_ages)
                    age_ratio = np.clip(min_age / args.max_age, 0.0, 1.0)
                    decay_factor = np.exp(-self.age_decay_factor * age_ratio)
                    new_acc_state[j] = self.env.initial_acc[j] * decay_factor
                # else: 保持当前精度
        
        new_acc_state = np.clip(new_acc_state, 0.1, 1.0)
        
        isolated_state['age_state'] = new_age_state
        isolated_state['accuracies'] = new_acc_state  # 保持(M,)维度
        
        return isolated_state
        
    def meta_train(self):
        # 保存初始环境状态，用于第一个任务
        current_snapshot = self.lightweight_snapshot()
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            epoch_metrics = {
                'meta_loss': 0.0, 'update_cost': 0.0, 'switching_cost': 0.0,
                'communication_cost': 0.0, 'inference_cost': 0.0, 'accuracy_cost': 0.0,
                'hit_rate': 0.0
            }

            exploration_factor = max(0.01, 0.5 * (1 - epoch / (self.num_epochs * 2)))
            processed_tasks = 0
            skipped_tasks = 0

            for task_idx in range(self.num_tasks):
                if task_idx % 20 == 0:
                    print(f"  Processing task {task_idx}/{self.num_tasks} (Skipped: {skipped_tasks})")

                task_start_time = task_idx * self.task_length
                if task_start_time >= self.end_time:
                    break
                task_end_time = min(task_start_time + self.task_length, self.env.total_time_slots)

                try:
                    self.restore_lightweight_snapshot(current_snapshot)
                except Exception as e:
                    print(f"Warning: Failed to restore snapshot for task {task_idx}: {e}")
                    skipped_tasks += 1
                    continue

                # === 划分支持集 (support) 和 查询集 (query) ===
                times = list(range(task_start_time, task_end_time))    
                # 确保时间序列的连续性，支持集在前，查询集在后
                split_idx = max(2, int(len(times) * 0.7))  # 70% 支持集，至少2个时隙
                support_times = times[:split_idx]
                query_times = times[split_idx:]

                # 确保查询集非空且有足够样本
                if len(query_times) < 1:
                    skipped_tasks += 1
                    continue

                # 初始化临时变量
                fast_weights = None
                adapted_fast_weights = None
                final_isolated_state = None
                meta_loss = None
                hit_rate = None
                switching_cost = None
                communication_cost = None
                update_cost = None
                inference_cost = None
                accuracy_cost = None

                try:
                    # === 内循环：基于支持集任务时隙进行快速适应 ===
                    fast_weights = {
                        name: param.detach().clone().requires_grad_(True)
                        for name, param in self.meta_policy.named_parameters()
                    }
                    
                    adapted_fast_weights, final_isolated_state = self.optimized_inner_loop(
                        support_times,
                        self.create_isolated_env_state(current_snapshot),
                        fast_weights,
                        exploration_factor
                    )

                    # === 外循环：在查询集时隙上评估，得到 meta_loss ===
                    (meta_loss, hit_rate, switching_cost,
                    communication_cost, update_cost,
                    inference_cost, accuracy_cost) = self._outer_evaluation(
                        query_times, adapted_fast_weights, final_isolated_state
                    )

                    # 检查损失值有效性
                    if torch.isnan(meta_loss) or torch.isinf(meta_loss):
                        print(f"Warning: Invalid loss at task {task_idx} (NaN or Inf), skipping...")
                        skipped_tasks += 1
                        continue

                    # === 元参数更新 ===
                    self.meta_optimizer.zero_grad()
                    try:
                        meta_loss.backward(retain_graph=False)
                        
                        # 修复 NaN 或 Inf 梯度，并检查梯度爆炸
                        grad_norm = 0.0
                        has_invalid_grad = False
                        
                        for name, param in self.meta_policy.named_parameters():
                            if param.grad is not None:
                                # 处理 NaN 和 Inf 梯度
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    param.grad[torch.isnan(param.grad) | torch.isinf(param.grad)] = 0.0
                                    has_invalid_grad = True
                                
                                grad_norm += param.grad.data.norm(2).item() ** 2
                        
                        grad_norm = grad_norm ** 0.5
                        
                        if has_invalid_grad:
                            print(f"Warning: Fixed invalid gradients at task {task_idx}")
                        
                        # 检查梯度爆炸
                        if grad_norm > 100.0:
                            print(f"Warning: Large gradient norm {grad_norm:.2f} at task {task_idx}")

                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.meta_policy.parameters(), max_norm=1.0)
                        self.meta_optimizer.step()
                        
                    except Exception as e:
                        print(f"反向传播失败: {e}")
                        skipped_tasks += 1
                        continue
                    
                    # === 更新指标 ===
                    epoch_metrics['meta_loss'] += meta_loss.detach().item()
                    epoch_metrics['hit_rate'] += hit_rate.detach().item()
                    epoch_metrics['switching_cost'] += switching_cost.detach().item()
                    epoch_metrics['communication_cost'] += communication_cost.detach().item()
                    epoch_metrics['update_cost'] += update_cost.detach().item()
                    epoch_metrics['inference_cost'] += inference_cost.detach().item()
                    epoch_metrics['accuracy_cost'] += accuracy_cost.detach().item()

                    processed_tasks += 1

                    # 更新当前状态为任务结束后的状态
                    current_snapshot = self.lightweight_snapshot()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"CUDA OOM at task {task_idx}, clearing cache and skipping...")
                        skipped_tasks += 1
                        continue
                    else:
                        print(f"Runtime error at task {task_idx}: {e}")
                        skipped_tasks += 1
                        continue
                except Exception as e:
                    print(f"Unexpected error at task {task_idx}: {e}")
                    skipped_tasks += 1
                    continue
                finally:
                    # 清理内存
                    variables_to_clean = [
                        fast_weights, adapted_fast_weights, final_isolated_state,
                        meta_loss, hit_rate, switching_cost, communication_cost,
                        update_cost, inference_cost, accuracy_cost
                    ]
                    
                    for var in variables_to_clean:
                        if var is not None:
                            del var
                    
                    # 清理局部变量
                    if 'times' in locals():
                        del times
                    if 'support_times' in locals():
                        del support_times
                    if 'query_times' in locals():
                        del query_times

            # === 计算平均指标 ===
            if processed_tasks > 0:
                for key in epoch_metrics:
                    epoch_metrics[key] /= processed_tasks
            else:
                print(f"Warning: No tasks processed in epoch {epoch+1}")
                for key in epoch_metrics:
                    epoch_metrics[key] = 0.0

            # 记录结果
            self._record_epoch_metrics(epoch, epoch_metrics)

            # 保存模型
            if (epoch + 1) % self.save_every == 0 or epoch == self.num_epochs - 1:
                shared_mode_str = "Shared" if self.shared_flag else "NonShared"
                try:
                    model_path = os.path.join(
                        self.save_dir,
                        f'meta_policy_{shared_mode_str}_Test_Innerlr{args.inner_lr}_Melr{args.meta_lr}_HIDDEN_DIM{args.HIDDEN_DIM}_train_NumTask{args.train_task_num}_epoch{epoch+1}.pth'
                    )
                    self.save_model(model_path)
                    print(f"模型已保存到 {model_path}")
                except Exception as e:
                    print(f"模型保存失败: {e}")

            # 打印指标
            print(f"Processed: {processed_tasks}/{self.num_tasks} tasks (Skipped: {skipped_tasks})")
            print(f"Meta Loss: {epoch_metrics['meta_loss']:.4f} | "
                f"Hit Rate: {epoch_metrics['hit_rate']:.4f} | "
                f"Switching Cost: {epoch_metrics['switching_cost']:.4f} | "
                f"Communication Cost: {epoch_metrics['communication_cost']:.4f} | "
                f"Update Cost: {epoch_metrics['update_cost']:.4f} | "
                f"Inference Cost: {epoch_metrics['inference_cost']:.4f} | "
                f"Accuracy Cost: {epoch_metrics['accuracy_cost']:.4f}")
                
            # 每个epoch结束后重置状态，以便下一个epoch从初始状态开始
            current_snapshot = self.lightweight_snapshot()

    def _calculate_communication_costs(self, mask, req_rates, availability_probs):
        """计算通信成本"""
        # 本地通信（完全命中）
        local_cost = (mask * req_rates * self.delay_local * 
                     availability_probs['full_availability'])
        
        # 部分云端通信（部分命中）
        partial_cost = (mask * req_rates * self.delay_cloud * 0.5 *  # 假设50%云端延迟
                       availability_probs['partial_availability'])
        
        # 完全云端通信（完全未命中）
        miss_prob = 1 - availability_probs['full_availability'] - availability_probs['partial_availability']
        cloud_cost = mask * req_rates * self.delay_cloud * miss_prob
        
        return torch.sum(local_cost + partial_cost + cloud_cost)

    def optimized_inner_loop(self, support_time_steps, base_isolated_state, fast_weights, exploration_factor):
        isolated_state = copy.deepcopy(base_isolated_state)
        device = self.device

        # 创建fast_weights的完全独立副本（可微）
        current_fast_weights = {}
        for k, v in fast_weights.items():
            current_fast_weights[k] = v.detach().clone().requires_grad_(True)

        # 我们逐步或批量地处理支持集时间步，计算损失并更新 fast_weights
        accumulated_loss = torch.tensor(0.0, device=device, requires_grad=True)
        grad_accum_count = 0

        for i in range(0, len(support_time_steps), self.batch_size):
            batch_steps = support_time_steps[i:i + self.batch_size]
            batch_loss, isolated_state = self.batch_process_inner_loop(
                batch_steps, isolated_state, current_fast_weights, exploration_factor
            )

            if batch_loss.dim() > 0:
                batch_loss = batch_loss.mean()

            accumulated_loss = accumulated_loss + batch_loss
            grad_accum_count += 1

            if grad_accum_count >= self.gradient_accumulation_steps:
                avg_loss = accumulated_loss / grad_accum_count

                # 关键修改：使用 retain_graph=True 并正确处理计算图
                grads = torch.autograd.grad(
                    avg_loss,
                    list(current_fast_weights.values()),
                    create_graph=True,
                    retain_graph=True,  # 保留计算图
                    allow_unused=False
                )

                # 使用 grads 更新 fast weights
                updated_fast_weights = {}
                for (name, w), g in zip(current_fast_weights.items(), grads):
                    if g is not None:
                        updated_fast_weights[name] = w - self.inner_lr * g
                    else:
                        updated_fast_weights[name] = w

                # 重新设置 current_fast_weights（保持计算图）
                current_fast_weights = {}
                for k, v in updated_fast_weights.items():
                    current_fast_weights[k] = v.requires_grad_(True)

                # 重置累积 - 修复：确保新的累积张量
                accumulated_loss = torch.tensor(0.0, device=device, requires_grad=True)
                grad_accum_count = 0
                
                # 手动释放不再需要的计算图
                del avg_loss, grads, updated_fast_weights
    
        # 处理剩余累积
        if grad_accum_count > 0:
            avg_loss = accumulated_loss / grad_accum_count
            
            # 修复：最后一次梯度计算也要正确处理
            grads = torch.autograd.grad(
                avg_loss,
                list(current_fast_weights.values()),
                create_graph=True,
                retain_graph=True,  # 保留计算图
                allow_unused=True
            )

            updated_fast_weights = {}
            for (name, w), g in zip(current_fast_weights.items(), grads):
                if g is not None:
                    updated_fast_weights[name] = w - self.inner_lr * g
                else:
                    updated_fast_weights[name] = w

            # 最终的 fast_weights 准备用于外层评估
            current_fast_weights = {}
            for k, v in updated_fast_weights.items():
                current_fast_weights[k] = v.requires_grad_(True)
                
            # 手动释放
            del avg_loss, grads, updated_fast_weights
    
        return current_fast_weights, isolated_state


    def save_model(self, path):
        """保存模型"""
        try:
            checkpoint = {
                'model_state_dict': self.meta_policy.state_dict(),
                'optimizer_state_dict': self.meta_optimizer.state_dict(),
                'loss_history': self.loss_history,
                'age_decay_factor': self.age_decay_factor,
                'env_config': {
                    'NUM_EDGE_NODES': self.N,
                    'NUM_MODELS': self.M,
                    'NUM_SPECIFIC_MODELS': self.NUM_SPECIFIC_MODELS,
                    'NUM_SHARED_MODELS': self.NUM_SHARED_MODELS
                }
            }
            torch.save(checkpoint, path)
            print(f"模型成功保存到: {path}")
        except Exception as e:
            print(f"保存模型时出错: {e}")

    def batch_process_inner_loop(self, time_steps, base_isolated_state, fast_weights, exploration_factor):
        """批量处理内循环时间步 - 适配联合输出网络 (deploy+update)"""
        batch_size = len(time_steps)
        device = self.device
        
        # 预分配批量张量
        batch_states = self.tensor_buffers.get_buffer('batch_states', (batch_size, self.calculate_input_dim()))
        batch_requests = self.tensor_buffers.get_buffer('batch_requests', (batch_size, self.N, self.M))
        # 新增缓冲区用于存储硬决策
        batch_hard_deploy = self.tensor_buffers.get_buffer('batch_hard_deploy', (batch_size, self.N, self.M))
        batch_hard_update = self.tensor_buffers.get_buffer('batch_hard_update', (batch_size, self.N, self.M))

        # 批量获取状态和请求
        isolated_state = copy.deepcopy(base_isolated_state)
            
        if isolated_state['cache_state'].ndim == 1:
            isolated_state['cache_state'] = isolated_state['cache_state'].reshape(self.N, self.M)
        if isolated_state['age_state'].ndim == 1:
            isolated_state['age_state'] = isolated_state['age_state'].reshape(self.N, self.M)
        
        for i, t in enumerate(time_steps):
            state_key = self._get_state_cache_key(t, isolated_state)
            if state_key in self.state_cache:
                batch_states[i] = self.state_cache[state_key]
            else:
                state = self._get_state_from_isolated(t, isolated_state)
                batch_states[i] = state
                self.state_cache[state_key] = state.clone()
            
            batch_requests[i] = self.get_requests_tensor(t)
        
        # 前向传播
        batch_outputs = self.meta_policy_forward_with_weights(batch_states, fast_weights)
        mid_point = self.N * self.M
        batch_deploy_logits = batch_outputs[..., :mid_point].view(batch_size, self.N, self.M)
        batch_update_logits = batch_outputs[..., mid_point:].view(batch_size, self.N, self.M)

        # 添加探索噪声
        if exploration_factor > 0:
            batch_deploy_logits += torch.randn_like(batch_deploy_logits) * exploration_factor
            batch_update_logits += torch.randn_like(batch_update_logits) * exploration_factor
        
        # 批量约束实施
        batch_deployed_surrogate = self.tensor_buffers.get_buffer('batch_deployed', (batch_size, self.N, self.M))
        batch_update_surrogate = self.tensor_buffers.get_buffer('batch_update', (batch_size, self.N, self.M))
        
        for i in range(batch_size):
            deployed_surr, hard_deploy = self.enforce_storage_constraints(batch_deploy_logits[i])
            update_surr, hard_update = self.enforce_update_constraints(batch_update_logits[i], deployed_surr)
            batch_deployed_surrogate[i] = deployed_surr
            batch_update_surrogate[i] = update_surr
            batch_hard_deploy[i] = hard_deploy  # 存储硬决策
            batch_hard_update[i] = hard_update  # 存储硬决策

        # 成本计算
        total_cost = torch.tensor(0.0, device=device, requires_grad=True)
        prev_dep_tensor = torch.as_tensor(isolated_state['cache_state'], device=device, dtype=torch.float32)
        
        for i in range(batch_size):
            with torch.no_grad():
                hard_deployment = batch_hard_deploy[i]
                hard_update_decision = batch_hard_update[i]
                
                isolated_state = self._update_isolated_state(
                    isolated_state, hard_deployment.cpu().numpy(), hard_update_decision.cpu().numpy()
                )
                            
                updated_age_tensor = torch.as_tensor(isolated_state['age_state'], device=device, dtype=torch.float32)
                updated_accuracies_tensor = torch.as_tensor(isolated_state['accuracies'], device=device, dtype=torch.float32)
            
            cost, _, _, _, _, _, _ = self.calculate_cost_with_gradients(
                batch_deployed_surrogate[i], 
                batch_update_surrogate[i], 
                batch_requests[i], 
                time_steps[i], 
                prev_dep_tensor, 
                updated_age_tensor,
                updated_accuracies_tensor
            )
            total_cost = total_cost + cost
            prev_dep_tensor = hard_deployment.detach()  # 使用硬决策更新prev_dep_tensor
        
        avg_loss = total_cost / batch_size
        # 释放所有缓冲区
        self.tensor_buffers.release_buffer('batch_states')
        self.tensor_buffers.release_buffer('batch_requests')
        self.tensor_buffers.release_buffer('batch_deployed')
        self.tensor_buffers.release_buffer('batch_update')
        self.tensor_buffers.release_buffer('batch_hard_deploy')  # 释放新增缓冲区
        self.tensor_buffers.release_buffer('batch_hard_update')
        return avg_loss, isolated_state

    def pseudo_huber_regularizer(self, y_t, y_t_minus_1, delta=0.1):
        epsilon = 1e-8
        delta = max(delta, epsilon)
        
        diff = y_t - y_t_minus_1
        inside_sqrt = 1 + (diff/delta)**2 + epsilon  # 关键改进
        
        return torch.sum(delta**2 * (torch.sqrt(inside_sqrt) - 1))

    def load_model(self, path, env=None):
        """加载模型"""
        try:
            if not os.path.exists(path):
                print(f"模型文件 {path} 不存在")
                return False
            
            checkpoint = torch.load(path, map_location=self.device)
            self.meta_policy.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'age_decay_factor' in checkpoint:
                self.age_decay_factor = checkpoint['age_decay_factor']
            
            print(f"模型已从 {path} 加载")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False

class StateCache:
    """状态缓存类"""
    def __init__(self, max_size=100):
        self.cache = {}
        self.access_order = deque()
        self.max_size = max_size
    
    def __contains__(self, key):
        return key in self.cache
    
    def __getitem__(self, key):
        if key in self.cache:
            # LRU更新
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        raise KeyError(key)
    
    def __setitem__(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # 移除最久未使用的
            oldest = self.access_order.popleft()
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)

class TensorBufferPool:
    """张量缓冲池"""
    def __init__(self, device, N, M):
        self.device = device
        self.N = N
        self.M = M
        self.buffers = {}
    
    def get_buffer(self, name, shape):
        """获取或创建缓冲区"""
        if name not in self.buffers or self.buffers[name].shape != shape:
            self.buffers[name] = torch.zeros(shape, device=self.device)
        return self.buffers[name]
    
    def clear_buffers(self):
        """清理所有缓冲区"""
        self.buffers.clear()

    def release_buffer(self, name):
        """释放指定名称的缓冲区"""
        if name in self.buffers:
            del self.buffers[name]
   
