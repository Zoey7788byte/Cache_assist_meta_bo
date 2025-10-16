# import torch
# import numpy as np
# import copy
# from collections import deque

# class MetaOptimization:
#     def __init__(self, env, args, meta_learner):
#         self.env = env
#         self.args = args
#         self.meta_learner = meta_learner
#         self.N = env.NUM_EDGE_NODES
#         self.M = env.NUM_MODELS  # 使用M而不是NUM_MODELS，与内循环保持一致
#         self.edge_cache_storage = args.edge_cache_storage
#         self.NUM_SHARED_MODELS = env.NUM_SHARED_MODELS
#         self.NUM_SPECIFIC_MODELS = env.NUM_SPECIFIC_MODELS
        
#         # 保存元学习的初始参数
#         self.meta_initial_params = copy.deepcopy(self.meta_learner.meta_policy.state_dict())
        
#         # 适应阶段参数
#         self.adaptation_steps = args.adaptation_steps
#         self.adaptation_lr = args.adaptation_lr
#         self.current_task_params = None
#         self.task_step_count = 0
#         self.current_task_id = None
        
#         # 改进的缓冲区管理
#         self.adaptation_buffer = deque(maxlen=getattr(args, 'buffer_size', 100))
#         self.min_adaptation_samples = getattr(args, 'min_adaptation_samples', 10)
        
#         # 渐进式适应
#         self.progressive_params = None
#         self.adaptation_momentum = getattr(args, 'adaptation_momentum', 0.9)
        
#         # 添加设备和缓冲区管理（与内循环保持一致）
#         self.device = meta_learner.device if hasattr(meta_learner, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.tensor_buffers = meta_learner.tensor_buffers if hasattr(meta_learner, 'tensor_buffers') else None
        
#         # 状态缓存（与内循环保持一致）
#         self.state_cache = {}

#     def calculate_input_dim(self):
#         """计算输入维度，与内循环保持一致"""
#         return self.meta_learner.calculate_input_dim() if hasattr(self.meta_learner, 'calculate_input_dim') else self.N * self.M * 2

#     def start_new_task(self, task_id):
#         """开始一个新任务，重置模型参数和计数器"""
#         print(f"Starting new task {task_id}")
#         self.current_task_id = task_id
#         self.task_step_count = 0
#         self.adaptation_buffer.clear()
        
#         # 清空状态缓存
#         self.state_cache.clear()
        
#         # 重置为元学习初始参数
#         self.meta_learner.meta_policy.load_state_dict(self.meta_initial_params)
#         self.current_task_params = None
#         self.progressive_params = copy.deepcopy(self.meta_initial_params)

#     def batch_adaptation_step(self, batch_states, batch_requests, fast_weights=None, exploration_factor=0.1, time_steps=None, base_isolated_state=None):
#         if len(batch_states) == 0:
#             return False
        
#         batch_size = len(batch_states)
#         device = self.device
        
#         # 准备批量数据
#         states_tensor = torch.stack(batch_states) if isinstance(batch_states[0], torch.Tensor) else torch.tensor(batch_states, device=device, dtype=torch.float32)
        
#         # 确保requests是正确格式 - 每个请求应该是(N, M)
#         if isinstance(batch_requests[0], torch.Tensor):
#             requests_tensor = torch.stack(batch_requests)
#         else:
#             requests_tensor = torch.tensor(batch_requests, device=device, dtype=torch.float32)
        
#         # 验证请求张量的形状
#         if requests_tensor.shape[-2:] != (self.N, self.M):
#             print(f"Warning: Unexpected requests shape {requests_tensor.shape}, expected (..., {self.N}, {self.M})")
#             # 尝试reshape或扩展
#             if requests_tensor.dim() == 2 and requests_tensor.shape[-1] == self.M:
#                 # 假设是单节点情况，扩展到所有节点
#                 requests_tensor = requests_tensor.unsqueeze(-2).expand(-1, self.N, -1)
#             elif requests_tensor.dim() == 1:
#                 # 单个请求向量，扩展到所有节点
#                 requests_tensor = requests_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, self.N, -1)
        
#         # 使用当前权重或fast_weights
#         current_weights = fast_weights if fast_weights is not None else None
        
#         # 创建优化器（如果不使用fast weights）
#         optimizer = None
#         if current_weights is None:
#             optimizer = torch.optim.Adam(self.meta_learner.meta_policy.parameters(), lr=self.adaptation_lr)
        
#         adaptation_losses = []
        
#         for step in range(self.adaptation_steps):
#             if optimizer:
#                 optimizer.zero_grad()
            
#             # 前向传播（类似内循环的批处理）
#             if current_weights is not None:
#                 batch_outputs = self.meta_learner.meta_policy_forward_with_weights(states_tensor, current_weights)
#             else:
#                 batch_outputs = self.meta_learner.meta_policy(states_tensor)
            
#             mid_point = self.N * self.M
#             batch_deploy_logits = batch_outputs[..., :mid_point].view(batch_size, self.N, self.M)
#             batch_update_logits = batch_outputs[..., mid_point:].view(batch_size, self.N, self.M)
            
#             # 添加探索噪声（与内循环保持一致）
#             if exploration_factor > 0:
#                 batch_deploy_logits = batch_deploy_logits + torch.randn_like(batch_deploy_logits) * exploration_factor
#                 batch_update_logits = batch_update_logits + torch.randn_like(batch_update_logits) * exploration_factor
            
#             # 批量约束实施（与内循环保持一致）
#             batch_deployed_surrogate = []
#             batch_update_surrogate = []
            
#             for i in range(batch_size):
#                 deployed_surr, _ = self.meta_learner.enforce_storage_constraints(batch_deploy_logits[i])
#                 update_surr, _ = self.meta_learner.enforce_update_constraints(batch_update_logits[i], deployed_surr)
#                 batch_deployed_surrogate.append(deployed_surr)
#                 batch_update_surrogate.append(update_surr)
            
#             batch_deployed_surrogate = torch.stack(batch_deployed_surrogate)
#             batch_update_surrogate = torch.stack(batch_update_surrogate)
            
#             # 计算适应损失（使用内循环的cost函数）
#             loss = self.compute_batch_adaptation_loss(
#                 batch_deployed_surrogate, 
#                 batch_update_surrogate,
#                 requests_tensor, 
#                 time_steps,
#                 base_isolated_state
#             )
            
#             adaptation_losses.append(loss.item())
            
#             # 反向传播和优化
#             if current_weights is None:
#                 # 直接优化模型参数
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.meta_learner.meta_policy.parameters(), max_norm=1.0)
#                 optimizer.step()
#             else:
#                 # 使用fast weights机制
#                 grads = torch.autograd.grad(loss, current_weights.values(), create_graph=True, allow_unused=True)
#                 # 过滤None梯度
#                 valid_grads = []
#                 valid_params = []
#                 for (name, param), grad in zip(current_weights.items(), grads):
#                     if grad is not None:
#                         valid_grads.append(grad)
#                         valid_params.append((name, param))
                
#                 # 更新有效参数
#                 for (name, param), grad in zip(valid_params, valid_grads):
#                     current_weights[name] = param - self.adaptation_lr * grad
        
#         # 保存适应后的参数
#         if current_weights is not None:
#             self.current_task_params = copy.deepcopy(current_weights)
#         else:
#             self.current_task_params = copy.deepcopy(self.meta_learner.meta_policy.state_dict())
        
#         # print(f"Batch adaptation completed. Average loss: {np.mean(adaptation_losses):.4f}")
#         return True

#     def compute_batch_adaptation_loss(self, batch_deployed, batch_update, batch_requests, time_steps=None, base_isolated_state=None):
#         batch_size = batch_deployed.shape[0]
#         total_cost = torch.tensor(0.0, device=self.device, requires_grad=True)
        
#         # 使用基础状态或创建默认状态
#         if base_isolated_state is None:
#             isolated_state = self.get_default_isolated_state()
#         else:
#             isolated_state = copy.deepcopy(base_isolated_state)
        
#         # 确保状态格式正确
#         if isolated_state['cache_state'].ndim == 1:
#             isolated_state['cache_state'] = isolated_state['cache_state'].reshape(self.N, self.M)
#         if isolated_state['age_state'].ndim == 1:
#             isolated_state['age_state'] = isolated_state['age_state'].reshape(self.N, self.M)
        
#         # 初始化前一部署状态
#         prev_dep_tensor = torch.as_tensor(
#             isolated_state['cache_state'], 
#             device=self.device, 
#             dtype=torch.float32
#         ).view(self.N, self.M)
        
#         # 按时隙顺序处理（与内循环保持一致）
#         for i in range(batch_size):
#             # 获取当前时隙信息
#             current_time = time_steps[i] if time_steps and i < len(time_steps) else i
            
#             # 当前时隙的请求应该是 (N, M) - 所有节点的请求
#             current_requests = batch_requests[i]  # shape: (N, M)
#             if current_requests.dim() == 1:
#                 # 如果只有一维，假设是单节点情况，需要扩展
#                 current_requests = current_requests.unsqueeze(0).expand(self.N, -1)
            
#             # 使用硬决策更新isolated_state（模拟实际执行）
#             with torch.no_grad():
#                 hard_deployment = (torch.sigmoid(batch_deployed[i]) > 0.5).float()
#                 hard_update = (torch.sigmoid(batch_update[i]) > 0.5).float()
                
#                 # 更新isolated_state
#                 isolated_state = self._update_isolated_state_torch(
#                     isolated_state, 
#                     hard_deployment, 
#                     hard_update,
#                     current_time
#                 )
            
#             # 准备calculate_cost_with_gradients所需的参数
#             updated_age_tensor = torch.as_tensor(
#                 isolated_state['age_state'], 
#                 device=self.device, 
#                 dtype=torch.float32
#             ).view(self.N, self.M)
            
#             updated_accuracies_tensor = torch.as_tensor(
#                 isolated_state['accuracies'], 
#                 device=self.device, 
#                 dtype=torch.float32
#             ).view(self.N, self.M)
            
#             # 调用内循环的calculate_cost_with_gradients函数
#             try:
#                 cost, _, _, _, _, _, _ = self.meta_learner.calculate_cost_with_gradients(
#                     batch_deployed[i],           # deployed_surrogate (N, M)
#                     batch_update[i],             # update_surrogate (N, M)
#                     current_requests,            # requests_tensor (N, M)
#                     current_time,                # time_step (scalar)
#                     prev_dep_tensor,             # prev_deployment_tensor (N, M)
#                     updated_age_tensor,          # updated_age_tensor (N, M)
#                     updated_accuracies_tensor    # updated_accuracies_tensor (N, M)
#                 )
                
#                 total_cost = total_cost + cost
                
#                 # 更新prev_dep_tensor为当前硬部署决策
#                 prev_dep_tensor = hard_deployment.detach()
                    
#             except Exception as e:
#                 print(f"Warning: calculate_cost_with_gradients failed at step {i}: {e}")
#                 print(f"Shapes - deployed: {batch_deployed[i].shape}, requests: {current_requests.shape}, prev_dep: {prev_dep_tensor.shape}")
                
#                 # 回退到简化的损失计算
#                 fallback_loss = self.compute_fallback_adaptation_loss(
#                     batch_deployed[i], batch_update[i], current_requests
#                 )
#                 total_cost = total_cost + fallback_loss
                
#                 # 更新prev_dep_tensor
#                 prev_dep_tensor = hard_deployment.detach()
        
#         avg_cost = total_cost / batch_size
#         return avg_cost

#     def _update_isolated_state_torch(self, isolated_state, hard_deployment, hard_update, current_time):
#         """
#         更新isolated_state，模拟一个时隙的执行
        
#         Args:
#             isolated_state: 当前状态字典
#             hard_deployment: (N, M) 硬部署决策
#             hard_update: (N, M) 硬更新决策  
#             current_time: 当前时间步
#         """
#         # 转换为numpy进行状态更新
#         deployment_np = hard_deployment.cpu().numpy().astype(int)
#         update_np = hard_update.cpu().numpy().astype(int)
        
#         # 更新缓存状态
#         isolated_state['cache_state'] = deployment_np
        
#         # 更新年龄状态
#         # 年龄增加1，更新的模型年龄重置为0
#         isolated_state['age_state'] += 1
#         isolated_state['age_state'][update_np == 1] = 0
        
#         # 更新准确性（这里使用简化模型，实际可能需要更复杂的逻辑）
#         # 可以根据年龄衰减准确性
#         if hasattr(self.env, 'accuracy_decay_factor'):
#             decay_factor = self.env.accuracy_decay_factor
#             isolated_state['accuracies'] = np.maximum(
#                 isolated_state['accuracies'] * (decay_factor ** isolated_state['age_state']),
#                 0.1  # 最小准确性
#             )
        
#         return isolated_state

#     def compute_fallback_adaptation_loss(self, deployed_surrogate, update_surrogate, requests_tensor):
#         # 部署概率与请求匹配度
#         deploy_prob = torch.sigmoid(deployed_surrogate)
#         request_match_loss = torch.nn.functional.mse_loss(deploy_prob, requests_tensor)
        
#         # 容量约束惩罚
#         if hasattr(self.env, 'cloud_manager') and hasattr(self.env.cloud_manager, 'model_sizes'):
#             model_sizes = torch.tensor(self.env.cloud_manager.model_sizes, dtype=torch.float32, device=self.device)
#             capacity_usage = torch.sum(deploy_prob * model_sizes.unsqueeze(0), dim=-1)
#             capacity_penalty = torch.mean(torch.relu(capacity_usage - self.edge_cache_storage))
#         else:
#             capacity_penalty = torch.tensor(0.0, device=self.device)
        
#         # 更新成本
#         update_prob = torch.sigmoid(update_surrogate)
#         update_cost = torch.mean(update_prob) * 0.1
        
#         total_loss = request_match_loss + 0.1 * capacity_penalty + update_cost
#         return total_loss

#     def get_default_isolated_state(self):
#         """获取默认的isolated_state结构"""
#         return {
#             'cache_state': np.zeros((self.N, self.M)),
#             'age_state': np.zeros((self.N, self.M)),
#             'accuracies': np.ones((self.N, self.M))
#         }

#     def batch_process_adaptation_phase(self, time_steps, base_isolated_state):
#         """批量处理适应阶段，类似内循环的批处理机制"""
#         batch_size = len(time_steps)
#         device = self.device
        
#         # 预分配批量张量（与内循环保持一致）
#         if self.tensor_buffers:
#             batch_states = self.tensor_buffers.get_buffer('adapt_batch_states', (batch_size, self.calculate_input_dim()))
#             batch_requests = self.tensor_buffers.get_buffer('adapt_batch_requests', (batch_size, self.N, self.M))
#         else:
#             batch_states = torch.zeros(batch_size, self.calculate_input_dim(), device=device)
#             batch_requests = torch.zeros(batch_size, self.N, self.M, device=device)
        
#         # 批量获取状态和请求（与内循环保持一致）
#         isolated_state = copy.deepcopy(base_isolated_state)
        
#         if isolated_state['cache_state'].ndim == 1:
#             isolated_state['cache_state'] = isolated_state['cache_state'].reshape(self.N, self.M)
#         if isolated_state['age_state'].ndim == 1:
#             isolated_state['age_state'] = isolated_state['age_state'].reshape(self.N, self.M)
        
#         for i, t in enumerate(time_steps):
#             # 使用与内循环相同的状态获取机制
#             state_key = self._get_state_cache_key(t, isolated_state) if hasattr(self, '_get_state_cache_key') else f"{t}_{hash(str(isolated_state))}"
            
#             if state_key in self.state_cache:
#                 batch_states[i] = self.state_cache[state_key]
#             else:
#                 state = self._get_state_from_isolated(t, isolated_state) if hasattr(self, '_get_state_from_isolated') else torch.tensor(self.env.get_state_vector(t), device=device, dtype=torch.float32)
#                 batch_states[i] = state
#                 self.state_cache[state_key] = state.clone()
            
#             batch_requests[i] = self.get_requests_tensor(t) if hasattr(self, 'get_requests_tensor') else torch.tensor(self.env.get_requests(t), device=device, dtype=torch.float32)
        
#         # 执行批量适应
#         success = self.batch_adaptation_step(batch_states, batch_requests)
        
#         # 释放缓冲区
#         if self.tensor_buffers:
#             self.tensor_buffers.release_buffer('adapt_batch_states')
#             self.tensor_buffers.release_buffer('adapt_batch_requests')
        
#         return success

#     def get_network_decision_with_exploration(self, state_tensor, exploration_factor=0.1, use_current_params=False):
#         """使用网络输出获取决策，可选择性地添加探索噪声"""
#         with torch.no_grad():
#             # 选择使用的参数
#             if use_current_params and self.current_task_params is not None:
#                 if isinstance(self.current_task_params, dict) and hasattr(self.meta_learner, 'meta_policy_forward_with_weights'):
#                     batch_outputs = self.meta_learner.meta_policy_forward_with_weights(state_tensor, self.current_task_params)
#                 else:
#                     # 临时加载适应后的参数
#                     original_params = copy.deepcopy(self.meta_learner.meta_policy.state_dict())
#                     self.meta_learner.meta_policy.load_state_dict(self.current_task_params)
#                     batch_outputs = self.meta_learner.meta_policy(state_tensor)
#                     self.meta_learner.meta_policy.load_state_dict(original_params)
#             else:
#                 # 使用当前参数（元学习初始参数或部分适应的参数）
#                 batch_outputs = self.meta_learner.meta_policy(state_tensor)
            
#             mid_point = self.N * self.M
#             deploy_logits = batch_outputs[..., :mid_point].view(self.N, self.M)
#             update_logits = batch_outputs[..., mid_point:].view(self.N, self.M)
            
#             # 添加探索噪声（在适应阶段增加探索）
#             if exploration_factor > 0:
#                 deploy_logits = deploy_logits + torch.randn_like(deploy_logits) * exploration_factor
#                 update_logits = update_logits + torch.randn_like(update_logits) * exploration_factor
            
#             # 应用约束获取硬决策
#             deployed_surrogate, hard_deployment = self.meta_learner.enforce_storage_constraints(deploy_logits)
#             update_surrogate, hard_update = self.meta_learner.enforce_update_constraints(update_logits, deployed_surrogate)
            
#             return hard_deployment.cpu().numpy().astype(int), hard_update.cpu().numpy(), deployed_surrogate, update_surrogate

#     def optimize(self, t, requests, prev_deployment=None):
#         # 检查是否需要开始新任务
#         if self.current_task_id is None or t % self.args.task_length == 0:
#             self.start_new_task(t // self.args.task_length)
        
#         # 获取当前状态
#         state = self.env.get_state_vector(t)
#         # state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
#         state_tensor = state.to(dtype=torch.float32, device=self.device).unsqueeze(0)

#         request_tensor = torch.tensor(requests, dtype=torch.float32, device=self.device)
        
#         # 适应阶段：前K个时隙
#         if self.task_step_count < self.args.adaptation_slots:
#             # 使用网络输出进行决策，添加探索噪声以促进学习
#             exploration_factor = getattr(self.args, 'adaptation_exploration_factor', 0.2)
            
#             # 在适应阶段使用渐进式的探索减少
#             current_exploration = exploration_factor * (1.0 - self.task_step_count / self.args.adaptation_slots)
            
#             # 使用网络输出获取决策
#             deployment, update_mask, deployed_surrogate, update_surrogate = self.get_network_decision_with_exploration(
#                 state_tensor, 
#                 exploration_factor=current_exploration,
#                 use_current_params=(self.current_task_params is not None)
#             )
            
#             # 收集适应数据（包括网络输出的中间结果）
#             self.adaptation_buffer.append({
#                 'state': state_tensor.squeeze(0),
#                 'request': request_tensor,
#                 'time_step': t,
#                 'deployment': torch.tensor(deployment, dtype=torch.float32, device=self.device),
#                 'deployed_surrogate': deployed_surrogate.detach(),
#                 'update_surrogate': update_surrogate.detach()
#             })
            
#             # 渐进式适应：每隔几个时隙进行一次小规模适应
#             progressive_adapt_interval = getattr(self.args, 'progressive_adapt_interval', 3)
#             if (self.task_step_count + 1) % progressive_adapt_interval == 0 and len(self.adaptation_buffer) >= self.min_adaptation_samples:
#                 # 使用最近的数据进行渐进式适应
#                 recent_data = list(self.adaptation_buffer)[-self.min_adaptation_samples:]
#                 batch_states = [item['state'] for item in recent_data]
#                 batch_requests = [item['request'] for item in recent_data]  # 每个request应该是(N, M)
#                 time_steps = [item['time_step'] for item in recent_data]
                
#                 # 执行渐进式适应（较小的学习率）
#                 progressive_lr = self.adaptation_lr * 0.5
#                 original_lr = self.adaptation_lr
#                 self.adaptation_lr = progressive_lr
                
#                 self.batch_adaptation_step(
#                     batch_states, batch_requests,
#                     time_steps=time_steps, base_isolated_state=self.get_default_isolated_state()
#                 )
                
#                 self.adaptation_lr = original_lr
#                 print(f"Progressive adaptation at step {self.task_step_count}")
            
#             # 在适应阶段的最后一个时隙执行最终适应
#             if self.task_step_count == self.args.adaptation_slots - 1:
#                 # 准备批量适应数据
#                 adaptation_data = list(self.adaptation_buffer)
#                 batch_states = [item['state'] for item in adaptation_data]
#                 batch_requests = [item['request'] for item in adaptation_data]  # 每个request应该是(N, M)
#                 time_steps = [item['time_step'] for item in adaptation_data]
                
#                 # 执行最终批量适应
#                 adaptation_success = self.batch_adaptation_step(
#                     batch_states, batch_requests,
#                     time_steps=time_steps, base_isolated_state=self.get_default_isolated_state()
#                 )
#                 if not adaptation_success:
#                     print("Warning: Final batch adaptation failed, using current parameters")
#                 else:
#                     print(f"Final adaptation completed for task {self.current_task_id}")
            
#             # 返回当前决策
#             update_mask = update_mask
            
#         else:
#             # 推理阶段：使用适应后的参数进行批量处理
#             if self.current_task_params is not None:
#                 # 如果有fast weights，使用它们
#                 if isinstance(self.current_task_params, dict) and hasattr(self.meta_learner, 'meta_policy_forward_with_weights'):
#                     batch_outputs = self.meta_learner.meta_policy_forward_with_weights(state_tensor, self.current_task_params)
#                 else:
#                     self.meta_learner.meta_policy.load_state_dict(self.current_task_params)
#                     batch_outputs = self.meta_learner.meta_policy(state_tensor)
#             else:
#                 batch_outputs = self.meta_learner.meta_policy(state_tensor)
            
#             mid_point = self.N * self.M
#             deploy_logits = batch_outputs[..., :mid_point].view(self.N, self.M)
#             update_logits = batch_outputs[..., mid_point:].view(self.N, self.M)
            
#             # 应用约束（与内循环保持一致）
#             deployed_surrogate, hard_deployment = self.meta_learner.enforce_storage_constraints(deploy_logits)
#             update_surrogate, hard_update = self.meta_learner.enforce_update_constraints(update_logits, deployed_surrogate)
            
#             deployment = hard_deployment.cpu().numpy().astype(int)
#             update_mask = hard_update.cpu().numpy()
            
#             # 在线适应（如果启用）
#             if (hasattr(self.args, 'online_adaptation') and self.args.online_adaptation and 
#                 self.task_step_count >= self.args.adaptation_slots):
                
#                 # 添加到缓冲区
#                 self.adaptation_buffer.append({
#                     'state': state_tensor.squeeze(0),
#                     'request': request_tensor,
#                     'time_step': t
#                 })
                
#                 # 定期在线适应
#                 online_batch_size = getattr(self.args, 'online_batch_size', 10)
#                 if len(self.adaptation_buffer) >= online_batch_size:
#                     recent_data = list(self.adaptation_buffer)[-online_batch_size:]
#                     batch_states = [item['state'] for item in recent_data]
#                     batch_requests = [item['request'] for item in recent_data]
#                     time_steps = [item['time_step'] for item in recent_data]
                    
#                     # 创建isolated_states
#                     isolated_states = [self.get_default_isolated_state() for _ in range(len(recent_data))]
                    
#                     self.batch_adaptation_step(
#                         batch_states, batch_requests,
#                         time_steps=time_steps, isolated_states=isolated_states
#                     )
                    
#                     # 清空部分缓冲区
#                     self.adaptation_buffer.clear()
#                     self.adaptation_buffer.extend(recent_data[-online_batch_size//2:])
        
#         # 更新时隙计数
#         self.task_step_count += 1
        
#         return deployment, update_mask

#     def _get_state_cache_key(self, t, isolated_state):
#         """生成状态缓存键"""
#         cache_key = f"{t}_{hash(isolated_state['cache_state'].tobytes() + isolated_state['age_state'].tobytes())}"
#         return cache_key

#     def _get_state_from_isolated(self, t, isolated_state):
#         """从隔离状态获取状态向量"""
#         state_vector = self.env.get_state_vector(t)  # 假设环境有此方法
#         return torch.tensor(state_vector, dtype=torch.float32, device=self.device)

#     def get_requests_tensor(self, t):
#         """获取请求张量"""
#         requests = self.env.get_requests(t)  # 假设环境有此方法
#         return torch.tensor(requests, dtype=torch.float32, device=self.device)


import torch
import numpy as np
import copy
from collections import deque

class MetaOptimization:
    def __init__(self, env, args, meta_learner):
        self.env = env
        self.args = args
        self.meta_learner = meta_learner
        self.N = env.NUM_EDGE_NODES
        self.M = env.NUM_MODELS  # 使用M而不是NUM_MODELS，与内循环保持一致
        self.edge_cache_storage = args.edge_cache_storage
        self.NUM_SHARED_MODELS = env.NUM_SHARED_MODELS
        self.NUM_SPECIFIC_MODELS = env.NUM_SPECIFIC_MODELS
        
        # 添加Shared_flag标识
        self.Shared_flag = getattr(args, 'Shared_flag', True)
        
        # 获取依赖矩阵
        self.dependency_matrix = np.array(env.cloud_manager.get_dependency_matrix(), dtype=float)
        
        # 获取模型大小
        self.model_sizes = np.array(env.cloud_manager.model_sizes, dtype=float)
        
        # 计算有效模型大小
        self.effective_model_sizes = self._compute_effective_model_sizes()
        
        # 保存元学习的初始参数
        self.meta_initial_params = copy.deepcopy(self.meta_learner.meta_policy.state_dict())
        
        # 适应阶段参数
        self.adaptation_steps = args.adaptation_steps
        self.adaptation_lr = args.adaptation_lr
        self.current_task_params = None
        self.task_step_count = 0
        self.current_task_id = None
        
        # 改进的缓冲区管理
        self.adaptation_buffer = deque(maxlen=getattr(args, 'buffer_size', 100))
        self.min_adaptation_samples = getattr(args, 'min_adaptation_samples', 10)
        
        # 渐进式适应
        self.progressive_params = None
        self.adaptation_momentum = getattr(args, 'adaptation_momentum', 0.9)
        
        # 添加设备和缓冲区管理（与内循环保持一致）
        self.device = meta_learner.device if hasattr(meta_learner, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tensor_buffers = meta_learner.tensor_buffers if hasattr(meta_learner, 'tensor_buffers') else None
        
        # 状态缓存（与内循环保持一致）
        self.state_cache = {}

    def _compute_effective_model_sizes(self):
        """计算有效模型大小（考虑共享标识）"""
        effective_sizes = self.model_sizes.copy()
        
        if not self.Shared_flag:
            # 不考虑共享时，特定模型大小需要加上依赖的共享模型大小
            for specific_idx in range(self.NUM_SPECIFIC_MODELS):
                model_idx = specific_idx + self.NUM_SHARED_MODELS
                shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
                for dep_idx in shared_deps:
                    effective_sizes[model_idx] += self.model_sizes[dep_idx]
        
        return effective_sizes

    def calculate_input_dim(self):
        """计算输入维度，与内循环保持一致"""
        return self.meta_learner.calculate_input_dim() if hasattr(self.meta_learner, 'calculate_input_dim') else self.N * self.M * 2

    def start_new_task(self, task_id):
        """开始一个新任务，重置模型参数和计数器"""
        print(f"Starting new task {task_id}")
        self.current_task_id = task_id
        self.task_step_count = 0
        self.adaptation_buffer.clear()
        
        # 清空状态缓存
        self.state_cache.clear()
        
        # 重置为元学习初始参数
        self.meta_learner.meta_policy.load_state_dict(self.meta_initial_params)
        self.current_task_params = None
        self.progressive_params = copy.deepcopy(self.meta_initial_params)

    def batch_adaptation_step(self, batch_states, batch_requests, fast_weights=None, exploration_factor=0.1, time_steps=None, base_isolated_state=None):
        if len(batch_states) == 0:
            return False
        
        batch_size = len(batch_states)
        device = self.device
        
        # 准备批量数据
        states_tensor = torch.stack(batch_states) if isinstance(batch_states[0], torch.Tensor) else torch.tensor(batch_states, device=device, dtype=torch.float32)
        
        # 确保requests是正确格式 - 每个请求应该是(N, M)
        if isinstance(batch_requests[0], torch.Tensor):
            requests_tensor = torch.stack(batch_requests)
        else:
            requests_tensor = torch.tensor(batch_requests, device=device, dtype=torch.float32)
        
        # 验证请求张量的形状
        if requests_tensor.shape[-2:] != (self.N, self.M):
            print(f"Warning: Unexpected requests shape {requests_tensor.shape}, expected (..., {self.N}, {self.M})")
            # 尝试reshape或扩展
            if requests_tensor.dim() == 2 and requests_tensor.shape[-1] == self.M:
                # 假设是单节点情况，扩展到所有节点
                requests_tensor = requests_tensor.unsqueeze(-2).expand(-1, self.N, -1)
            elif requests_tensor.dim() == 1:
                # 单个请求向量，扩展到所有节点
                requests_tensor = requests_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, self.N, -1)
        
        # 使用当前权重或fast_weights
        current_weights = fast_weights if fast_weights is not None else None
        
        # 创建优化器（如果不使用fast weights）
        optimizer = None
        if current_weights is None:
            optimizer = torch.optim.Adam(self.meta_learner.meta_policy.parameters(), lr=self.adaptation_lr)
        
        adaptation_losses = []
        
        for step in range(self.adaptation_steps):
            if optimizer:
                optimizer.zero_grad()
            
            # 前向传播（类似内循环的批处理）
            if current_weights is not None:
                batch_outputs = self.meta_learner.meta_policy_forward_with_weights(states_tensor, current_weights)
            else:
                batch_outputs = self.meta_learner.meta_policy(states_tensor)
            
            mid_point = self.N * self.M
            batch_deploy_logits = batch_outputs[..., :mid_point].view(batch_size, self.N, self.M)
            batch_update_logits = batch_outputs[..., mid_point:].view(batch_size, self.N, self.M)
            
            # 添加探索噪声（与内循环保持一致）
            if exploration_factor > 0:
                batch_deploy_logits = batch_deploy_logits + torch.randn_like(batch_deploy_logits) * exploration_factor
                batch_update_logits = batch_update_logits + torch.randn_like(batch_update_logits) * exploration_factor
            
            # 批量约束实施（与内循环保持一致）
            batch_deployed_surrogate = []
            batch_update_surrogate = []
            
            for i in range(batch_size):
                deployed_surr, _ = self.meta_learner.enforce_storage_constraints(batch_deploy_logits[i])
                update_surr, _ = self.meta_learner.enforce_update_constraints(batch_update_logits[i], deployed_surr)
                batch_deployed_surrogate.append(deployed_surr)
                batch_update_surrogate.append(update_surr)
            
            batch_deployed_surrogate = torch.stack(batch_deployed_surrogate)
            batch_update_surrogate = torch.stack(batch_update_surrogate)
            
            # 计算适应损失（使用内循环的cost函数）
            loss = self.compute_batch_adaptation_loss(
                batch_deployed_surrogate, 
                batch_update_surrogate,
                requests_tensor, 
                time_steps,
                base_isolated_state
            )
            
            adaptation_losses.append(loss.item())
            
            # 反向传播和优化
            if current_weights is None:
                # 直接优化模型参数
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.meta_learner.meta_policy.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                # 使用fast weights机制
                grads = torch.autograd.grad(loss, current_weights.values(), create_graph=True, allow_unused=True)
                # 过滤None梯度
                valid_grads = []
                valid_params = []
                for (name, param), grad in zip(current_weights.items(), grads):
                    if grad is not None:
                        valid_grads.append(grad)
                        valid_params.append((name, param))
                
                # 更新有效参数
                for (name, param), grad in zip(valid_params, valid_grads):
                    current_weights[name] = param - self.adaptation_lr * grad
        
        # 保存适应后的参数
        if current_weights is not None:
            self.current_task_params = copy.deepcopy(current_weights)
        else:
            self.current_task_params = copy.deepcopy(self.meta_learner.meta_policy.state_dict())
        
        # print(f"Batch adaptation completed. Average loss: {np.mean(adaptation_losses):.4f}")
        return True

    def compute_batch_adaptation_loss(self, batch_deployed, batch_update, batch_requests, time_steps=None, base_isolated_state=None):
        batch_size = batch_deployed.shape[0]
        total_cost = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 使用基础状态或创建默认状态
        if base_isolated_state is None:
            isolated_state = self.get_default_isolated_state()
        else:
            isolated_state = copy.deepcopy(base_isolated_state)
        
        # 确保状态格式正确
        if isolated_state['cache_state'].ndim == 1:
            isolated_state['cache_state'] = isolated_state['cache_state'].reshape(self.N, self.M)
        if isolated_state['age_state'].ndim == 1:
            isolated_state['age_state'] = isolated_state['age_state'].reshape(self.N, self.M)
        
        # 初始化前一部署状态
        prev_dep_tensor = torch.as_tensor(
            isolated_state['cache_state'], 
            device=self.device, 
            dtype=torch.float32
        ).view(self.N, self.M)
        
        # 按时隙顺序处理（与内循环保持一致）
        for i in range(batch_size):
            # 获取当前时隙信息
            current_time = time_steps[i] if time_steps and i < len(time_steps) else i
            
            # 当前时隙的请求应该是 (N, M) - 所有节点的请求
            current_requests = batch_requests[i]  # shape: (N, M)
            if current_requests.dim() == 1:
                # 如果只有一维，假设是单节点情况，需要扩展
                current_requests = current_requests.unsqueeze(0).expand(self.N, -1)
            
            # 使用硬决策更新isolated_state（模拟实际执行）
            with torch.no_grad():
                hard_deployment = (torch.sigmoid(batch_deployed[i]) > 0.5).float()
                hard_update = (torch.sigmoid(batch_update[i]) > 0.5).float()
                
                # 如果考虑共享，需要处理依赖关系
                if self.Shared_flag:
                    hard_deployment, hard_update = self._enforce_dependency_constraints(
                        hard_deployment, hard_update
                    )
                
                # 更新isolated_state
                isolated_state = self._update_isolated_state_torch(
                    isolated_state, 
                    hard_deployment, 
                    hard_update,
                    current_time
                )
            
            # 准备calculate_cost_with_gradients所需的参数
            updated_age_tensor = torch.as_tensor(
                isolated_state['age_state'], 
                device=self.device, 
                dtype=torch.float32
            ).view(self.N, self.M)
            
            updated_accuracies_tensor = torch.as_tensor(
                isolated_state['accuracies'], 
                device=self.device, 
                dtype=torch.float32
            ).view(self.N, self.M)
            
            # 调用内循环的calculate_cost_with_gradients函数
            try:
                cost, _, _, _, _, _, _ = self.meta_learner.calculate_cost_with_gradients(
                    batch_deployed[i],           # deployed_surrogate (N, M)
                    batch_update[i],             # update_surrogate (N, M)
                    current_requests,            # requests_tensor (N, M)
                    current_time,                # time_step (scalar)
                    prev_dep_tensor,             # prev_deployment_tensor (N, M)
                    updated_age_tensor,          # updated_age_tensor (N, M)
                    updated_accuracies_tensor    # updated_accuracies_tensor (N, M)
                )
                
                total_cost = total_cost + cost
                
                # 更新prev_dep_tensor为当前硬部署决策
                prev_dep_tensor = hard_deployment.detach()
                    
            except Exception as e:
                print(f"Warning: calculate_cost_with_gradients failed at step {i}: {e}")
                print(f"Shapes - deployed: {batch_deployed[i].shape}, requests: {current_requests.shape}, prev_dep: {prev_dep_tensor.shape}")
                
                # 回退到简化的损失计算
                fallback_loss = self.compute_fallback_adaptation_loss(
                    batch_deployed[i], batch_update[i], current_requests
                )
                total_cost = total_cost + fallback_loss
                
                # 更新prev_dep_tensor
                prev_dep_tensor = hard_deployment.detach()
        
        avg_cost = total_cost / batch_size
        return avg_cost

    def _enforce_dependency_constraints(self, hard_deployment, hard_update):
        """
        强制执行依赖约束
        只在Shared_flag=True时需要
        """
        if not self.Shared_flag:
            return hard_deployment, hard_update
        
        deployment_np = hard_deployment.cpu().numpy()
        update_np = hard_update.cpu().numpy()
        
        # 确保部署特定模型时，依赖的共享模型也被部署
        for node in range(self.N):
            for specific_idx in range(self.NUM_SPECIFIC_MODELS):
                model_idx = specific_idx + self.NUM_SHARED_MODELS
                if deployment_np[node, model_idx] == 1:
                    # 获取依赖的共享模型
                    shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
                    for dep_idx in shared_deps:
                        deployment_np[node, dep_idx] = 1
            
            # 确保更新特定模型时，依赖的共享模型也被更新
            for specific_idx in range(self.NUM_SPECIFIC_MODELS):
                model_idx = specific_idx + self.NUM_SHARED_MODELS
                if update_np[node, model_idx] == 1:
                    shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
                    for dep_idx in shared_deps:
                        if deployment_np[node, dep_idx] == 1:  # 只有已部署的才更新
                            update_np[node, dep_idx] = 1
        
        return (torch.tensor(deployment_np, device=self.device, dtype=torch.float32),
                torch.tensor(update_np, device=self.device, dtype=torch.float32))

    def _update_isolated_state_torch(self, isolated_state, hard_deployment, hard_update, current_time):
        """
        更新isolated_state，模拟一个时隙的执行
        
        Args:
            isolated_state: 当前状态字典
            hard_deployment: (N, M) 硬部署决策
            hard_update: (N, M) 硬更新决策  
            current_time: 当前时间步
        """
        # 转换为numpy进行状态更新
        deployment_np = hard_deployment.cpu().numpy().astype(int)
        update_np = hard_update.cpu().numpy().astype(int)
        
        # 更新缓存状态
        isolated_state['cache_state'] = deployment_np
        
        # 如果不考虑共享，共享模型的状态应该为0
        if not self.Shared_flag:
            isolated_state['cache_state'][:, :self.NUM_SHARED_MODELS] = 0
        
        # 更新年龄状态
        # 年龄增加1，更新的模型年龄重置为0
        isolated_state['age_state'] += 1
        isolated_state['age_state'][update_np == 1] = 0
        
        # 更新准确性（这里使用简化模型，实际可能需要更复杂的逻辑）
        # 可以根据年龄衰减准确性
        if hasattr(self.env, 'accuracy_decay_factor'):
            decay_factor = self.env.accuracy_decay_factor
            isolated_state['accuracies'] = np.maximum(
                isolated_state['accuracies'] * (decay_factor ** isolated_state['age_state']),
                0.1  # 最小准确性
            )
        
        return isolated_state

    def compute_fallback_adaptation_loss(self, deployed_surrogate, update_surrogate, requests_tensor):
        # 部署概率与请求匹配度
        deploy_prob = torch.sigmoid(deployed_surrogate)
        request_match_loss = torch.nn.functional.mse_loss(deploy_prob, requests_tensor)
        
        # 容量约束惩罚 - 使用有效模型大小
        if self.Shared_flag:
            model_sizes_tensor = torch.tensor(self.model_sizes, dtype=torch.float32, device=self.device)
        else:
            model_sizes_tensor = torch.tensor(self.effective_model_sizes, dtype=torch.float32, device=self.device)
        
        capacity_usage = torch.sum(deploy_prob * model_sizes_tensor.unsqueeze(0), dim=-1)
        capacity_penalty = torch.mean(torch.relu(capacity_usage - self.edge_cache_storage))
        
        # 更新成本
        update_prob = torch.sigmoid(update_surrogate)
        update_cost = torch.mean(update_prob) * 0.1
        
        total_loss = request_match_loss + 0.1 * capacity_penalty + update_cost
        return total_loss

    def get_default_isolated_state(self):
        """获取默认的isolated_state结构"""
        cache_state = np.zeros((self.N, self.M))
        # 如果不考虑共享，共享模型状态为0
        if not self.Shared_flag:
            cache_state[:, :self.NUM_SHARED_MODELS] = 0
        
        return {
            'cache_state': cache_state,
            'age_state': np.zeros((self.N, self.M)),
            'accuracies': np.ones((self.N, self.M))
        }

    def batch_process_adaptation_phase(self, time_steps, base_isolated_state):
        """批量处理适应阶段，类似内循环的批处理机制"""
        batch_size = len(time_steps)
        device = self.device
        
        # 预分配批量张量（与内循环保持一致）
        if self.tensor_buffers:
            batch_states = self.tensor_buffers.get_buffer('adapt_batch_states', (batch_size, self.calculate_input_dim()))
            batch_requests = self.tensor_buffers.get_buffer('adapt_batch_requests', (batch_size, self.N, self.M))
        else:
            batch_states = torch.zeros(batch_size, self.calculate_input_dim(), device=device)
            batch_requests = torch.zeros(batch_size, self.N, self.M, device=device)
        
        # 批量获取状态和请求（与内循环保持一致）
        isolated_state = copy.deepcopy(base_isolated_state)
        
        if isolated_state['cache_state'].ndim == 1:
            isolated_state['cache_state'] = isolated_state['cache_state'].reshape(self.N, self.M)
        if isolated_state['age_state'].ndim == 1:
            isolated_state['age_state'] = isolated_state['age_state'].reshape(self.N, self.M)
        
        for i, t in enumerate(time_steps):
            # 使用与内循环相同的状态获取机制
            state_key = self._get_state_cache_key(t, isolated_state) if hasattr(self, '_get_state_cache_key') else f"{t}_{hash(str(isolated_state))}"
            
            if state_key in self.state_cache:
                batch_states[i] = self.state_cache[state_key]
            else:
                state = self._get_state_from_isolated(t, isolated_state) if hasattr(self, '_get_state_from_isolated') else torch.tensor(self.env.get_state_vector(t), device=device, dtype=torch.float32)
                batch_states[i] = state
                self.state_cache[state_key] = state.clone()
            
            batch_requests[i] = self.get_requests_tensor(t) if hasattr(self, 'get_requests_tensor') else torch.tensor(self.env.get_requests(t), device=device, dtype=torch.float32)
        
        # 执行批量适应
        success = self.batch_adaptation_step(batch_states, batch_requests)
        
        # 释放缓冲区
        if self.tensor_buffers:
            self.tensor_buffers.release_buffer('adapt_batch_states')
            self.tensor_buffers.release_buffer('adapt_batch_requests')
        
        return success

    def get_network_decision_with_exploration(self, state_tensor, exploration_factor=0.1, use_current_params=False):
        """使用网络输出获取决策，可选择性地添加探索噪声"""
        with torch.no_grad():
            # 选择使用的参数
            if use_current_params and self.current_task_params is not None:
                if isinstance(self.current_task_params, dict) and hasattr(self.meta_learner, 'meta_policy_forward_with_weights'):
                    batch_outputs = self.meta_learner.meta_policy_forward_with_weights(state_tensor, self.current_task_params)
                else:
                    # 临时加载适应后的参数
                    original_params = copy.deepcopy(self.meta_learner.meta_policy.state_dict())
                    self.meta_learner.meta_policy.load_state_dict(self.current_task_params)
                    batch_outputs = self.meta_learner.meta_policy(state_tensor)
                    self.meta_learner.meta_policy.load_state_dict(original_params)
            else:
                # 使用当前参数（元学习初始参数或部分适应的参数）
                batch_outputs = self.meta_learner.meta_policy(state_tensor)
            
            mid_point = self.N * self.M
            deploy_logits = batch_outputs[..., :mid_point].view(self.N, self.M)
            update_logits = batch_outputs[..., mid_point:].view(self.N, self.M)
            
            # 添加探索噪声（在适应阶段增加探索）
            if exploration_factor > 0:
                deploy_logits = deploy_logits + torch.randn_like(deploy_logits) * exploration_factor
                update_logits = update_logits + torch.randn_like(update_logits) * exploration_factor
            
            # 应用约束获取硬决策
            deployed_surrogate, hard_deployment = self.meta_learner.enforce_storage_constraints(deploy_logits)
            update_surrogate, hard_update = self.meta_learner.enforce_update_constraints(update_logits, deployed_surrogate)
            
            # 如果考虑共享，强制执行依赖约束
            if self.Shared_flag:
                hard_deployment, hard_update = self._enforce_dependency_constraints(
                    hard_deployment, hard_update
                )
            else:
                # 不考虑共享时，确保共享模型状态为0
                hard_deployment[:, :self.NUM_SHARED_MODELS] = 0
                hard_update[:, :self.NUM_SHARED_MODELS] = 0
            
            return hard_deployment.cpu().numpy().astype(int), hard_update.cpu().numpy(), deployed_surrogate, update_surrogate

    def optimize(self, t, requests, prev_deployment=None):
        # 检查是否需要开始新任务
        if self.current_task_id is None or t % self.args.task_length == 0:
            self.start_new_task(t // self.args.task_length)
        
        # 获取当前状态
        state = self.env.get_state_vector(t)
        # state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_tensor = state.to(dtype=torch.float32, device=self.device).unsqueeze(0)

        request_tensor = torch.tensor(requests, dtype=torch.float32, device=self.device)
        
        # 适应阶段：前K个时隙
        if self.task_step_count < self.args.adaptation_slots:
            # 使用网络输出进行决策，添加探索噪声以促进学习
            exploration_factor = getattr(self.args, 'adaptation_exploration_factor', 0.2)
            
            # 在适应阶段使用渐进式的探索减少
            current_exploration = exploration_factor * (1.0 - self.task_step_count / self.args.adaptation_slots)
            
            # 使用网络输出获取决策
            deployment, update_mask, deployed_surrogate, update_surrogate = self.get_network_decision_with_exploration(
                state_tensor, 
                exploration_factor=current_exploration,
                use_current_params=(self.current_task_params is not None)
            )
            
            # 收集适应数据（包括网络输出的中间结果）
            self.adaptation_buffer.append({
                'state': state_tensor.squeeze(0),
                'request': request_tensor,
                'time_step': t,
                'deployment': torch.tensor(deployment, dtype=torch.float32, device=self.device),
                'deployed_surrogate': deployed_surrogate.detach(),
                'update_surrogate': update_surrogate.detach()
            })
            
            # 渐进式适应：每隔几个时隙进行一次小规模适应
            progressive_adapt_interval = getattr(self.args, 'progressive_adapt_interval', 3)
            if (self.task_step_count + 1) % progressive_adapt_interval == 0 and len(self.adaptation_buffer) >= self.min_adaptation_samples:
                # 使用最近的数据进行渐进式适应
                recent_data = list(self.adaptation_buffer)[-self.min_adaptation_samples:]
                batch_states = [item['state'] for item in recent_data]
                batch_requests = [item['request'] for item in recent_data]  # 每个request应该是(N, M)
                time_steps = [item['time_step'] for item in recent_data]
                
                # 执行渐进式适应（较小的学习率）
                progressive_lr = self.adaptation_lr * 0.5
                original_lr = self.adaptation_lr
                self.adaptation_lr = progressive_lr
                
                self.batch_adaptation_step(
                    batch_states, batch_requests,
                    time_steps=time_steps, base_isolated_state=self.get_default_isolated_state()
                )
                
                self.adaptation_lr = original_lr
                print(f"Progressive adaptation at step {self.task_step_count}")
            
            # 在适应阶段的最后一个时隙执行最终适应
            if self.task_step_count == self.args.adaptation_slots - 1:
                # 准备批量适应数据
                adaptation_data = list(self.adaptation_buffer)
                batch_states = [item['state'] for item in adaptation_data]
                batch_requests = [item['request'] for item in adaptation_data]  # 每个request应该是(N, M)
                time_steps = [item['time_step'] for item in adaptation_data]
                
                # 执行最终批量适应
                adaptation_success = self.batch_adaptation_step(
                    batch_states, batch_requests,
                    time_steps=time_steps, base_isolated_state=self.get_default_isolated_state()
                )
                if not adaptation_success:
                    print("Warning: Final batch adaptation failed, using current parameters")
                else:
                    print(f"Final adaptation completed for task {self.current_task_id}")
            
            # 返回当前决策
            update_mask = update_mask
            
        else:
            # 推理阶段：使用适应后的参数进行批量处理
            if self.current_task_params is not None:
                # 如果有fast weights，使用它们
                if isinstance(self.current_task_params, dict) and hasattr(self.meta_learner, 'meta_policy_forward_with_weights'):
                    batch_outputs = self.meta_learner.meta_policy_forward_with_weights(state_tensor, self.current_task_params)
                else:
                    self.meta_learner.meta_policy.load_state_dict(self.current_task_params)
                    batch_outputs = self.meta_learner.meta_policy(state_tensor)
            else:
                batch_outputs = self.meta_learner.meta_policy(state_tensor)
            
            mid_point = self.N * self.M
            deploy_logits = batch_outputs[..., :mid_point].view(self.N, self.M)
            update_logits = batch_outputs[..., mid_point:].view(self.N, self.M)
            
            # 应用约束（与内循环保持一致）
            deployed_surrogate, hard_deployment = self.meta_learner.enforce_storage_constraints(deploy_logits)
            update_surrogate, hard_update = self.meta_learner.enforce_update_constraints(update_logits, deployed_surrogate)
            
            # 如果考虑共享，强制执行依赖约束
            if self.Shared_flag:
                hard_deployment, hard_update = self._enforce_dependency_constraints(
                    hard_deployment, hard_update
                )
            else:
                # 不考虑共享时，确保共享模型状态为0
                hard_deployment[:, :self.NUM_SHARED_MODELS] = 0
                hard_update[:, :self.NUM_SHARED_MODELS] = 0
            
            deployment = hard_deployment.cpu().numpy().astype(int)
            update_mask = hard_update.cpu().numpy()
            
            # 在线适应（如果启用）
            if (hasattr(self.args, 'online_adaptation') and self.args.online_adaptation and 
                self.task_step_count >= self.args.adaptation_slots):
                
                # 添加到缓冲区
                self.adaptation_buffer.append({
                    'state': state_tensor.squeeze(0),
                    'request': request_tensor,
                    'time_step': t
                })
                
                # 定期在线适应
                online_batch_size = getattr(self.args, 'online_batch_size', 10)
                if len(self.adaptation_buffer) >= online_batch_size:
                    recent_data = list(self.adaptation_buffer)[-online_batch_size:]
                    batch_states = [item['state'] for item in recent_data]
                    batch_requests = [item['request'] for item in recent_data]
                    time_steps = [item['time_step'] for item in recent_data]
                    
                    # 创建isolated_states
                    isolated_states = [self.get_default_isolated_state() for _ in range(len(recent_data))]
                    
                    self.batch_adaptation_step(
                        batch_states, batch_requests,
                        time_steps=time_steps, isolated_states=isolated_states
                    )
                    
                    # 清空部分缓冲区
                    self.adaptation_buffer.clear()
                    self.adaptation_buffer.extend(recent_data[-online_batch_size//2:])
        
        # 更新时隙计数
        self.task_step_count += 1
        
        return deployment, update_mask

    def _get_state_cache_key(self, t, isolated_state):
        """生成状态缓存键"""
        cache_key = f"{t}_{hash(isolated_state['cache_state'].tobytes() + isolated_state['age_state'].tobytes())}"
        return cache_key

    def _get_state_from_isolated(self, t, isolated_state):
        """从隔离状态获取状态向量"""
        state_vector = self.env.get_state_vector(t)  # 假设环境有此方法
        return torch.tensor(state_vector, dtype=torch.float32, device=self.device)

    def get_requests_tensor(self, t):
        """获取请求张量"""
        requests = self.env.get_requests(t)  # 假设环境有此方法
        return torch.tensor(requests, dtype=torch.float32, device=self.device)