import numpy as np
from scipy.optimize import minimize
import numba
from numba import jit, prange
from parameters import args_parser
import warnings
warnings.filterwarnings('ignore', category=numba.NumbaPerformanceWarning)

args = args_parser()

@jit(nopython=True, cache=True)
def fast_sigmoid(x):
    """快速sigmoid函数"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

@jit(nopython=True, cache=True, parallel=True)
def fast_capacity_constraint_check(config, model_sizes, capacity):
    """快速检查容量约束"""
    num_nodes = config.shape[0]
    results = np.zeros(num_nodes, dtype=numba.boolean)
    for i in prange(num_nodes):
        total_size = 0.0
        for j in range(config.shape[1]):
            total_size += config[i, j] * model_sizes[j]
        results[i] = total_size <= capacity
    return results

@jit(nopython=True, cache=True, parallel=True)
def fast_project_capacity_constraint(config, model_sizes, capacity):
    """快速投影到容量约束集合"""
    num_nodes, num_models = config.shape
    result = config.copy()
    
    for i in prange(num_nodes):
        total_size = 0.0
        for j in range(num_models):
            total_size += config[i, j] * model_sizes[j]
        
        if total_size > capacity:
            scale = capacity / total_size
            for j in range(num_models):
                result[i, j] = config[i, j] * scale
    
    return result

@jit(nopython=True, cache=True, parallel=True)
def fast_continuous_to_binary(continuous_config, model_sizes, capacity):
    """快速连续到二值转换"""
    num_nodes, num_models = continuous_config.shape
    binary_config = np.zeros_like(continuous_config)
    
    for i in prange(num_nodes):
        # 计算价值
        values = np.zeros(num_models)
        for j in range(num_models):
            if model_sizes[j] > 0:
                values[j] = continuous_config[i, j] / model_sizes[j]
            else:
                values[j] = 0.0
        
        # 贪心选择
        current_size = 0.0
        selected = np.zeros(num_models, dtype=numba.boolean)
        
        for _ in range(num_models):
            best_idx = -1
            best_value = -1.0
            
            for j in range(num_models):
                if not selected[j] and values[j] > best_value:
                    if current_size + model_sizes[j] <= capacity:
                        best_value = values[j]
                        best_idx = j
            
            if best_idx >= 0:
                binary_config[i, best_idx] = 1.0
                current_size += model_sizes[best_idx]
                selected[best_idx] = True
            else:
                break
    
    return binary_config

@jit(nopython=True, cache=True)
def fast_norm_calculation(x, y):
    """快速计算L2范数"""
    diff = x - y
    return np.sqrt(np.sum(diff * diff))

class MetaOCO:
    def __init__(self, env, all_requests, request_records, num_specific, num_shared, NUM_MODELS, args):
        self.env = env
        self.all_requests = all_requests
        self.request_records = request_records
        self.num_specific = num_specific
        self.num_shared = num_shared
        self.dependency_matrix = self.env.cloud_manager.dependency_matrix
        self.edge_cache_storage = args.edge_cache_storage
        self.task_length = args.task_length
        self.num_models = NUM_MODELS
        self.num_edge_nodes = args.edge_num
        self.args = args
        
        # 添加Shared_flag标识
        self.Shared_flag = getattr(args, 'Shared_flag', True)
        
        # 获取模型大小并转换为numpy数组以支持numba
        self.model_sizes = np.array(env.cloud_manager.model_sizes, dtype=np.float64)
        
        # 计算有效模型大小（考虑共享标识）
        self.effective_model_sizes = self._compute_effective_model_sizes()
        
        # 初始化元配置（均匀分布）- 论文中的Φ
        self.Phi = np.zeros((self.num_edge_nodes, self.num_models), dtype=np.float64)
        
        # 任务相似度距离D* - 论文中的D*
        self.D_star = 1.0
        
        # 当前任务计数器
        self.current_task_step = 0
        self.current_task_id = 0
        
        # 当前缓存配置
        self.current_cache_config = self.Phi.copy()
        
        # 记录当前任务的请求
        self.current_task_requests = []
        
        # 权重矩阵 - 预计算
        self.weights = np.ones((self.num_edge_nodes, self.num_edge_nodes), dtype=np.float64)
        self.d_max = self.num_edge_nodes
        self.w_inf = np.max(self.weights)
        
        # 存储上一次的部署配置
        self.prev_deployment = None
        
        # 获取依赖矩阵
        self.dependency_matrix = np.array(self.env.cloud_manager.get_dependency_matrix(), dtype=np.float64)
        
        # 预计算常用值
        self.eta_base = self.D_star / (np.sqrt(self.task_length) * self.w_inf * np.sqrt(self.d_max))
        
        # 缓存计算结果
        self._cache_objective_values = {}
        self._cache_gradients = {}
        
        # 预分配数组以避免重复内存分配
        self._temp_config = np.zeros((self.num_edge_nodes, self.num_models), dtype=np.float64)
        self._temp_gradient = np.zeros((self.num_edge_nodes, self.num_models), dtype=np.float64)
        
        # 批量处理参数
        self.batch_size = min(32, self.num_edge_nodes * self.num_models)

    def _compute_effective_model_sizes(self):
        """计算有效模型大小（考虑共享标识）"""
        effective_sizes = self.model_sizes.copy()
        
        if not self.Shared_flag:
            # 不考虑共享时，特定模型大小需要加上依赖的共享模型大小
            for specific_idx in range(self.num_specific):
                model_idx = specific_idx + self.num_shared
                shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
                for dep_idx in shared_deps:
                    effective_sizes[model_idx] += self.model_sizes[dep_idx]
        
        return effective_sizes

    def get_model_sizes_for_optimization(self):
        """获取用于优化的模型大小"""
        return self.effective_model_sizes if not self.Shared_flag else self.model_sizes

    def project_to_feasible_set(self, y):
        """优化的投影操作"""
        # 根据Shared_flag选择合适的模型大小
        sizes = self.get_model_sizes_for_optimization()
        return fast_project_capacity_constraint(y, sizes, self.edge_cache_storage)
    
    def continuous_to_binary(self, continuous_config):
        """优化的连续到二值转换"""
        sizes = self.get_model_sizes_for_optimization()
        binary_config = fast_continuous_to_binary(continuous_config, sizes, self.edge_cache_storage)
        
        # 如果不考虑共享，需要将共享模型的缓存状态清零
        if not self.Shared_flag:
            binary_config[:, :self.num_shared] = 0
        
        return binary_config
    
    def compute_gradient(self, y, requests, t):
        """优化的梯度计算 - 使用批量数值微分"""
        # 创建缓存键
        cache_key = (id(y), id(requests), t)
        if cache_key in self._cache_gradients:
            return self._cache_gradients[cache_key]
        
        y = y.astype(np.float64)
        if y.ndim == 1:
            y = y.reshape(self.num_edge_nodes, self.num_models)
        
        # 预分配梯度数组
        g = np.zeros_like(y, dtype=np.float64)
        
        # 使用更大的epsilon以减少数值不稳定性
        epsilon = 1e-4
        
        # 计算当前目标函数值
        params_current = np.concatenate([y.flatten(), np.zeros_like(y.flatten())])
        current_cost = self._fast_objective_function(params_current, requests, t)
        
        # 批量计算梯度 - 减少函数调用开销
        flat_y = y.flatten()
        
        for i in range(0, len(flat_y), self.batch_size):
            end_i = min(i + self.batch_size, len(flat_y))
            
            for j in range(i, end_i):
                # 创建扰动
                perturbed_y = flat_y.copy()
                perturbed_y[j] += epsilon
                
                params_perturbed = np.concatenate([perturbed_y, np.zeros_like(perturbed_y)])
                perturbed_cost = self._fast_objective_function(params_perturbed, requests, t)
                
                # 计算数值梯度
                grad_val = (perturbed_cost - current_cost) / epsilon
                
                # 将梯度映射回二维数组
                row_idx = j // self.num_models
                col_idx = j % self.num_models
                g[row_idx, col_idx] = grad_val
        
        # 缓存结果
        self._cache_gradients[cache_key] = -g
        
        # 清理缓存以避免内存泄漏
        if len(self._cache_gradients) > 100:
            self._cache_gradients.clear()
        
        return -g
    
    def _fast_objective_function(self, params, requests, t):
        """快速目标函数计算 - 简化版本"""
        cache_key = (tuple(params), id(requests), t)
        if cache_key in self._cache_objective_values:
            return self._cache_objective_values[cache_key]
        
        try:
            result = self.objective_function(params, self.dependency_matrix, requests, t, self.prev_deployment)
            self._cache_objective_values[cache_key] = result
            
            # 清理缓存
            if len(self._cache_objective_values) > 50:
                self._cache_objective_values.clear()
            
            return result
        except:
            return 1e10
    
    def compute_utility(self, y, requests, t):
        """优化的效用计算"""
        cost = self._fast_objective_function(
            np.concatenate([y.flatten(), np.zeros_like(y.flatten())]),
            requests, t
        )
        return -cost
    
    def compute_optimal_static_config(self, requests, t):
        """优化的最优静态配置计算"""
        # 使用更高效的初始化策略
        # 基于请求频率的启发式初始点
        x0 = self._compute_heuristic_initial_point(requests)
        
        def objective(params):
            try:
                cost = self._fast_objective_function(
                    np.concatenate([params, np.zeros_like(params)]),
                    requests, t
                )
                
                # 快速tensor转换
                if hasattr(cost, 'cpu'):
                    cost = float(cost.cpu().detach().numpy().item())
                elif hasattr(cost, 'numpy'):
                    cost = float(cost.numpy().item())
                elif hasattr(cost, 'item'):
                    cost = float(cost.item())
                else:
                    cost = float(cost)
                
                return cost
            except:
                return 1e10
        
        # 简化约束条件 - 使用向量化操作
        def global_capacity_constraint(params):
            config = params.reshape(self.num_edge_nodes, self.num_models)
            sizes = self.get_model_sizes_for_optimization()
            capacity_usage = np.sum(config * sizes[np.newaxis, :], axis=1)
            return self.edge_cache_storage - np.max(capacity_usage)
        
        constraints = [{'type': 'ineq', 'fun': global_capacity_constraint}]
        bounds = [(0, 1) for _ in range(self.num_edge_nodes * self.num_models)]
        
        try:
            # 使用更快的优化方法
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            
            if result.success:
                y_star = result.x.reshape(self.num_edge_nodes, self.num_models)
                # 确保满足容量约束
                y_star = self.project_to_feasible_set(y_star)
                return y_star
            else:
                return self.compute_greedy_optimal(requests)
        except:
            return self.compute_greedy_optimal(requests)
    
    def _compute_heuristic_initial_point(self, requests):
        """基于请求频率的启发式初始点"""
        x0 = np.zeros(self.num_edge_nodes * self.num_models)
        
        if len(self.current_task_requests) > 0:
            # 计算平均请求频率
            avg_requests = np.mean(self.current_task_requests, axis=0)
            if avg_requests.ndim == 1:
                avg_requests = avg_requests.reshape(self.num_edge_nodes, self.num_models)
            
            # 归一化请求频率
            max_req = np.max(avg_requests)
            if max_req > 0:
                normalized_requests = avg_requests / max_req
                x0 = normalized_requests.flatten()
        else:
            # 使用随机初始化，但偏向较小的值以满足容量约束
            x0 = np.random.exponential(0.1, size=self.num_edge_nodes * self.num_models)
            x0 = np.clip(x0, 0, 1)
        
        return x0
    
    def compute_greedy_optimal(self, requests):
        """优化的贪心方法"""
        y_star = np.zeros((self.num_edge_nodes, self.num_models), dtype=np.float64)
        
        # 预计算加权请求频率
        weighted_freq = np.zeros((self.num_edge_nodes, self.num_models), dtype=np.float64)
        
        for request in self.current_task_requests:
            if request.ndim == 1:
                request = request.reshape(self.num_edge_nodes, self.num_models)
            weighted_freq += request
        
        # 向量化价值计算 - 使用有效模型大小
        sizes = self.get_model_sizes_for_optimization()
        with np.errstate(divide='ignore', invalid='ignore'):
            values = np.divide(weighted_freq, sizes[np.newaxis, :], 
                             out=np.zeros_like(weighted_freq), where=sizes[np.newaxis, :] != 0)
        
        # 对每个边缘节点并行处理
        for b in range(self.num_edge_nodes):
            node_values = values[b]
            sorted_indices = np.argsort(node_values)[::-1]
            
            current_size = 0.0
            for idx in sorted_indices:
                # 如果不考虑共享，跳过共享模型
                if not self.Shared_flag and idx < self.num_shared:
                    continue
                
                if current_size + sizes[idx] <= self.edge_cache_storage:
                    y_star[b, idx] = 1.0
                    current_size += sizes[idx]
        
        return y_star
    
    def objective_function(self, params, dependency_matrix, requests, t, prev_deployment=None):
        """优化的目标函数 - 保持原有逻辑但提高性能"""
        try:
            # 快速参数检查和修正
            expected_length = 2 * self.args.edge_num * self.num_models
            if len(params) != expected_length:
                if len(params) > expected_length:
                    params = params[:expected_length]
                else:
                    params = np.pad(params, (0, expected_length - len(params)), 'constant', constant_values=0.5)
            
            # 快速requests处理
            if not isinstance(requests, np.ndarray):
                requests = np.array(requests, dtype=np.float64)
            
            if requests.ndim == 3:
                requests = requests[-1]
            if requests.ndim == 1 or requests.shape != (self.num_edge_nodes, self.num_models):
                requests = requests.reshape(self.num_edge_nodes, self.num_models)

            # 快速sigmoid计算
            half_length = len(params) // 2
            deployment = params[:half_length].reshape(self.args.edge_num, self.num_models)
            update_decision = params[half_length:].reshape(self.args.edge_num, self.num_models)
            
            deployment_probs = fast_sigmoid(deployment)
            update_probs = fast_sigmoid(update_decision)

            # 获取环境状态
            current_age_state = self.env._get_env_age_state()

            # 向量化成本计算
            switching_cost = 0.0
            update_cost = 0.0
            inference_cost = 0.0
            communication_cost = 0.0
            accuracy_cost = 0.0

            # 切换成本 - 向量化计算
            if prev_deployment is not None:
                deployment_changes = np.abs(deployment_probs - prev_deployment)
                model_switch_costs = np.array([self.env.get_switch_cost(mid) for mid in range(self.num_models)])
                switching_cost = np.sum(deployment_changes * model_switch_costs[np.newaxis, :])

            # 更新成本 - 向量化计算
            model_update_costs = np.array([self.env.get_update_cost(mid) for mid in range(self.num_models)])
            update_cost = np.sum(update_probs * model_update_costs[np.newaxis, :])

            # 批量处理请求 - 减少循环开销
            for node_id in range(min(self.env.NUM_EDGE_NODES, requests.shape[0])):
                node_requests = requests[node_id]
                node_deployment = deployment_probs[node_id]
                
                for specific_model_idx in range(self.env.NUM_SPECIFIC_MODELS):
                    model_global_id = specific_model_idx + self.num_shared
                    
                    if model_global_id >= len(node_requests):
                        continue
                        
                    req_rate = node_requests[model_global_id]
                    if req_rate <= 1e-8:  # 快速零检查
                        continue

                    # 精度成本
                    if model_global_id < current_age_state.shape[1]:
                        current_age = current_age_state[node_id, model_global_id]
                        model_accuracy = self.env.calculate_accuracy_from_age(current_age, model_global_id)
                        model_accuracy_min = getattr(self.env, 'accuracy_min', [0.9] * self.num_models)[model_global_id]
                        accuracy_deficit = max(0.0, model_accuracy_min - model_accuracy)
                        accuracy_cost += req_rate * accuracy_deficit * self.env.get_model_accuracy_cost(model_global_id)

                    # 分块依赖 - 优化索引计算
                    dependency_indices = np.nonzero(dependency_matrix[specific_model_idx])[0]
                    
                    # 部署概率计算
                    specific_cached = node_deployment[model_global_id]
                    
                    if self.Shared_flag:
                        # 考虑共享：需要检查共享模型的缓存概率
                        shared_cached_probs = node_deployment[dependency_indices[dependency_indices < self.num_shared]]

                        # 快速概率计算
                        if len(shared_cached_probs) > 0:
                            all_fragments_prob = specific_cached * np.prod(shared_cached_probs)
                            any_fragment_prob = 1.0 - np.prod(1.0 - shared_cached_probs)
                            partial_hit_prob = any_fragment_prob - all_fragments_prob
                        else:
                            all_fragments_prob = specific_cached
                            partial_hit_prob = 0.0
                    else:
                        # 不考虑共享：特定模型已包含共享部分
                        all_fragments_prob = specific_cached
                        partial_hit_prob = 0.0

                    # 成本计算
                    inference_cost += req_rate * args.INFERENCE_COST_PER_MB * all_fragments_prob
                    
                    communication_cost += (
                        req_rate * args.DELAY_LOCAL * all_fragments_prob +
                        req_rate * args.DELAY_CLOUD * partial_hit_prob +
                        req_rate * args.DELAY_CLOUD * (1.0 - all_fragments_prob - partial_hit_prob)
                    )

            # 共享模型下载成本 - 只在考虑共享时计算
            if self.Shared_flag:
                for shared_model_idx in range(min(self.num_shared, deployment_probs.shape[1])):
                    num_nodes = np.sum(deployment_probs[:, shared_model_idx] > 0.5)
                    if num_nodes > 0:
                        communication_cost += self.env.get_communication_cost(shared_model_idx) * num_nodes

            total_cost = switching_cost + communication_cost + update_cost + inference_cost + accuracy_cost
            
            # 快速tensor转换
            if hasattr(total_cost, 'cpu'):
                return float(total_cost.cpu().detach().numpy().item())
            elif hasattr(total_cost, 'item'):
                return float(total_cost.item())
            else:
                return float(total_cost)

        except Exception as e:
            return 1e10
    
    def optimize(self, t, requests):
        """优化的主优化函数"""
        # 快速请求处理
        if not isinstance(requests, np.ndarray):
            requests = np.array(requests, dtype=np.float64)
        
        if requests.ndim == 3:
            requests = requests[-1]
        
        if requests.ndim == 1:
            requests = requests.reshape(self.num_edge_nodes, self.num_models)
        elif requests.shape != (self.num_edge_nodes, self.num_models):
            try:
                requests = requests.reshape(self.num_edge_nodes, self.num_models)
            except:
                requests = np.zeros((self.num_edge_nodes, self.num_models), dtype=np.float64)
        
        # 记录请求
        self.current_task_requests.append(requests.copy())
        self.current_task_step += 1
        
        # 使用预计算的学习率
        eta = self.eta_base
        
        # 梯度计算和更新
        g = self.compute_gradient(self.current_cache_config, requests, t)
        
        # 原地更新以减少内存分配
        np.add(self.current_cache_config, eta * g, out=self.current_cache_config)
        
        # 投影到可行集
        self.current_cache_config = self.project_to_feasible_set(self.current_cache_config)
        
        # 任务边界检查
        if self.current_task_step >= self.task_length:
            # 元学习更新
            y_star = self.compute_optimal_static_config(self.current_task_requests, t)
            
            # 快速距离计算
            task_D_star = fast_norm_calculation(y_star.flatten(), self.Phi.flatten())
            if task_D_star > self.D_star:
                self.D_star = task_D_star
            
            # 元配置更新
            kappa = 1.0
            alpha = 1.0 / (kappa * (self.current_task_id + 1))
            
            # 原地更新
            np.subtract(self.Phi, alpha * (self.Phi - y_star), out=self.Phi)
            self.Phi = self.project_to_feasible_set(self.Phi)
            
            # 重置状态
            self.current_task_id += 1
            self.current_task_step = 0
            self.current_task_requests.clear()
            
            # 更新学习率
            self.eta_base = self.D_star / (np.sqrt(self.task_length) * self.w_inf * np.sqrt(self.d_max))
            
            # 重新初始化
            np.copyto(self.current_cache_config, self.Phi)
            self.prev_deployment = self.continuous_to_binary(self.current_cache_config)
        
        # 转换为二值配置
        new_cache_state = self.continuous_to_binary(self.current_cache_config)
        
        # 快速更新掩码生成
        update_mask = np.zeros_like(new_cache_state, dtype=bool)

        for node in range(args.edge_num):
            cached_models = np.where(new_cache_state[node] == 1)[0]
            if len(cached_models) == 0:
                continue

            # 随机选择至少一个模型进行更新
            num_selected = np.random.randint(1, len(cached_models) + 1)
            selected = np.random.choice(cached_models, num_selected, replace=False)

            for m in selected:
                update_mask[node, m] = True

                # # 只在考虑共享时处理特定模型的依赖关系
                # if self.Shared_flag and m >= self.num_shared:
                #     specific_idx = m - self.num_shared
                #     # 检查索引范围
                #     if specific_idx < self.dependency_matrix.shape[0]:
                #         shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
                #         # 将依赖的共享模型也标记为需要更新
                #         for dep_idx in shared_deps:
                #             if dep_idx < self.num_shared:  # 确保索引有效
                #                 update_mask[node, dep_idx] = True
                                
        return new_cache_state, update_mask