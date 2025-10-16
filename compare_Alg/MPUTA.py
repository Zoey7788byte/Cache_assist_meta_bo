# import numpy as np
# import cvxpy as cp
# import time

# class MPUTADeployment:
#     def __init__(self, env, all_requests, request_records, num_specific, num_shared, NUM_MODELS, args):
#         self.env = env
#         self.all_requests = all_requests
#         self.request_records = request_records
#         self.num_specific = num_specific
#         self.num_shared = num_shared
#         self.edge_cache_storage = args.edge_cache_storage
#         self.task_length = args.task_length
#         self.num_models = NUM_MODELS
#         self.num_edge_nodes = args.edge_num
#         self.args = args
        
#         # 获取模型大小
#         self.model_sizes = self.to_numpy(env.cloud_manager.model_sizes).astype(float)
        
#         # MPUTA算法参数
#         self.beta = 10.0  # QoE权重
#         self.rho = 0.1   # 效用函数衰减参数
#         self.epsilon = 1e-3  # 增大epsilon值以提高数值稳定性
#         self.gamma = np.log(1 + 1/self.epsilon)  # 正则化权重
#         self.max_exp_value = 700  # 防止指数溢出的最大值
        
#         # 成本参数
#         self.placement_cost = args.SWITCHING_COST_PER_MB
#         self.update_cost = args.UPDATE_COST_PER_MB
        
#         # 初始化请求频率
#         self.request_frequency = np.zeros((self.num_edge_nodes, self.num_models))
        
#         # 存储容量
#         self.storage_capacity = np.full(self.num_edge_nodes, self.edge_cache_storage).astype(float)
        
#         # 当前时隙
#         self.current_slot = 0
        
#         # 记录求解时间
#         self.solve_times = []
        
#         # AOI参数 - 使用二维向量 (节点数 × 模型数)
#         self.aoi_matrix = env.age_state
#         self.max_aoi = args.max_age # 最大AOI值
        
#         # 保存上一个时隙的部署决策
#         self.prev_Y = None
        
#     def to_numpy(self, tensor):
#         """将PyTorch张量转换为NumPy数组"""
#         if hasattr(tensor, 'cpu'):
#             return tensor.cpu().numpy()
#         return np.array(tensor)
    
#     def safe_exp(self, x):
#         """数值稳定的指数函数"""
#         return np.exp(np.minimum(x, self.max_exp_value))
    
#     def safe_exp_cvxpy(self, x):
#         """CVXPY中数值稳定的指数函数"""
#         return cp.exp(cp.minimum(x, self.max_exp_value))
    
#     def update_request_frequency(self, requests):
#         """更新请求频率矩阵"""
#         self.request_frequency = self.to_numpy(requests)
    
#     def relative_entropy_regularization(self, Y, Y_prev):
#         """优化的相对熵正则化项 """
#         if Y_prev is None:
#             return 0
        
#         # 使用更大的epsilon值确保数值稳定性
#         epsilon_reg = 1e-3
        
#         # 将Y_prev转换为参数而不是变量，避免复杂的双线性项
#         Y_prev_np = np.clip(Y_prev, epsilon_reg, 1 - epsilon_reg)
        
#         # 预计算常数项以提高效率
#         log_odds_prev = np.log(Y_prev_np / (1 - Y_prev_np))
        
#         # 使用向量化操作而不是循环
#         # 简化的KL散度形式：(y - y_prev) * log_odds_prev
#         Y_diff = Y - Y_prev_np
#         reg_matrix = cp.multiply(Y_diff, log_odds_prev)
        
#         return cp.sum(reg_matrix)

#     def solve_P3r(self, prev_Y):
#         """求解完全线性化的凸问题P3r_t - 优化版本避免SCIPY后端警告"""
#         start_time = time.time()
        
#         # 决策变量
#         Y = cp.Variable((self.num_edge_nodes, self.num_models), nonneg=True)  # 部署决策
#         U = cp.Variable((self.num_edge_nodes, self.num_models), nonneg=True)  # 更新决策
        
#         # 约束条件
#         constraints = []
        
#         # 存储约束 (10f) - 使用向量化操作
#         storage_usage = Y @ self.model_sizes  # 矩阵乘法，更高效
#         constraints.append(storage_usage <= self.storage_capacity)
        
#         # 更新决策约束: 只有部署的模型才能更新
#         constraints.append(U <= Y)
        
#         # 变量范围约束
#         constraints.append(Y <= 1)
#         constraints.append(U <= 1)
        
#         # 预计算所有效用值（避免在优化循环中重复计算）
#         base_utilities = np.zeros((self.num_edge_nodes, self.num_models))
#         update_gains = np.zeros((self.num_edge_nodes, self.num_models))
        
#         for j in range(self.num_edge_nodes):
#             for k in range(self.num_models):
#                 current_aoi = self.aoi_matrix[j, k]
#                 # 使用数值稳定的指数计算
#                 base_utilities[j, k] = self.safe_exp(-self.rho * min(current_aoi + 1, self.max_aoi))
#                 updated_utility = self.safe_exp(-self.rho * 1)
#                 update_gains[j, k] = max(0, updated_utility - base_utilities[j, k])
        
#         # 完全线性化的目标函数 - 使用向量化操作
#         # 1. QoE部分 - 部署效益
#         deployment_benefit_matrix = self.request_frequency * base_utilities
#         deployment_benefit = cp.sum(cp.multiply(deployment_benefit_matrix, Y))
        
#         # 2. QoE部分 - 更新效益  
#         update_benefit_matrix = self.request_frequency * update_gains
#         update_benefit = cp.sum(cp.multiply(update_benefit_matrix, U))
        
#         QoE = deployment_benefit + update_benefit
        
#         # 3. 部署成本 - 考虑模型大小
#         if prev_Y is not None:
#             # 选择正则化类型
#             use_relative_entropy = hasattr(self.args, 'use_relative_entropy') and self.args.use_relative_entropy
            
#             if use_relative_entropy:
#                 # 使用相对熵正则化
#                 placement_cost = self.placement_cost * self.relative_entropy_regularization(Y, prev_Y)
#             else:
#                 # 使用L1正则化，考虑模型大小 - 向量化版本
#                 Y_diff = Y - prev_Y
#                 # 使用cp.norm(..., 1)代替cp.sum(cp.abs(...))以提高兼容性
#                 model_weighted_changes = cp.multiply(Y_diff, self.model_sizes.reshape(1, -1))
#                 placement_cost = self.placement_cost * cp.norm(model_weighted_changes, 1)
#         else:
#             placement_cost = 0
        
#         # 4. 更新成本 - 向量化版本
#         model_sizes_matrix = np.tile(self.model_sizes, (self.num_edge_nodes, 1))
#         update_cost = self.update_cost * cp.sum(cp.multiply(U, model_sizes_matrix))
        
#         # 总目标函数 - 完全线性
#         objective = cp.Maximize(self.beta * QoE - placement_cost - update_cost)
        
#         # 创建问题并求解
#         problem = cp.Problem(objective, constraints)
        
#         # 尝试多个求解器，按稳定性排序
#         solvers_config = [
#             (cp.ECOS, {'max_iters': 1000, 'abstol': 1e-6, 'reltol': 1e-6}),
#             (cp.OSQP, {'max_iter': 2000, 'eps_abs': 1e-5, 'eps_rel': 1e-5, 'verbose': False}),
#             (cp.SCS, {'max_iters': 2000, 'eps': 1e-4, 'verbose': False})
#         ]
        
#         for solver, params in solvers_config:
#             try:
#                 problem.solve(solver=solver, **params)
                
#                 if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
#                     Y_frac = np.clip(Y.value, 0, 1)
#                     U_frac = np.clip(U.value, 0, 1)
#                     Y_frac = np.nan_to_num(Y_frac, nan=0.0)
#                     U_frac = np.nan_to_num(U_frac, nan=0.0)
                    
#                     end_time = time.time()
#                     self.solve_times.append(end_time - start_time)
#                     return Y_frac, U_frac, problem.value
#                 else:
#                     print(f"求解器 {solver} 状态: {problem.status}")
                    
#             except Exception as e:
#                 print(f"求解器 {solver} 失败: {e}")
#                 continue
        
#         # 如果所有求解器都失败，使用启发式解
#         print("所有求解器都失败，使用启发式解")
#         Y_frac, U_frac = self.heuristic_solution()
#         end_time = time.time()
#         self.solve_times.append(end_time - start_time)
#         return Y_frac, U_frac, 0
    
#     def heuristic_solution(self):
#         """改进的启发式解决方案 - 基于效用密度的贪心算法"""
#         Y_frac = np.zeros((self.num_edge_nodes, self.num_models))
#         U_frac = np.zeros((self.num_edge_nodes, self.num_models))
        
#         # 计算每个模型在每个节点上的效用密度 (效用增益/存储成本)
#         utility_densities = np.zeros((self.num_edge_nodes, self.num_models))
#         update_utilities = np.zeros((self.num_edge_nodes, self.num_models))
        
#         for j in range(self.num_edge_nodes):
#             for k in range(self.num_models):
#                 current_aoi = self.aoi_matrix[j, k]
#                 request_freq = self.request_frequency[j, k]
#                 model_size = self.model_sizes[k]
                
#                 # 当前效用 (不更新)
#                 current_utility = self.safe_exp(-self.rho * min(current_aoi + 1, self.max_aoi))
#                 # 更新后的效用
#                 updated_utility = self.safe_exp(-self.rho * 1)
                
#                 # 部署效用密度 = (请求频率 * 基础效用) / 模型大小
#                 base_utility_gain = request_freq * current_utility
#                 utility_densities[j, k] = base_utility_gain / (model_size + self.epsilon)
                
#                 # 更新效用增益
#                 update_gain = updated_utility - current_utility
#                 update_utilities[j, k] = request_freq * update_gain
        
#         # 为每个边缘节点分配模型
#         for j in range(self.num_edge_nodes):
#             # 按效用密度降序排序
#             sorted_indices = np.argsort(-utility_densities[j, :])
            
#             remaining_capacity = self.storage_capacity[j]
#             total_request_freq = np.sum(self.request_frequency[j, :]) + self.epsilon
#             max_utility_density = np.max(utility_densities[j, :]) + self.epsilon
            
#             for k in sorted_indices:
#                 if remaining_capacity >= self.model_sizes[k]:
#                     # 基于效用密度和请求频率计算部署概率
#                     freq_ratio = self.request_frequency[j, k] / total_request_freq
#                     density_ratio = utility_densities[j, k] / max_utility_density
                    
#                     # 自适应部署概率：结合频率比和密度比
#                     deploy_prob = min(1.0, 
#                                     0.1 + 0.4 * freq_ratio + 0.5 * density_ratio)
                    
#                     # 考虑存储利用率的调整
#                     capacity_usage = 1 - (remaining_capacity / self.storage_capacity[j])
#                     if capacity_usage > 0.7:  # 当存储使用率超过70%时，更加谨慎
#                         deploy_prob *= (0.5 + 0.5 * density_ratio)
                    
#                     Y_frac[j, k] = deploy_prob
                    
#                     # 计算更新概率
#                     if update_utilities[j, k] > 0:
#                         current_aoi = self.aoi_matrix[j, k]
#                         max_update_utility = np.max(update_utilities[j, :]) + self.epsilon
                        
#                         # 更新概率基于：更新效用增益 + AOI紧迫性
#                         utility_ratio = update_utilities[j, k] / max_update_utility
#                         aoi_urgency = min(current_aoi / self.max_aoi, 1.0)  # AOI紧迫程度
                        
#                         # 自适应更新概率
#                         update_prob = min(1.0, 
#                                         0.05 + 0.4 * utility_ratio + 0.35 * aoi_urgency)
                        
#                         # 更新概率不能超过部署概率
#                         U_frac[j, k] = update_prob * deploy_prob
                    
#                     # 更新剩余容量
#                     remaining_capacity -= self.model_sizes[k] * deploy_prob
                    
#                     # 如果剩余容量不足，停止分配
#                     if remaining_capacity < np.min(self.model_sizes) * 0.1:
#                         break
        
#         return Y_frac, U_frac
    
#     def randomized_rounding(self, Y_frac, U_frac):
#         """改进的随机舍入算法"""
#         Y_int = np.zeros_like(Y_frac, dtype=int)
#         U_int = np.zeros_like(U_frac, dtype=int)
        
#         # 首先处理部署决策
#         for j in range(self.num_edge_nodes):
#             # 按分数值降序排序
#             sorted_indices = np.argsort(-Y_frac[j, :])
#             remaining_capacity = self.storage_capacity[j]
            
#             for k in sorted_indices:
#                 if remaining_capacity >= self.model_sizes[k]:
#                     # 使用概率舍入
#                     if np.random.rand() < Y_frac[j, k]:
#                         Y_int[j, k] = 1
#                         remaining_capacity -= self.model_sizes[k]
        
#         # 然后处理更新决策
#         for j in range(self.num_edge_nodes):
#             for k in range(self.num_models):
#                 if Y_int[j, k] == 1 and U_frac[j, k] > 0:
#                     # 避免除以零
#                     y_frac_val = max(Y_frac[j, k], 1e-10)
#                     update_prob = min(1.0, U_frac[j, k] / y_frac_val)
#                     if np.random.rand() < update_prob:
#                         U_int[j, k] = 1
        
#         # 新增：特定模型更新触发共享模型更新
#         for j in range(self.num_edge_nodes):
#             for k in range(self.num_shared, self.num_models):  # 遍历所有特定模型
#                 if U_int[j, k] == 1:  # 如果特定模型需要更新
#                     # 找到对应的共享模型索引
#                     # 这里假设特定模型k对应共享模型k % num_shared
#                     # 您可能需要根据实际映射关系调整这个逻辑
#                     shared_idx = k % self.num_shared
#                     # 确保共享模型已被部署
#                     if Y_int[j, shared_idx] == 1:
#                         U_int[j, shared_idx] = 1  # 标记共享模型也需要更新
#                     # else:
#                     #     # 如果共享模型未被部署，可能需要先部署再更新
#                     #     # 这里可以根据您的需求决定如何处理
#                     #     print(f"警告: 特定模型{k}需要更新，但对应共享模型{shared_idx}未部署")
        
#         return Y_int, U_int
    
#     def optimize(self, t, requests, prev_Y):
#         """MPUTA算法主函数"""
#         self.current_slot = t
        
#         # 更新请求频率矩阵
#         self.update_request_frequency(requests)
        
#         # 求解正则化凸问题
#         Y_frac, U_frac, obj_value = self.solve_P3r(prev_Y)
        
#         # 随机舍入得到整数解
#         Y_int, U_int = self.randomized_rounding(Y_frac, U_frac)
        
#         return Y_int, U_int



import numpy as np
import cvxpy as cp
import time

class MPUTADeployment:
    def __init__(self, env, all_requests, request_records, num_specific, num_shared, NUM_MODELS, args):
        self.env = env
        self.all_requests = all_requests
        self.request_records = request_records
        self.num_specific = num_specific
        self.num_shared = num_shared
        self.edge_cache_storage = args.edge_cache_storage
        self.task_length = args.task_length
        self.num_models = NUM_MODELS
        self.num_edge_nodes = args.edge_num
        self.args = args
        
        # 添加Shared_flag标识
        self.Shared_flag = getattr(args, 'Shared_flag', True)
        
        # 获取模型大小
        self.model_sizes = self.to_numpy(env.cloud_manager.model_sizes).astype(float)
        
        # 获取依赖矩阵
        self.dependency_matrix = np.array(env.cloud_manager.get_dependency_matrix(), dtype=float)
        
        # 计算有效模型大小
        self.effective_model_sizes = self._compute_effective_model_sizes()
        
        # MPUTA算法参数
        self.beta = 10.0  # QoE权重
        self.rho = 0.1   # 效用函数衰减参数
        self.epsilon = 1e-3  # 增大epsilon值以提高数值稳定性
        self.gamma = np.log(1 + 1/self.epsilon)  # 正则化权重
        self.max_exp_value = 700  # 防止指数溢出的最大值
        
        # 成本参数
        self.placement_cost = args.SWITCHING_COST_PER_MB
        self.update_cost = args.UPDATE_COST_PER_MB
        
        # 初始化请求频率
        self.request_frequency = np.zeros((self.num_edge_nodes, self.num_models))
        
        # 存储容量
        self.storage_capacity = np.full(self.num_edge_nodes, self.edge_cache_storage).astype(float)
        
        # 当前时隙
        self.current_slot = 0
        
        # 记录求解时间
        self.solve_times = []
        
        # AOI参数 - 使用二维向量 (节点数 × 模型数)
        self.aoi_matrix = env.age_state
        self.max_aoi = args.max_age # 最大AOI值
        
        # 保存上一个时隙的部署决策
        self.prev_Y = None
    
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
        
    def to_numpy(self, tensor):
        """将PyTorch张量转换为NumPy数组"""
        if hasattr(tensor, 'cpu'):
            return tensor.cpu().numpy()
        return np.array(tensor)
    
    def safe_exp(self, x):
        """数值稳定的指数函数"""
        return np.exp(np.minimum(x, self.max_exp_value))
    
    def safe_exp_cvxpy(self, x):
        """CVXPY中数值稳定的指数函数"""
        return cp.exp(cp.minimum(x, self.max_exp_value))
    
    def update_request_frequency(self, requests):
        """更新请求频率矩阵"""
        self.request_frequency = self.to_numpy(requests)
    
    def relative_entropy_regularization(self, Y, Y_prev):
        """优化的相对熵正则化项 """
        if Y_prev is None:
            return 0
        
        # 使用更大的epsilon值确保数值稳定性
        epsilon_reg = 1e-3
        
        # 将Y_prev转换为参数而不是变量，避免复杂的双线性项
        Y_prev_np = np.clip(Y_prev, epsilon_reg, 1 - epsilon_reg)
        
        # 预计算常数项以提高效率
        log_odds_prev = np.log(Y_prev_np / (1 - Y_prev_np))
        
        # 使用向量化操作而不是循环
        # 简化的KL散度形式：(y - y_prev) * log_odds_prev
        Y_diff = Y - Y_prev_np
        reg_matrix = cp.multiply(Y_diff, log_odds_prev)
        
        return cp.sum(reg_matrix)

    def solve_P3r(self, prev_Y):
        """求解完全线性化的凸问题P3r_t - 优化版本避免SCIPY后端警告"""
        start_time = time.time()
        
        # 决策变量
        Y = cp.Variable((self.num_edge_nodes, self.num_models), nonneg=True)  # 部署决策
        U = cp.Variable((self.num_edge_nodes, self.num_models), nonneg=True)  # 更新决策
        
        # 约束条件
        constraints = []
        
        # 获取用于优化的模型大小
        opt_model_sizes = self.get_model_sizes_for_optimization()
        
        if self.Shared_flag:
            # 考虑共享：使用原始模型大小
            storage_usage = Y @ opt_model_sizes
            constraints.append(storage_usage <= self.storage_capacity)
        else:
            # 不考虑共享：只考虑特定模型，使用有效大小
            # 共享模型不应该被部署
            for k in range(self.num_shared):
                constraints.append(Y[:, k] == 0)
            
            # 特定模型使用有效大小进行存储约束
            specific_sizes = opt_model_sizes[self.num_shared:]
            Y_specific = Y[:, self.num_shared:]
            storage_usage = Y_specific @ specific_sizes
            constraints.append(storage_usage <= self.storage_capacity)
        
        # 更新决策约束: 只有部署的模型才能更新
        constraints.append(U <= Y)
        
        # 变量范围约束
        constraints.append(Y <= 1)
        constraints.append(U <= 1)
        
        # 预计算所有效用值（避免在优化循环中重复计算）
        base_utilities = np.zeros((self.num_edge_nodes, self.num_models))
        update_gains = np.zeros((self.num_edge_nodes, self.num_models))
        
        for j in range(self.num_edge_nodes):
            for k in range(self.num_models):
                current_aoi = self.aoi_matrix[j, k]
                # 使用数值稳定的指数计算
                base_utilities[j, k] = self.safe_exp(-self.rho * min(current_aoi + 1, self.max_aoi))
                updated_utility = self.safe_exp(-self.rho * 1)
                update_gains[j, k] = max(0, updated_utility - base_utilities[j, k])
        
        # 完全线性化的目标函数 - 使用向量化操作
        # 1. QoE部分 - 部署效益
        deployment_benefit_matrix = self.request_frequency * base_utilities
        deployment_benefit = cp.sum(cp.multiply(deployment_benefit_matrix, Y))
        
        # 2. QoE部分 - 更新效益  
        update_benefit_matrix = self.request_frequency * update_gains
        update_benefit = cp.sum(cp.multiply(update_benefit_matrix, U))
        
        QoE = deployment_benefit + update_benefit
        
        # 3. 部署成本 - 考虑模型大小
        if prev_Y is not None:
            # 选择正则化类型
            use_relative_entropy = hasattr(self.args, 'use_relative_entropy') and self.args.use_relative_entropy
            
            if use_relative_entropy:
                # 使用相对熵正则化
                placement_cost = self.placement_cost * self.relative_entropy_regularization(Y, prev_Y)
            else:
                # 使用L1正则化，考虑模型大小 - 向量化版本
                Y_diff = Y - prev_Y
                # 使用有效模型大小计算切换成本
                model_weighted_changes = cp.multiply(Y_diff, opt_model_sizes.reshape(1, -1))
                placement_cost = self.placement_cost * cp.norm(model_weighted_changes, 1)
        else:
            placement_cost = 0
        
        # 4. 更新成本 - 向量化版本
        model_sizes_matrix = np.tile(opt_model_sizes, (self.num_edge_nodes, 1))
        update_cost = self.update_cost * cp.sum(cp.multiply(U, model_sizes_matrix))
        
        # 总目标函数 - 完全线性
        objective = cp.Maximize(self.beta * QoE - placement_cost - update_cost)
        
        # 创建问题并求解
        problem = cp.Problem(objective, constraints)
        
        # 尝试多个求解器，按稳定性排序
        solvers_config = [
            (cp.ECOS, {'max_iters': 1000, 'abstol': 1e-6, 'reltol': 1e-6}),
            (cp.OSQP, {'max_iter': 2000, 'eps_abs': 1e-5, 'eps_rel': 1e-5, 'verbose': False}),
            (cp.SCS, {'max_iters': 2000, 'eps': 1e-4, 'verbose': False})
        ]
        
        for solver, params in solvers_config:
            try:
                problem.solve(solver=solver, **params)
                
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    Y_frac = np.clip(Y.value, 0, 1)
                    U_frac = np.clip(U.value, 0, 1)
                    Y_frac = np.nan_to_num(Y_frac, nan=0.0)
                    U_frac = np.nan_to_num(U_frac, nan=0.0)
                    
                    # 如果不考虑共享，确保共享模型状态为0
                    if not self.Shared_flag:
                        Y_frac[:, :self.num_shared] = 0
                        U_frac[:, :self.num_shared] = 0
                    
                    end_time = time.time()
                    self.solve_times.append(end_time - start_time)
                    return Y_frac, U_frac, problem.value
                else:
                    print(f"求解器 {solver} 状态: {problem.status}")
                    
            except Exception as e:
                print(f"求解器 {solver} 失败: {e}")
                continue
        
        # 如果所有求解器都失败，使用启发式解
        print("所有求解器都失败，使用启发式解")
        Y_frac, U_frac = self.heuristic_solution()
        end_time = time.time()
        self.solve_times.append(end_time - start_time)
        return Y_frac, U_frac, 0
    
    def heuristic_solution(self):
        """改进的启发式解决方案 - 基于效用密度的贪心算法"""
        Y_frac = np.zeros((self.num_edge_nodes, self.num_models))
        U_frac = np.zeros((self.num_edge_nodes, self.num_models))
        
        # 获取用于优化的模型大小
        opt_model_sizes = self.get_model_sizes_for_optimization()
        
        # 计算每个模型在每个节点上的效用密度 (效用增益/存储成本)
        utility_densities = np.zeros((self.num_edge_nodes, self.num_models))
        update_utilities = np.zeros((self.num_edge_nodes, self.num_models))
        
        for j in range(self.num_edge_nodes):
            # 确定候选模型范围
            if self.Shared_flag:
                candidate_range = range(self.num_models)
            else:
                candidate_range = range(self.num_shared, self.num_models)
            
            for k in candidate_range:
                current_aoi = self.aoi_matrix[j, k]
                request_freq = self.request_frequency[j, k]
                model_size = opt_model_sizes[k]
                
                # 当前效用 (不更新)
                current_utility = self.safe_exp(-self.rho * min(current_aoi + 1, self.max_aoi))
                # 更新后的效用
                updated_utility = self.safe_exp(-self.rho * 1)
                
                # 部署效用密度 = (请求频率 * 基础效用) / 模型大小
                base_utility_gain = request_freq * current_utility
                utility_densities[j, k] = base_utility_gain / (model_size + self.epsilon)
                
                # 更新效用增益
                update_gain = updated_utility - current_utility
                update_utilities[j, k] = request_freq * update_gain
        
        # 为每个边缘节点分配模型
        for j in range(self.num_edge_nodes):
            # 确定候选模型
            if self.Shared_flag:
                candidate_models = np.arange(self.num_models)
            else:
                candidate_models = np.arange(self.num_shared, self.num_models)
            
            # 按效用密度降序排序
            sorted_indices = candidate_models[np.argsort(-utility_densities[j, candidate_models])]
            
            remaining_capacity = self.storage_capacity[j]
            total_request_freq = np.sum(self.request_frequency[j, candidate_models]) + self.epsilon
            max_utility_density = np.max(utility_densities[j, candidate_models]) + self.epsilon
            
            for k in sorted_indices:
                if remaining_capacity >= opt_model_sizes[k]:
                    # 基于效用密度和请求频率计算部署概率
                    freq_ratio = self.request_frequency[j, k] / total_request_freq
                    density_ratio = utility_densities[j, k] / max_utility_density
                    
                    # 自适应部署概率：结合频率比和密度比
                    deploy_prob = min(1.0, 
                                    0.1 + 0.4 * freq_ratio + 0.5 * density_ratio)
                    
                    # 考虑存储利用率的调整
                    capacity_usage = 1 - (remaining_capacity / self.storage_capacity[j])
                    if capacity_usage > 0.7:  # 当存储使用率超过70%时，更加谨慎
                        deploy_prob *= (0.5 + 0.5 * density_ratio)
                    
                    Y_frac[j, k] = deploy_prob
                    
                    # 计算更新概率
                    if update_utilities[j, k] > 0:
                        current_aoi = self.aoi_matrix[j, k]
                        max_update_utility = np.max(update_utilities[j, candidate_models]) + self.epsilon
                        
                        # 更新概率基于：更新效用增益 + AOI紧迫性
                        utility_ratio = update_utilities[j, k] / max_update_utility
                        aoi_urgency = min(current_aoi / self.max_aoi, 1.0)  # AOI紧迫程度
                        
                        # 自适应更新概率
                        update_prob = min(1.0, 
                                        0.05 + 0.4 * utility_ratio + 0.35 * aoi_urgency)
                        
                        # 更新概率不能超过部署概率
                        U_frac[j, k] = update_prob * deploy_prob
                    
                    # 更新剩余容量
                    remaining_capacity -= opt_model_sizes[k] * deploy_prob
                    
                    # 如果剩余容量不足，停止分配
                    if remaining_capacity < np.min(opt_model_sizes[candidate_models]) * 0.1:
                        break
        
        return Y_frac, U_frac
    
    def randomized_rounding(self, Y_frac, U_frac):
        """改进的随机舍入算法"""
        Y_int = np.zeros_like(Y_frac, dtype=int)
        U_int = np.zeros_like(U_frac, dtype=int)
        
        # 获取用于优化的模型大小
        opt_model_sizes = self.get_model_sizes_for_optimization()
        
        # 首先处理部署决策
        for j in range(self.num_edge_nodes):
            # 确定候选模型
            if self.Shared_flag:
                candidate_models = np.arange(self.num_models)
            else:
                candidate_models = np.arange(self.num_shared, self.num_models)
            
            # 按分数值降序排序
            sorted_indices = candidate_models[np.argsort(-Y_frac[j, candidate_models])]
            remaining_capacity = self.storage_capacity[j]
            
            for k in sorted_indices:
                if remaining_capacity >= opt_model_sizes[k]:
                    # 使用概率舍入
                    if np.random.rand() < Y_frac[j, k]:
                        Y_int[j, k] = 1
                        remaining_capacity -= opt_model_sizes[k]
                        
                        # 如果考虑共享且是特定模型，需要确保依赖的共享模型也被部署
                        if self.Shared_flag and k >= self.num_shared:
                            specific_idx = k - self.num_shared
                            shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
                            for dep_idx in shared_deps:
                                if Y_int[j, dep_idx] == 0:
                                    if remaining_capacity >= self.model_sizes[dep_idx]:
                                        Y_int[j, dep_idx] = 1
                                        remaining_capacity -= self.model_sizes[dep_idx]
        
        # 然后处理更新决策
        for j in range(self.num_edge_nodes):
            for k in range(self.num_models):
                if Y_int[j, k] == 1 and U_frac[j, k] > 0:
                    # 避免除以零
                    y_frac_val = max(Y_frac[j, k], 1e-10)
                    update_prob = min(1.0, U_frac[j, k] / y_frac_val)
                    if np.random.rand() < update_prob:
                        U_int[j, k] = 1
        
        # 只在考虑共享时：特定模型更新触发共享模型更新
        if self.Shared_flag:
            for j in range(self.num_edge_nodes):
                for k in range(self.num_shared, self.num_models):  # 遍历所有特定模型
                    if U_int[j, k] == 1:  # 如果特定模型需要更新
                        specific_idx = k - self.num_shared
                        shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
                        for dep_idx in shared_deps:
                            # 确保共享模型已被部署
                            if Y_int[j, dep_idx] == 1:
                                U_int[j, dep_idx] = 1  # 标记共享模型也需要更新
        
        return Y_int, U_int
    
    def optimize(self, t, requests, prev_Y):
        """MPUTA算法主函数"""
        self.current_slot = t
        
        # 更新请求频率矩阵
        self.update_request_frequency(requests)
        
        # 求解正则化凸问题
        Y_frac, U_frac, obj_value = self.solve_P3r(prev_Y)
        
        # 随机舍入得到整数解
        Y_int, U_int = self.randomized_rounding(Y_frac, U_frac)
        
        return Y_int, U_int