import torch
import numpy as np
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from parameters import args_parser
from botorch.models.transforms import Standardize, Normalize
args = args_parser()

class BayesianOptimizer:
    def __init__(self, env, meta_net, NUM_MODELS, num_edges, bo_iters=10, q=1):
        self.env = env
        self.meta_net = meta_net
        self.NUM_MODELS = NUM_MODELS
        self.num_edges = num_edges
        self.observed_points = []
        self.observed_values = []
        self.best_params = None
        self.best_value = float('inf')  
        self.model_deployed = np.zeros((args.edge_num, NUM_MODELS), dtype=bool)
        # 使用双精度浮点数
        self.dtype = torch.float64
        
        self.bo_iters = bo_iters
        self.init_points = args.init_points
        self.q = q
        
        # 预缓存模型大小
        self.model_sizes = np.array(self.env.cloud_manager.model_sizes) if hasattr(self.env, 'cloud_manager') else np.ones(NUM_MODELS)

    def optimize_step(self, t, requests, prev_deployment=None):
        """执行统一优化步骤"""
        # 1. 从元网络获取初始参数
        state = self.env.get_state_vector(t)
        with torch.no_grad():
            deploy_output = self.meta_net(state)
        
        mid_point = self.num_edges * self.NUM_MODELS
        deploy_logits = deploy_output[..., :mid_point].view(self.num_edges, self.NUM_MODELS)
        update_logits = deploy_output[..., mid_point:].view(self.num_edges, self.NUM_MODELS)  

        # 确保所有参数在 [0,1] 范围内
        deploy_params = torch.sigmoid(deploy_logits).cpu().numpy().flatten()
        update_params = torch.sigmoid(update_logits).cpu().numpy().flatten()
        
        # 合并部署和更新参数
        init_params = np.concatenate([deploy_params, update_params])
        
        # 处理prev_deployment参数
        if prev_deployment is None:
            cache_state = np.array(self.env.cache_state)
            if cache_state.ndim == 1:
                cache_state = cache_state.reshape(self.env.NUM_EDGE_NODES, self.env.NUM_MODELS)
            prev_deployment = cache_state
        else:
            prev_deployment = np.array(prev_deployment)
            if prev_deployment.ndim == 1:
                prev_deployment = prev_deployment.reshape(self.env.NUM_EDGE_NODES, self.env.NUM_MODELS)
        
        # 初始化最佳值
        if self.best_value is None or self.best_value == float('inf'):
            self.best_value = float('inf')
            self.best_params = init_params
        
        # 收集初始点 - 使用更智能的初始化
        if len(self.observed_points) < self.init_points:
            # 使用均匀分布而不是随机分布，确保更好的覆盖
            deploy_random = np.random.uniform(0, 1, deploy_params.shape)
            update_random = np.random.uniform(0, 1, update_params.shape)
            params = np.concatenate([deploy_random, update_random])
            
            value = self.objective_function(params, self.env.cloud_manager.dependency_matrix, requests, t, prev_deployment)
            self.observed_points.append(params.tolist())
            self.observed_values.append(value)
            
            # 更新最佳值
            if value < self.best_value:
                self.best_value = value
                self.best_params = params
            
            # print(f"初始点 {len(self.observed_points)}: 值 = {value:.3f}")
            return params, value
        
        try:
            # 2. 构建高斯过程模型
            train_x = torch.tensor(np.array(self.observed_points), dtype=self.dtype, device=self.env.device)
            train_y = -torch.tensor(self.observed_values, dtype=self.dtype, device=self.env.device).view(-1, 1)
            
            # 检查训练数据
            # print(f"训练数据: {train_x.shape[0]} 个点, 目标值范围: {train_y.min().item():.3f} 到 {train_y.max().item():.3f}")

            model = SingleTaskGP(
                train_x,
                train_y,
                input_transform=Normalize(train_x.shape[-1]),
                outcome_transform=Standardize(m=1)
            )

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # 检查模型拟合
            with torch.no_grad():
                posterior = model.posterior(train_x)
                # print(f"后验均值范围: {posterior.mean.min().item():.3f} 到 {posterior.mean.max().item():.3f}")
            
            # 3. 使用期望改进(EI)获取候选点
            best_f = train_y.max()
            # print(f"当前最佳值: {best_f.item():.3f}")
            
            EI = LogExpectedImprovement(model, best_f=best_f)
            
            # 定义参数边界
            bounds = torch.tensor(
                [[0.0] * train_x.shape[1], [1.0] * train_x.shape[1]], 
                dtype=self.dtype, 
                device=self.env.device
            )
            
            # 优化采集函数 - 增加探索能力
            candidates, acq_value = optimize_acqf(
                EI, 
                bounds=bounds, 
                q=1, 
                num_restarts=10,  # 增加重启次数
                raw_samples=50,   # 增加原始样本数
                options={"batch_limit": 5}
            )
            
            # print(f"采集函数值: {acq_value.item():.6f}")
            
            new_params = candidates[0].cpu().numpy()
            new_value = self.objective_function(new_params, self.env.cloud_manager.dependency_matrix, requests, t, prev_deployment)
            
            # 4. 更新观察记录
            self.observed_points.append(new_params.tolist())
            self.observed_values.append(new_value)
            
            # 5. 更新最佳值
            if new_value < self.best_value:
                self.best_value = new_value
                self.best_params = new_params
            
            # print(f"新点: 值 = {new_value:.3f}, 最佳值 = {self.best_value:.3f}")
            return new_params, new_value
        
        except Exception as e:
            print(f"贝叶斯优化失败: {e}")
            # 返回当前最佳参数并添加到观察点
            if self.best_params is not None and self.best_value < float('inf'):
                params = self.best_params
                value = self.best_value
                self.observed_points.append(params.tolist())
                self.observed_values.append(value)
            else:
                params = init_params
                value = self.objective_function(params, self.env.cloud_manager.dependency_matrix, requests, t, prev_deployment)
                self.observed_points.append(params.tolist())
                self.observed_values.append(value)
            return params, value

    def objective_function(self, params, dependency_matrix, requests, t, prev_deployment=None):
        try:
            # 参数长度校验与修正
            expected_length = 2 * self.env.NUM_EDGE_NODES * self.env.NUM_MODELS
            if len(params) != expected_length:
                if len(params) > expected_length:
                    params = params[:expected_length]
                else:
                    padding = np.full(expected_length - len(params), 0.5)
                    params = np.concatenate([params, padding])

            half_length = len(params) // 2
            deployment_probs = 1 / (1 + np.exp(-params[:half_length].reshape(self.env.NUM_EDGE_NODES, self.env.NUM_MODELS)))
            update_probs = 1 / (1 + np.exp(-params[half_length:].reshape(self.env.NUM_EDGE_NODES, self.env.NUM_MODELS)))

            # 获取当前年龄状态
            current_age_state = self.env._get_env_age_state()
            
            # 修正年龄状态更新逻辑，使其与 _update_isolated_state 一致
            updated_age_state = np.zeros_like(current_age_state)
            for i in range(self.env.NUM_EDGE_NODES):
                for j in range(self.env.NUM_MODELS):
                    if update_probs[i, j] > 0.5:  # 模拟更新
                        updated_age_state[i, j] = 0
                    elif deployment_probs[i, j] > 0.5:  # 模拟部署但未更新
                        if prev_deployment is not None and prev_deployment[i, j] > 0.5:  # 之前已部署
                            updated_age_state[i, j] = current_age_state[i, j] + 1
                        else:  # 新部署
                            updated_age_state[i, j] = 0
                    else:  # 模拟未部署
                        updated_age_state[i, j] = current_age_state[i, j]  # 保持不变
            
            # 修正精度计算方式，使其与 _update_isolated_state 一致
            # 计算精度衰减因子（基于更新后的年龄状态）
            max_age = getattr(self.env, 'max_age', getattr(args, 'max_age', 100))
            age_decay_factor = getattr(self.env, 'age_decay_factor', getattr(args, 'age_decay_factor', 1.0))
            age_ratio = np.clip(updated_age_state / max_age, 0.0, 1.0)
            accuracy_decay = np.exp(-age_decay_factor * age_ratio)
            
            # 获取初始精度
            if hasattr(self.env, 'initial_acc'):
                initial_acc = self.env.initial_acc
                if isinstance(initial_acc, list):
                    initial_acc = np.array(initial_acc)
                if initial_acc.ndim == 1:
                    initial_acc_expanded = np.tile(initial_acc, (self.env.NUM_EDGE_NODES, 1))
                else:
                    initial_acc_expanded = initial_acc
            else:
                initial_acc_expanded = np.ones((self.env.NUM_EDGE_NODES, self.env.NUM_MODELS))
            
            # 计算更新后的精度
            updated_accuracies = np.zeros_like(updated_age_state)
            for i in range(self.env.NUM_EDGE_NODES):
                for j in range(self.env.NUM_MODELS):
                    if update_probs[i, j] > 0.5:  # 模拟更新
                        updated_accuracies[i, j] = initial_acc_expanded[i, j]
                    elif deployment_probs[i, j] > 0.5:  # 模拟部署但未更新
                        updated_accuracies[i, j] = initial_acc_expanded[i, j] * accuracy_decay[i, j]
                    else:  # 模拟未部署
                        # 保持当前精度不变（这里简化处理，使用初始精度）
                        updated_accuracies[i, j] = initial_acc_expanded[i, j]
            
            # 确保精度在合理范围内
            accuracy_min = getattr(self.env, 'accuracy_min', 0.1)
            updated_accuracies = np.clip(updated_accuracies, accuracy_min, 1.0)
            
            # 其余代码保持不变...
            # 初始化成本项
            switching_cost = 0.0
            update_cost = 0.0
            inference_cost = 0.0
            communication_cost = 0.0
            accuracy_cost = 0.0

            total_fragments_needed = 0.0
            total_fragments_available = 0.0

            # 1. 切换成本
            if prev_deployment is not None:
                activation_switches = deployment_probs * (1 - prev_deployment)
                model_switch_costs = np.array([self.env.get_switch_cost(mid) for mid in range(self.env.NUM_MODELS)])
                switching_cost = np.sum(activation_switches * model_switch_costs)

            # 2. 更新成本
            model_update_costs = np.array([self.env.get_update_cost(mid) for mid in range(self.env.NUM_MODELS)])
            update_cost = np.sum(update_probs * deployment_probs * model_update_costs)

            # 3. 精度和通信成本计算
            specific_requests = requests[:, self.env.NUM_SHARED_MODELS:]
            specific_deployed = deployment_probs[:, self.env.NUM_SHARED_MODELS:]
            
            # 为每个特定模型计算
            for specific_idx in range(self.env.NUM_SPECIFIC_MODELS):
                model_global_id = specific_idx + self.env.NUM_SHARED_MODELS
                req_rates = specific_requests[:, specific_idx]
                mask = req_rates > 1e-6

                if not np.any(mask):
                    continue

                # ===== 精度成本 =====
                model_accuracies = updated_accuracies[:, model_global_id]
                accuracy_min = self.env.accuracy_min[model_global_id]
                accuracy_deficits = np.maximum(0, accuracy_min - model_accuracies)
                
                # 只对有请求的节点计算精度成本
                accuracy_cost += np.sum(mask * req_rates * accuracy_deficits)

                # 计算依赖的分片数量
                dependency_indices = np.where(dependency_matrix[specific_idx] != 0)[0].astype(int)
                fragments_needed = 1 + len(dependency_indices)
                total_fragments_needed += np.sum(mask * req_rates) * fragments_needed

                # 计算分片可用性
                availability_probs = self.env._calculate_fragment_availability_numpy(
                    specific_idx, specific_deployed[:, specific_idx], deployment_probs, dependency_matrix
                )
                
                total_fragments_available += np.sum(
                    mask * req_rates * availability_probs['expected_fragments']
                )

                # 推理成本
                inference_cost += np.sum(
                    mask * req_rates * args.INFERENCE_COST_PER_MB * 
                    availability_probs['full_availability']
                )

                # 通信成本
                comm_costs = self._calculate_communication_costs_numpy(
                    mask, req_rates, availability_probs
                )
                communication_cost += comm_costs

            # 总成本
            total_cost = switching_cost + communication_cost + update_cost + inference_cost + accuracy_cost

            return total_cost

        except Exception as e:
            print(f"目标函数计算错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1e10

    # 添加辅助函数来计算通信成本
    def _calculate_communication_costs_numpy(self, mask, req_rates, availability_probs):
        # 完全命中：本地通信成本
        local_comm = np.sum(mask * req_rates * availability_probs['full_availability'] * args.DELAY_LOCAL)
        # 部分命中：云端通信成本
        partial_comm = np.sum(mask * req_rates * availability_probs['partial_availability'] * args.DELAY_CLOUD)
        # 完全缺失：云端通信成本
        cloud_comm = np.sum(mask * req_rates * availability_probs['no_availability'] * args.DELAY_CLOUD)
        return local_comm + partial_comm + cloud_comm