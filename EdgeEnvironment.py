import numpy as np
import torch.nn as nn
import torch
import re
import os
import torchvision.transforms as transforms
from scipy.stats import entropy
from torch import nn
import random
import math
from PIL import Image
import os
from parameters import args_parser
args = args_parser()

class EdgeEnvironment:
    def __init__(self, cloud_edge_coordinator, cloud_manager, request_records,
                 NUM_EDGE_NODES, Shared_flag, storage_dir: str = "./model_data/cloud_registry_model"):
        self.cloud_edge_coordinator = cloud_edge_coordinator
        self.cloud_manager = cloud_manager
        self.request_records = request_records
        self.NUM_SHARED_MODELS = cloud_manager.num_shared
        self.NUM_EDGE_NODES = NUM_EDGE_NODES
        self.NUM_SPECIFIC_MODELS = cloud_manager.num_specific
        self.NUM_MODELS = self.NUM_SPECIFIC_MODELS + self.NUM_SHARED_MODELS
        self.storage_dir = storage_dir
        self.cache_state = np.zeros((NUM_EDGE_NODES, self.NUM_MODELS))
        self.pre_cache_state = np.zeros((NUM_EDGE_NODES, self.NUM_MODELS))
        self.initial_acc = self.initial_accuracies()
        self.accuracies = self.initial_acc.copy()
        self.accuracy_min = self.initial_acc * np.random.uniform(0.85, 0.95, self.NUM_MODELS)
        self.Shared_flag = Shared_flag
        
        # ===== 新增：根据Shared_flag计算有效模型大小 =====
        self.effective_model_sizes = self._compute_effective_model_sizes()
        
        self.power_history = []
        self.communication_cost_history =[]
        self.total_objective_history = []
        self.accuracy_penalty_history = []
        self.delay_penalty_history = []
        self.last_trained = [0] * self.NUM_MODELS
        self.current_step = 0
        self.total_time_slots = args.time_slot
        self.last_update_time = np.zeros(self.NUM_MODELS, dtype=int)
        
        self.age_state = self._get_env_age_state()
        self.max_age = args.max_age
        self.age_decay_factor = args.age_decay_factor
        self.current_time = 0
        
        # 成本相关张量 - 使用有效模型大小
        self.model_switch_costs = [self.get_switch_cost(mid) for mid in range(self.NUM_MODELS)] 
        self.model_update_costs = [self.get_update_cost(mid) for mid in range(self.NUM_MODELS)]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_effective_model_sizes(self):
        original_sizes = np.array(self.cloud_manager.model_sizes)
        effective_sizes = original_sizes.copy()
        
        if not self.Shared_flag:
            # 非共享模式：特定模型需要包含依赖的共享模型大小
            dependency_matrix = self.cloud_manager.dependency_matrix
            
            for specific_idx in range(self.NUM_SPECIFIC_MODELS):
                model_global_id = specific_idx + self.NUM_SHARED_MODELS
                
                # 获取依赖的共享模型索引
                dependency_indices = np.where(dependency_matrix[specific_idx] != 0)[0]
                
                # 计算依赖的共享模型总大小
                dependent_size = np.sum(original_sizes[dependency_indices])
                
                # 特定模型的有效大小 = 自身大小 + 依赖的共享模型大小
                effective_sizes[model_global_id] = original_sizes[model_global_id] + dependent_size
            
            for specific_idx in range(self.NUM_SPECIFIC_MODELS):
                model_global_id = specific_idx + self.NUM_SHARED_MODELS
                # print(f"模型 {model_global_id}: 原始大小={original_sizes[model_global_id]:.2f}MB, "
                    #   f"有效大小={effective_sizes[model_global_id]:.2f}MB")
        else:
            print(f"\n===== 共享模式：使用原始模型大小 =====")
        
        return effective_sizes

    def _get_effective_size(self, model_id):
        """获取模型的有效大小（考虑Shared_flag）"""
        return self.effective_model_sizes[model_id]

    def _get_env_age_state(self):
        """从环境中获取年龄状态"""
        if hasattr(self, 'age_state') and self.age_state is not None:
            age_state = np.array(self.age_state)
            if age_state.ndim == 1:
                return age_state.reshape(self.NUM_EDGE_NODES, self.NUM_MODELS)
            return age_state
        return np.zeros((self.NUM_EDGE_NODES, self.NUM_MODELS))

    def get_state_vector(self, t): 
        try:
            state = []
            cache_state = np.array(self.cache_state)
            if cache_state.ndim == 1:
                cache_state = cache_state.reshape(self.NUM_EDGE_NODES, self.NUM_MODELS)
            state.extend(cache_state.flatten())
            
            if hasattr(self, 'age_state'):
                age_state = np.array(self.age_state)
                if age_state.ndim == 1:
                    age_state = age_state.reshape(self.NUM_EDGE_NODES, self.NUM_MODELS)
                state.extend(age_state.flatten())
            else:
                state.extend(np.zeros(self.NUM_EDGE_NODES * self.NUM_MODELS))
            
            if hasattr(self, 'accuracies'):
                accuracies = np.array(self.accuracies)
                if accuracies.ndim == 1:
                    accuracies_tiled = np.tile(accuracies, (self.NUM_EDGE_NODES, 1))
                    state.extend(accuracies_tiled.flatten())
                else:
                    state.extend(accuracies.flatten())
            else:
                state.extend(np.zeros(self.NUM_EDGE_NODES * self.NUM_MODELS))
            
            # ===== 修改：使用有效模型大小 =====
            state.extend(self.effective_model_sizes.flatten())

            # ===== 修改：存储使用计算使用有效大小 =====
            for s in range(self.NUM_EDGE_NODES):
                storage_used = float(np.sum(self.effective_model_sizes * cache_state[s]))
                state.append(storage_used / float(args.edge_cache_storage))

            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            return state_tensor
            
        except Exception as e:
            print(f"获取状态向量时出错 (t={t}): {e}")
            raise

    def calculate_hit_rate(self, deployment, requests, dependency_matrix):
        """计算基于模型可用性的命中率，考虑模型依赖关系"""
        total_requests = np.sum(requests)
        if total_requests == 0:
            return 0.0
        
        num_shared = self.NUM_SHARED_MODELS
        num_specific = self.NUM_SPECIFIC_MODELS
        
        total_models_needed = 0
        total_models_obtained = 0
        
        for b in range(self.NUM_EDGE_NODES):
            for m in range(num_specific):
                specific_idx = num_shared + m
                request_count = requests[b, specific_idx]
                if request_count == 0:
                    continue
                
                # ===== 修改：根据Shared_flag调整命中率计算 =====
                if self.Shared_flag:
                    # 共享模式：需要检查特定模型和依赖的共享模型
                    num_models_needed = 1 + np.sum(dependency_matrix[m])
                    total_models_needed += request_count * num_models_needed
                    
                    models_obtained = 0
                    if deployment[b, specific_idx]:
                        models_obtained += 1
                    
                    for s in range(num_shared):
                        if dependency_matrix[m, s] and deployment[b, s]:
                            models_obtained += 1
                    
                    total_models_obtained += request_count * models_obtained
                else:
                    # 非共享模式：只需要特定模型（已包含依赖）
                    total_models_needed += request_count
                    if deployment[b, specific_idx]:
                        total_models_obtained += request_count
        
        if total_models_needed == 0:
            return 0.0
        
        return total_models_obtained / total_models_needed

    def determine_update_mask(self, t, deployment_decisions, update_decisions):
        update_mask = np.zeros_like(deployment_decisions, dtype=bool)
        
        if deployment_decisions.ndim == 1:
            deployment_decisions = deployment_decisions.reshape(self.NUM_EDGE_NODES, self.NUM_MODELS)
        if update_decisions.ndim == 1:
            update_decisions = update_decisions.reshape(self.NUM_EDGE_NODES, self.NUM_MODELS)
        
        update_mask = (deployment_decisions.astype(bool)) & (update_decisions.astype(bool))
        
        # ===== 修改：只在共享模式下同步更新依赖的共享模型 =====
        if self.Shared_flag and hasattr(self, 'cloud_manager') and hasattr(self.cloud_manager, 'dependency_matrix'):
            dependency_matrix = self.cloud_manager.dependency_matrix

            for specific_idx in range(self.NUM_SPECIFIC_MODELS):
                global_idx = specific_idx + self.NUM_SHARED_MODELS
                specific_update_mask = update_mask[:, global_idx]

                if np.any(specific_update_mask):
                    dependencies = np.where(dependency_matrix[global_idx, :self.NUM_SHARED_MODELS] > 0)[0]
                    for dep in dependencies:
                        update_mask[:, dep] = update_mask[:, dep] | specific_update_mask

        return update_mask

    def enforce_storage_constraints(self, deployment_prob, model_sizes, edge_storage_capacity):
        # ===== 修改：使用有效模型大小 =====
        effective_sizes = self.effective_model_sizes
        cache_prob = np.array(deployment_prob).reshape(self.NUM_EDGE_NODES, self.NUM_MODELS)

        deployment = np.zeros_like(cache_prob, dtype=int)

        for server in range(self.NUM_EDGE_NODES):
            probs = cache_prob[server]
            sorted_indices = np.argsort(-probs)

            storage_used = 0.0
            selected = []

            for idx in sorted_indices:
                size = effective_sizes[idx]
                if storage_used + size <= edge_storage_capacity:
                    selected.append(idx)
                    storage_used += size

            deployment[server, selected] = 1

        return deployment

    def initial_accuracies(self):
        """使用现有ID信息初始化模型精度"""
        NUM_MODELS = len(self.cloud_manager.available_models)
        initial_accuracies = np.zeros(NUM_MODELS)
        model_size = np.zeros(NUM_MODELS)
        
        print("\n===== 使用ID信息初始化模型精度 =====")
        
        for model_info in self.cloud_manager.available_models:
            model_id = model_info['id']
            accuracy = model_info.get('accuracy')
            model_size = model_info.get('size')
            if accuracy is not None:
                initial_accuracies[model_id] = accuracy
        return initial_accuracies

    def update_accuracies(self, t, update_mask, previous_cache_state=None):
        if previous_cache_state is None:
            previous_cache_state = self.cache_state.copy()
        elif previous_cache_state.ndim == 1:
            previous_cache_state = previous_cache_state.reshape(self.NUM_EDGE_NODES, self.NUM_MODELS)
        
        update_mask = np.array(update_mask).reshape(self.NUM_EDGE_NODES, self.NUM_MODELS)
        
        if not hasattr(self, 'age_state') or self.age_state is None:
            self.age_state = np.zeros((self.NUM_EDGE_NODES, self.NUM_MODELS))
        elif self.age_state.ndim == 1:
            self.age_state = self.age_state.reshape(self.NUM_EDGE_NODES, self.NUM_MODELS)
        
        if isinstance(self.initial_acc, list):
            initial_acc_array = np.array(self.initial_acc)
        else:
            initial_acc_array = self.initial_acc
        
        if not hasattr(self, 'accuracies') or self.accuracies is None:
            self.accuracies = initial_acc_array.copy()
        elif self.accuracies.ndim != 1 or len(self.accuracies) != self.NUM_MODELS:
            self.accuracies = initial_acc_array.copy()
        
        new_age_state = self.age_state.copy()
        new_acc_state = self.accuracies.copy()
        
        for i in range(self.NUM_EDGE_NODES):
            for j in range(self.NUM_MODELS):
                if update_mask[i, j] > 0:
                    new_age_state[i, j] = 0
                elif self.cache_state[i, j] > 0:
                    if previous_cache_state[i, j] > 0:
                        new_age_state[i, j] += 1
                    else:
                        new_age_state[i, j] = 0
                else:
                    new_age_state[i, j] = 0
        
        for j in range(self.NUM_MODELS):
            updated_nodes = [i for i in range(self.NUM_EDGE_NODES) if update_mask[i, j] > 0]
            
            if len(updated_nodes) > 0:
                new_acc_state[j] = initial_acc_array[j]
            else:
                deployed_ages = [new_age_state[i, j] for i in range(self.NUM_EDGE_NODES) 
                            if self.cache_state[i, j] > 0]
                
                if deployed_ages:
                    min_age = min(deployed_ages)
                    age_ratio = np.clip(min_age / self.max_age, 0.0, 1.0)
                    decay_factor = np.exp(-self.age_decay_factor * age_ratio)
                    new_acc_state[j] = initial_acc_array[j] * decay_factor
        
        new_acc_state = np.clip(new_acc_state, 0.1, 1.0)
        
        self.age_state = new_age_state
        self.accuracies = new_acc_state
        
    def get_switch_cost(self, model_id):
        # ===== 修改：使用有效模型大小 =====
        switch_cost = self._get_effective_size(model_id) * args.SWITCHING_COST_PER_MB
        return switch_cost

    def get_update_cost(self, model_id):
        # ===== 修改：使用有效模型大小 =====
        return self._get_effective_size(model_id) * args.UPDATE_COST_PER_MB
         
    def get_communication_cost(self, model_id):
        # ===== 修改：使用有效模型大小 =====
        return self._get_effective_size(model_id) * args.COMMUNICATION_COST_PER_MB
    
    def get_model_accuracy_cost(self, model_id):
        # ===== 修改：使用有效模型大小 =====
        return self._get_effective_size(model_id) * args.ACCURACY_COST_PER_MB

    def get_inference_cost(self, t, model_id, node_id=None):
        rho_m = self.cloud_manager.rho_m_values[model_id]
        input_size = self.calculate_avg_input_size(t, model_id, node_id)
        
        node_factor = 1.0
        if node_id is not None and hasattr(self, 'node_performance_factors'):
            node_factor = self.node_performance_factors.get(node_id, 1.0)
        
        base_cost = rho_m * input_size * args.INFERENCE_COST_PER_MB
        return base_cost / max(0.1, node_factor)

    def calculate_avg_input_size(self, t, model_id, node_id=None):
        total_size = 0.0
        request_count = 0
        
        if node_id is not None:
            request_key = (t, node_id, model_id)
            if request_key in self.request_records:
                image_paths = self.request_records[request_key]
                for path in image_paths:
                    total_size += self.get_image_size(path)
                    request_count += 1
        else:
            for n_id in range(self.NUM_EDGE_NODES):
                request_key = (t, n_id, model_id)
                if request_key in self.request_records:
                    image_paths = self.request_records[request_key]
                    for path in image_paths:
                        total_size += self.get_image_size(path)
                        request_count += 1
        
        if request_count > 0:
            return total_size / request_count
        else:
            return args.DEFAULT_INPUT_SIZE_MB
            
    def get_image_size(self, image_path):
        try:
            with Image.open(image_path) as img:
                width, height = img.size
            
            image_size_bytes = width * height * 3
            input_size_mb = image_size_bytes / (1024 * 1024)
            return input_size_mb
        except Exception as e:
            print(f"无法读取图片: {e}")
            return None
    
    def evaluate_performance(self, t, requests, dependency_matrix, prev_deployment=None, update_decision=None):
        assert self.accuracies.ndim == 1, f"accuracies must be 1D, got shape {self.accuracies.shape}"
        assert len(self.accuracies) == self.NUM_MODELS, \
            f"accuracies length mismatch: {len(self.accuracies)} != {self.NUM_MODELS}"
        
        total_fragments_needed = 0.0
        total_fragments_available = 0.0
        cloud_access = 0.0
        total_requests = 0.0

        switching_cost = 0.0
        communication_cost = 0.0
        update_cost = 0.0
        inference_cost = 0.0
        accuracy_cost = 0.0

        # ===== 修改：切换成本使用有效模型大小 =====
        if prev_deployment is not None:
            deployment_changes = np.abs(self.cache_state - prev_deployment)
            for model_id in range(self.NUM_MODELS):
                model_switch_cost = self.get_switch_cost(model_id)
                switching_cost += np.sum(deployment_changes[:, model_id]) * model_switch_cost

        # ===== 修改：更新成本使用有效模型大小 =====
        if update_decision is not None:
            update_decision = np.array(update_decision)
            if update_decision.ndim == 1:
                update_decision = update_decision.reshape(self.NUM_EDGE_NODES, self.NUM_MODELS)
            
            for model_id in range(self.NUM_MODELS):
                model_update_cost = self.get_update_cost(model_id)
                nodes_needing_update = np.sum(update_decision[:, model_id])
                update_cost += nodes_needing_update * model_update_cost

        # ===== 请求处理 =====
        for node_id in range(self.NUM_EDGE_NODES):
            for specific_model_idx in range(self.NUM_SPECIFIC_MODELS):
                model_global_id = specific_model_idx + self.NUM_SHARED_MODELS
                
                if model_global_id >= self.NUM_MODELS:
                    print(f"[ERROR] Invalid model_global_id: {model_global_id} >= {self.NUM_MODELS}")
                    continue
                
                req_rate = requests[node_id, model_global_id]

                if req_rate <= 0:
                    continue

                total_requests += req_rate

                model_accuracy = self.accuracies[model_global_id]
                model_accuracy_min = self.accuracy_min[model_global_id]
                accuracy_deficit = max(0.0, model_accuracy_min - model_accuracy)
                accuracy_cost += req_rate * accuracy_deficit * self.get_model_accuracy_cost(model_global_id)

                # ===== 修改：根据Shared_flag调整分块逻辑 =====
                if self.Shared_flag:
                    # 共享模式：需要检查依赖
                    dependency_indices = np.where(dependency_matrix[specific_model_idx] != 0)[0]
                    fragments_needed = 1 + len(dependency_indices)
                    total_fragments_needed += req_rate * fragments_needed

                    specific_cached = self.cache_state[node_id, model_global_id]
                    shared_cached = [self.cache_state[node_id, idx] for idx in dependency_indices]

                    all_fragments_available = specific_cached and all(shared_cached)
                    any_fragment_available = specific_cached or any(shared_cached)
                    partial_hit = any_fragment_available and not all_fragments_available

                    fragments_available = specific_cached + sum(shared_cached)
                    total_fragments_available += req_rate * fragments_available
                else:
                    # 非共享模式：只需检查特定模型
                    fragments_needed = 1
                    total_fragments_needed += req_rate * fragments_needed

                    specific_cached = self.cache_state[node_id, model_global_id]
                    all_fragments_available = specific_cached
                    partial_hit = False
                    fragments_available = specific_cached
                    total_fragments_available += req_rate * fragments_available

                # 推理成本
                if all_fragments_available:
                    model_inference_cost = args.INFERENCE_COST_PER_MB
                    inference_cost += req_rate * model_inference_cost

                # 通信成本
                if all_fragments_available:
                    communication_cost += req_rate * args.DELAY_LOCAL
                elif partial_hit:
                    communication_cost += req_rate * args.DELAY_CLOUD
                    cloud_access += req_rate
                else:
                    communication_cost += req_rate * args.DELAY_CLOUD
                    cloud_access += req_rate

        hit_rate = total_fragments_available / total_fragments_needed if total_fragments_needed > 0 else 0.0
        cloud_access_rate = cloud_access / total_requests if total_requests > 0 else 0.0

        accuracy_weighted_sum = 0.0
        accuracy_sum = 0.0
        for node_id in range(self.NUM_EDGE_NODES):
            for specific_model_idx in range(self.NUM_SPECIFIC_MODELS):
                model_global_id = specific_model_idx + self.NUM_SHARED_MODELS
                
                if model_global_id >= self.NUM_MODELS:
                    continue
                
                req_rate = requests[node_id, model_global_id]
                if req_rate > 0:
                    accuracy = self.accuracies[model_global_id]
                    accuracy_weighted_sum += req_rate * accuracy
                    accuracy_sum += req_rate
        
        avg_accuracy = accuracy_weighted_sum / accuracy_sum if accuracy_sum > 0 else 0.0

        total_cost = switching_cost + communication_cost + update_cost + inference_cost + accuracy_cost

        return (
            hit_rate,
            avg_accuracy,
            cloud_access_rate,
            total_cost,
            switching_cost,
            communication_cost,
            update_cost,
            inference_cost,
            accuracy_cost
        )
        
    def _calculate_fragment_availability_numpy(self, specific_idx, specific_deployed, deployed_surrogate, dependency_matrix):
        dependency_indices = np.where(dependency_matrix[specific_idx] != 0)[0].astype(int)
        
        if len(dependency_indices) > 0:
            shared_probs = deployed_surrogate[:, dependency_indices]
            full_availability = specific_deployed * np.prod(shared_probs, axis=1)
        else:
            full_availability = specific_deployed
        
        if len(dependency_indices) > 0:
            any_shared_available = 1 - np.prod(1 - shared_probs, axis=1)
            partial_availability = any_shared_available * (1 - full_availability)
        else:
            partial_availability = np.zeros_like(full_availability)
        
        no_availability = 1 - full_availability - partial_availability
        
        expected_fragments = specific_deployed
        if len(dependency_indices) > 0:
            expected_fragments += np.sum(shared_probs, axis=1)
        
        return {
            'full_availability': full_availability,
            'partial_availability': partial_availability,
            'no_availability': no_availability,
            'expected_fragments': expected_fragments
        }
        
    def reset(self):
        self.cache_state = np.zeros((self.NUM_EDGE_NODES, self.NUM_MODELS))
        self.age_state = np.zeros((self.NUM_EDGE_NODES, self.NUM_MODELS))
        self.initial_acc = self.initial_accuracies()
        self.accuracies = self.initial_acc.copy()
        self.power_history = []
        self.placement_cost_history = []
        self.communication_cost_history =[]
        self.inference_cost_history = []
        self.accuracy_penalty_history = []
        self.delay_penalty_history = []
        self.total_objective_history = []
        self.current_step = 0
        
        self.last_update_time = np.zeros(self.NUM_MODELS, dtype=int)
        self.current_time = 0