# import numpy as np
# import random

# class RandomCachingPlacement:
#     def __init__(self, env, args):
#         self.env = env
#         self.args = args
#         self.model_sizes = self.env.cloud_manager.model_sizes
#         self.num_shared = self.env.NUM_SHARED_MODELS
#         self.num_specific = self.env.NUM_SPECIFIC_MODELS
#         self.dependency_matrix = self.env.cloud_manager.dependency_matrix
#         self.M = len(self.model_sizes)  # 总模型数量
        
#     def update_cache(self, prev_state):
#         B, M = prev_state.shape  # 基站数量，模型数量
#         new_state = np.zeros((B, M), dtype=int)
        
#         for b in range(B):
#             available_capacity = self.args.edge_cache_storage

#             # 只随机排列“特定模型”，共享模型不直接参与选择
#             candidate_models = list(range(self.num_shared, M))
#             random.shuffle(candidate_models)

#             # 按随机顺序尝试添加特定模型
#             for m in candidate_models:
#                 specific_idx = m - self.num_shared
#                 model_size = self.model_sizes[m]

#                 # 计算依赖的共享模型额外大小（仅当未被选中时才计算）
#                 shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
#                 extra_size = sum(
#                     self.model_sizes[dep]
#                     for dep in shared_deps
#                     if new_state[b, dep] == 0
#                 )

#                 total_size = model_size + extra_size

#                 if total_size <= available_capacity:
#                     # 添加依赖的共享模型
#                     for dep in shared_deps:
#                         if new_state[b, dep] == 0:
#                             new_state[b, dep] = 1
#                             available_capacity -= self.model_sizes[dep]

#                     # 添加当前特定模型
#                     new_state[b, m] = 1
#                     available_capacity -= model_size

#                 # 如果容量不足以放下任何模型，提前退出
#                 if available_capacity < min(self.model_sizes):
#                     break
        
#         # 创建更新掩码
#         Update_flag = self.args.Update_flag
#         update_mask = self._create_update_mask(new_state, Update_flag=Update_flag)
        
#         return new_state, update_mask
    
#     def _create_update_mask(self, new_state, Update_flag=False):
#         B, M = new_state.shape
#         update_mask = np.zeros((B, M), dtype=bool)
#         if Update_flag:
#             return update_mask
        
#         for b in range(B):
#             cached_models = np.where(new_state[b] == 1)[0]
#             if len(cached_models) == 0:
#                 continue

#             # 随机选择要更新的模型
#             num_to_update = random.randint(1, len(cached_models))
#             models_to_update = np.random.choice(
#                 cached_models, size=num_to_update, replace=False
#             )
#             update_mask[b, models_to_update] = True

#             # 如果是特定模型，传播依赖到共享模型
#             for m in models_to_update:
#                 if m >= self.num_shared:
#                     specific_idx = m - self.num_shared
#                     shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
#                     update_mask[b, shared_deps] = True
        
#         return update_mask

import numpy as np
import random

class RandomCachingPlacement:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.model_sizes = self.env.cloud_manager.model_sizes
        self.num_shared = self.env.NUM_SHARED_MODELS
        self.num_specific = self.env.NUM_SPECIFIC_MODELS
        self.dependency_matrix = self.env.cloud_manager.dependency_matrix
        self.M = len(self.model_sizes)  # 总模型数量
        
        # 添加Shared_flag标识
        self.Shared_flag = getattr(args, 'Shared_flag', True)
        
        # 计算有效模型大小
        self.effective_model_sizes = self._compute_effective_model_sizes()
    
    def _compute_effective_model_sizes(self):
        """计算有效模型大小（考虑共享标识）"""
        effective_sizes = np.array(self.model_sizes, dtype=np.float64)
        
        if not self.Shared_flag:
            # 不考虑共享时，特定模型大小需要加上依赖的共享模型大小
            for specific_idx in range(self.num_specific):
                model_idx = specific_idx + self.num_shared
                shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
                for dep_idx in shared_deps:
                    effective_sizes[model_idx] += self.model_sizes[dep_idx]
        
        return effective_sizes
    
    def update_cache(self, prev_state):
        B, M = prev_state.shape  # 基站数量,模型数量
        new_state = np.zeros((B, M), dtype=int)
        
        if self.Shared_flag:
            # 考虑共享：需要单独缓存共享模型和特定模型
            for b in range(B):
                available_capacity = self.args.edge_cache_storage
                
                # 只随机排列"特定模型",共享模型不直接参与选择
                candidate_models = list(range(self.num_shared, M))
                random.shuffle(candidate_models)
                
                # 按随机顺序尝试添加特定模型
                for m in candidate_models:
                    specific_idx = m - self.num_shared
                    model_size = self.model_sizes[m]
                    
                    # 计算依赖的共享模型额外大小(仅当未被选中时才计算)
                    shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
                    extra_size = sum(
                        self.model_sizes[dep]
                        for dep in shared_deps
                        if new_state[b, dep] == 0
                    )
                    
                    total_size = model_size + extra_size
                    
                    if total_size <= available_capacity:
                        # 添加依赖的共享模型
                        for dep in shared_deps:
                            if new_state[b, dep] == 0:
                                new_state[b, dep] = 1
                                available_capacity -= self.model_sizes[dep]
                        
                        # 添加当前特定模型
                        new_state[b, m] = 1
                        available_capacity -= model_size
                    
                    # 如果容量不足以放下任何模型,提前退出
                    if available_capacity < min(self.model_sizes):
                        break
        else:
            # 不考虑共享：只缓存特定模型，使用有效大小
            for b in range(B):
                available_capacity = self.args.edge_cache_storage
                
                # 只考虑特定模型
                candidate_models = list(range(self.num_shared, M))
                random.shuffle(candidate_models)
                
                # 按随机顺序尝试添加特定模型
                for m in candidate_models:
                    model_size = self.effective_model_sizes[m]
                    
                    if model_size <= available_capacity:
                        # 只添加特定模型（大小已包含共享部分）
                        new_state[b, m] = 1
                        available_capacity -= model_size
                    
                    # 如果容量不足以放下最小的特定模型,提前退出
                    min_specific_size = min(self.effective_model_sizes[self.num_shared:])
                    if available_capacity < min_specific_size:
                        break
        
        # 创建更新掩码
        Update_flag = self.args.Update_flag
        update_mask = self._create_update_mask(new_state, Update_flag=Update_flag)
        
        return new_state, update_mask
    
    def _create_update_mask(self, new_state, Update_flag=False):
        B, M = new_state.shape
        update_mask = np.zeros((B, M), dtype=bool)
        
        if Update_flag:
            return update_mask
        
        for b in range(B):
            cached_models = np.where(new_state[b] == 1)[0]
            if len(cached_models) == 0:
                continue
            
            # 随机选择要更新的模型
            num_to_update = random.randint(1, len(cached_models))
            models_to_update = np.random.choice(
                cached_models, size=num_to_update, replace=False
            )
            
            update_mask[b, models_to_update] = True
            
            # # 只在考虑共享时传播依赖关系
            # if self.Shared_flag:
            #     # 如果是特定模型,传播依赖到共享模型
            #     for m in models_to_update:
            #         if m >= self.num_shared:
            #             specific_idx = m - self.num_shared
            #             shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
            #             update_mask[b, shared_deps] = True
        
        return update_mask