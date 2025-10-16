# import numpy as np
# from collections import deque

# class LRUCacheOptimizer:
#     def __init__(self, env, args, dependency_matrix, all_requests):
#         self.env = env
#         self.args = args
#         self.dependency_matrix = dependency_matrix
#         self.all_requests = all_requests
#         self.NUM_MODELS = env.NUM_MODELS

#         # specific/shared 数量
#         self.num_specific = env.NUM_SPECIFIC_MODELS
#         self.num_shared = env.NUM_SHARED_MODELS

#         # 初始化缓存队列和容量
#         self.edge_num = args.edge_num
#         self.cache_capacity = args.edge_cache_storage
#         self.model_sizes = env.cloud_manager.model_sizes
#         self.cache_queues = [deque() for _ in range(self.edge_num)]
#         self.current_usage = [0 for _ in range(self.edge_num)]

#         # 最近使用时间数组
#         self.last_used = np.full(self.NUM_MODELS, -1)

#     def update_last_used(self, t):
#         """只更新实际被请求的模型"""
#         node_requests = self.all_requests[t]  # shape: (edge_num, NUM_MODELS)
#         for node in range(self.edge_num):
#             used_models = np.where(node_requests[node] > 0)[0]
#             self.last_used[used_models] = t

#     def compute_update_mask(self, new_cache_state, Update_flag=False):
#         if Update_flag:
#             return np.zeros_like(new_cache_state, dtype=bool)

#         if new_cache_state.ndim == 1:
#             new_cache_state = new_cache_state.reshape(1, -1)

#         num_nodes, total_models = new_cache_state.shape
#         assert total_models == self.num_shared + self.num_specific, \
#             f"模型总数不匹配: 期望 {self.num_shared + self.num_specific}, 实际 {total_models}"

#         update_mask = np.zeros_like(new_cache_state, dtype=bool)

#         for node in range(num_nodes):
#             cached_models = np.where(new_cache_state[node] == 1)[0]
#             if len(cached_models) == 0:
#                 continue

#             # 随机选择至少一个模型进行更新
#             num_selected = np.random.randint(1, len(cached_models) + 1)
#             selected = np.random.choice(cached_models, num_selected, replace=False)

#             for m in selected:
#                 update_mask[node, m] = True

#                 # 只传播特定模型的依赖
#                 if m >= self.num_shared:
#                     specific_idx = m - self.num_shared
#                     if specific_idx < self.dependency_matrix.shape[0]:
#                         deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
#                         for j in deps:
#                             if j < self.num_shared:
#                                 update_mask[node, j] = True

#         return update_mask

#     def lru_optimize(self, t):
#         # 更新最近使用时间
#         self.update_last_used(t)

#         new_cache_state = np.zeros((self.edge_num, self.NUM_MODELS), dtype=int)

#         # 更新当前缓存使用量
#         self.current_usage = [sum(self.model_sizes[m] for m in q) for q in self.cache_queues]

#         for node in range(self.edge_num):
#             requested_models = np.where(self.all_requests[t, node, :] > 0)[0]

#             for m in requested_models:
#                 if m in self.cache_queues[node]:
#                     # 命中 → 移到队尾
#                     self.cache_queues[node].remove(m)
#                     self.cache_queues[node].append(m)
#                 else:
#                     # ========== 修改点：容量检查时考虑依赖 ==========
#                     extra_size = 0
#                     deps_to_add = []
#                     if m >= self.num_shared:  # 只对特定模型检查依赖
#                         specific_idx = m - self.num_shared
#                         deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
#                         for d in deps:
#                             if d not in self.cache_queues[node]:
#                                 extra_size += self.model_sizes[d]
#                                 deps_to_add.append(d)

#                     total_size = self.model_sizes[m] + extra_size

#                     # 驱逐直到腾出足够空间
#                     while self.current_usage[node] + total_size > self.cache_capacity:
#                         if self.cache_queues[node]:
#                             evict = self.cache_queues[node].popleft()
#                             self.current_usage[node] -= self.model_sizes[evict]
#                         else:
#                             break

#                     # 插入依赖的共享模型
#                     for d in deps_to_add:
#                         if self.current_usage[node] + self.model_sizes[d] <= self.cache_capacity:
#                             self.cache_queues[node].append(d)
#                             self.current_usage[node] += self.model_sizes[d]

#                     # 插入当前特定模型（或共享模型本身）
#                     if self.current_usage[node] + self.model_sizes[m] <= self.cache_capacity:
#                         self.cache_queues[node].append(m)
#                         self.current_usage[node] += self.model_sizes[m]

#             # 更新缓存状态矩阵
#             for cached_model in self.cache_queues[node]:
#                 new_cache_state[node, cached_model] = 1

#         Update_flag = self.args.Update_flag
#         update_mask = self.compute_update_mask(new_cache_state, Update_flag=Update_flag)
#         return new_cache_state, update_mask

import numpy as np
from collections import deque

class LRUCacheOptimizer:
    def __init__(self, env, args, dependency_matrix, all_requests):
        self.env = env
        self.args = args
        self.dependency_matrix = dependency_matrix
        self.all_requests = all_requests
        self.NUM_MODELS = env.NUM_MODELS

        # specific/shared 数量
        self.num_specific = env.NUM_SPECIFIC_MODELS
        self.num_shared = env.NUM_SHARED_MODELS

        # 初始化缓存队列和容量
        self.edge_num = args.edge_num
        self.cache_capacity = args.edge_cache_storage
        self.model_sizes = env.cloud_manager.model_sizes
        self.cache_queues = [deque() for _ in range(self.edge_num)]
        self.current_usage = [0 for _ in range(self.edge_num)]

        # 最近使用时间数组
        self.last_used = np.full(self.NUM_MODELS, -1)
        
        # 添加Shared_flag标识
        self.Shared_flag = getattr(args, 'Shared_flag', True)

    def get_effective_model_size(self, model_idx):
        """
        获取模型的有效大小
        如果Shared_flag=False且是特定模型，需要加上依赖的共享模型大小
        """
        if not self.Shared_flag and model_idx >= self.num_shared:
            # 不考虑共享时，特定模型大小 = 自身大小 + 依赖的共享模型大小
            specific_idx = model_idx - self.num_shared
            shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
            total_size = self.model_sizes[model_idx]
            for dep_idx in shared_deps:
                total_size += self.model_sizes[dep_idx]
            return total_size
        else:
            # 考虑共享时，返回原始大小
            return self.model_sizes[model_idx]

    def update_last_used(self, t):
        """只更新实际被请求的模型"""
        node_requests = self.all_requests[t]  # shape: (edge_num, NUM_MODELS)
        for node in range(self.edge_num):
            used_models = np.where(node_requests[node] > 0)[0]
            self.last_used[used_models] = t

    def compute_update_mask(self, new_cache_state, Update_flag=False):
        if Update_flag:
            return np.zeros_like(new_cache_state, dtype=bool)

        if new_cache_state.ndim == 1:
            new_cache_state = new_cache_state.reshape(1, -1)

        num_nodes, total_models = new_cache_state.shape
        assert total_models == self.num_shared + self.num_specific, \
            f"模型总数不匹配: 期望 {self.num_shared + self.num_specific}, 实际 {total_models}"

        update_mask = np.zeros_like(new_cache_state, dtype=bool)

        for node in range(num_nodes):
            cached_models = np.where(new_cache_state[node] == 1)[0]
            if len(cached_models) == 0:
                continue

            # 随机选择至少一个模型进行更新
            num_selected = np.random.randint(1, len(cached_models) + 1)
            selected = np.random.choice(cached_models, num_selected, replace=False)

            for m in selected:
                # 只更新特定模型，不更新共享模型
                if m >= self.num_shared:  # 确保只选择特定模型
                    update_mask[node, m] = True

        return update_mask

    def lru_optimize(self, t):
        # 更新最近使用时间
        self.update_last_used(t)

        new_cache_state = np.zeros((self.edge_num, self.NUM_MODELS), dtype=int)

        # 更新当前缓存使用量 - 使用有效大小
        if self.Shared_flag:
            self.current_usage = [sum(self.model_sizes[m] for m in q) for q in self.cache_queues]
        else:
            self.current_usage = [sum(self.get_effective_model_size(m) for m in q) for q in self.cache_queues]

        for node in range(self.edge_num):
            requested_models = np.where(self.all_requests[t, node, :] > 0)[0]

            for m in requested_models:
                if m in self.cache_queues[node]:
                    # 命中 → 移到队尾
                    self.cache_queues[node].remove(m)
                    self.cache_queues[node].append(m)
                else:
                    # 未命中 → 需要插入
                    if self.Shared_flag:
                        # 考虑共享：需要检查并添加依赖的共享模型
                        extra_size = 0
                        deps_to_add = []
                        if m >= self.num_shared:  # 只对特定模型检查依赖
                            specific_idx = m - self.num_shared
                            deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
                            for d in deps:
                                if d not in self.cache_queues[node]:
                                    extra_size += self.model_sizes[d]
                                    deps_to_add.append(d)

                        total_size = self.model_sizes[m] + extra_size

                        # 驱逐直到腾出足够空间
                        while self.current_usage[node] + total_size > self.cache_capacity:
                            if self.cache_queues[node]:
                                evict = self.cache_queues[node].popleft()
                                self.current_usage[node] -= self.model_sizes[evict]
                            else:
                                break

                        # 插入依赖的共享模型
                        for d in deps_to_add:
                            if self.current_usage[node] + self.model_sizes[d] <= self.cache_capacity:
                                self.cache_queues[node].append(d)
                                self.current_usage[node] += self.model_sizes[d]

                        # 插入当前模型
                        if self.current_usage[node] + self.model_sizes[m] <= self.cache_capacity:
                            self.cache_queues[node].append(m)
                            self.current_usage[node] += self.model_sizes[m]
                    else:
                        # 不考虑共享：只缓存特定模型，使用有效大小
                        # 共享模型不应该被单独请求，跳过
                        if m < self.num_shared:
                            continue
                        
                        total_size = self.get_effective_model_size(m)

                        # 驱逐直到腾出足够空间
                        while self.current_usage[node] + total_size > self.cache_capacity:
                            if self.cache_queues[node]:
                                evict = self.cache_queues[node].popleft()
                                self.current_usage[node] -= self.get_effective_model_size(evict)
                            else:
                                break

                        # 插入当前特定模型（大小已包含共享部分）
                        if self.current_usage[node] + total_size <= self.cache_capacity:
                            self.cache_queues[node].append(m)
                            self.current_usage[node] += total_size

            # 更新缓存状态矩阵
            for cached_model in self.cache_queues[node]:
                new_cache_state[node, cached_model] = 1

        Update_flag = self.args.Update_flag
        update_mask = self.compute_update_mask(new_cache_state, Update_flag=Update_flag)
        return new_cache_state, update_mask