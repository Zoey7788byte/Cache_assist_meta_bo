# import numpy as np
# class GreedyCacheOptimizer:
#     def __init__(self, env, args, all_requests, window_size=20):
#         self.env = env
#         self.args = args
#         self.dependency_matrix = env.cloud_manager.dependency_matrix  # (num_specific, num_shared)
#         self.all_requests = all_requests
#         self.NUM_MODELS = env.NUM_MODELS
#         self.NUM_SHARED_MODELS = env.NUM_SHARED_MODELS
#         self.NUM_SPECIFIC_MODELS = env.NUM_SPECIFIC_MODELS
#         self.window_size = window_size
#         self.model_sizes = self.env.cloud_manager.model_sizes
#         self.edge_num = args.edge_num
#         self.cache_capacity = args.edge_cache_storage  

#     def compute_window_popularity(self, current_slot):
#         # 取滑动窗口
#         start_slot = max(0, current_slot - self.window_size)
#         window_requests = self.all_requests[start_slot:current_slot, :, :]

#         # 只计算特定模型的流行度
#         specific_popularity = np.sum(
#             window_requests[:, :, self.NUM_SHARED_MODELS:], axis=(0, 1)
#         )

#         # 构造整体流行度数组（共享模型流行度置零）
#         popularity = np.concatenate([
#             np.zeros(self.NUM_SHARED_MODELS), 
#             specific_popularity
#         ])

#         # 归一化
#         total_requests = np.sum(specific_popularity)
#         if total_requests > 0:
#             popularity = popularity / total_requests
#         else:
#             popularity = np.ones(self.NUM_MODELS) / self.NUM_MODELS

#         return popularity

#     def compute_new_cache_state(self, popularity):
#         new_cache_state = np.zeros((self.edge_num, self.NUM_MODELS), dtype=int)

#         for node in range(self.edge_num):
#             # 只考虑特定模型的效率
#             efficiency = []
#             for m in range(self.NUM_SHARED_MODELS, self.NUM_MODELS):
#                 if self.model_sizes[m] > 0:
#                     efficiency.append((m, popularity[m] / self.model_sizes[m]))

#             efficiency.sort(key=lambda x: x[1], reverse=True)

#             used_capacity = 0
#             selected_models = set()

#             for m, eff in efficiency:
#                 if m in selected_models:
#                     continue

#                 specific_idx = m - self.NUM_SHARED_MODELS
#                 shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]

#                 # 计算额外依赖大小（只在容量约束时考虑）
#                 total_size = self.model_sizes[m] + sum(
#                     self.model_sizes[dep] for dep in shared_deps if dep not in selected_models
#                 )

#                 if used_capacity + total_size > self.cache_capacity:
#                     continue

#                 # 添加依赖的共享模型（只因为容量原因需要）
#                 for dep in shared_deps:
#                     if dep not in selected_models:
#                         selected_models.add(dep)
#                         used_capacity += self.model_sizes[dep]
#                         new_cache_state[node, dep] = 1

#                 # 添加当前特定模型
#                 selected_models.add(m)
#                 used_capacity += self.model_sizes[m]
#                 new_cache_state[node, m] = 1

#         return new_cache_state

#     def compute_update_mask(self, new_cache_state, Update_flag=False):
#         if Update_flag:
#             return np.zeros_like(new_cache_state, dtype=bool)

#         self.edge_num, total_models = new_cache_state.shape
#         assert total_models == self.NUM_MODELS

#         update_mask = np.zeros_like(new_cache_state, dtype=bool)

#         for node in range(self.edge_num):
#             cached_models = np.where(new_cache_state[node] == 1)[0]
#             if len(cached_models) == 0:
#                 continue

#             # 随机选择至少一个模型
#             num_selected = np.random.randint(1, len(cached_models) + 1)
#             selected = np.random.choice(cached_models, num_selected, replace=False)

#             for m in selected:
#                 update_mask[node, m] = True

#                 # 只处理特定模型依赖关系
#                 if m >= self.NUM_SHARED_MODELS:
#                     specific_idx = m - self.NUM_SHARED_MODELS
#                     if specific_idx < self.dependency_matrix.shape[0]:
#                         shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
#                         for dep_idx in shared_deps:
#                             if dep_idx < self.NUM_SHARED_MODELS:
#                                 update_mask[node, dep_idx] = True

#         return update_mask

#     def greedy_optimize(self, t):
#         popularity = self.compute_window_popularity(t)
#         new_cache_state = self.compute_new_cache_state(popularity)
#         Update_flag = self.args.Update_flag
#         update_mask = self.compute_update_mask(new_cache_state, Update_flag=Update_flag)
#         return new_cache_state, update_mask

import numpy as np

class GreedyCacheOptimizer:
    def __init__(self, env, args, all_requests, window_size=20):
        self.env = env
        self.args = args
        self.dependency_matrix = env.cloud_manager.dependency_matrix  # (num_specific, num_shared)
        self.all_requests = all_requests
        self.NUM_MODELS = env.NUM_MODELS
        self.NUM_SHARED_MODELS = env.NUM_SHARED_MODELS
        self.NUM_SPECIFIC_MODELS = env.NUM_SPECIFIC_MODELS
        self.window_size = window_size
        self.model_sizes = self.env.cloud_manager.model_sizes
        self.edge_num = args.edge_num
        self.cache_capacity = args.edge_cache_storage
        # 添加Shared_flag标识
        self.Shared_flag = getattr(args, 'Shared_flag', True)

    def get_effective_model_size(self, model_idx):
        """
        获取模型的有效大小
        如果Shared_flag=False且是特定模型，需要加上依赖的共享模型大小
        """
        if not self.Shared_flag and model_idx >= self.NUM_SHARED_MODELS:
            # 不考虑共享时，特定模型大小 = 自身大小 + 依赖的共享模型大小
            specific_idx = model_idx - self.NUM_SHARED_MODELS
            shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
            total_size = self.model_sizes[model_idx]
            for dep_idx in shared_deps:
                total_size += self.model_sizes[dep_idx]
            return total_size
        else:
            # 考虑共享时，返回原始大小
            return self.model_sizes[model_idx]

    def compute_window_popularity(self, current_slot):
        # 取滑动窗口
        start_slot = max(0, current_slot - self.window_size)
        window_requests = self.all_requests[start_slot:current_slot, :, :]

        # 只计算特定模型的流行度
        specific_popularity = np.sum(
            window_requests[:, :, self.NUM_SHARED_MODELS:], axis=(0, 1)
        )

        # 构造整体流行度数组（共享模型流行度置零）
        popularity = np.concatenate([
            np.zeros(self.NUM_SHARED_MODELS), 
            specific_popularity
        ])

        # 归一化
        total_requests = np.sum(specific_popularity)
        if total_requests > 0:
            popularity = popularity / total_requests
        else:
            popularity = np.ones(self.NUM_MODELS) / self.NUM_MODELS

        return popularity

    def compute_new_cache_state(self, popularity):
        new_cache_state = np.zeros((self.edge_num, self.NUM_MODELS), dtype=int)

        for node in range(self.edge_num):
            # 只考虑特定模型的效率
            efficiency = []
            for m in range(self.NUM_SHARED_MODELS, self.NUM_MODELS):
                effective_size = self.get_effective_model_size(m)
                if effective_size > 0:
                    efficiency.append((m, popularity[m] / effective_size))

            efficiency.sort(key=lambda x: x[1], reverse=True)

            used_capacity = 0
            selected_models = set()

            for m, eff in efficiency:
                if m in selected_models:
                    continue

                specific_idx = m - self.NUM_SHARED_MODELS
                shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]

                if self.Shared_flag:
                    # 考虑共享：需要分别缓存共享模型和特定模型
                    total_size = self.model_sizes[m] + sum(
                        self.model_sizes[dep] for dep in shared_deps if dep not in selected_models
                    )

                    if used_capacity + total_size > self.cache_capacity:
                        continue

                    # 添加依赖的共享模型
                    for dep in shared_deps:
                        if dep not in selected_models:
                            selected_models.add(dep)
                            used_capacity += self.model_sizes[dep]
                            new_cache_state[node, dep] = 1

                    # 添加当前特定模型
                    selected_models.add(m)
                    used_capacity += self.model_sizes[m]
                    new_cache_state[node, m] = 1
                else:
                    # 不考虑共享：只缓存特定模型，但大小包含了共享部分
                    total_size = self.get_effective_model_size(m)

                    if used_capacity + total_size > self.cache_capacity:
                        continue

                    # 只添加特定模型（共享模型不单独缓存）
                    selected_models.add(m)
                    used_capacity += total_size
                    new_cache_state[node, m] = 1

        return new_cache_state

    # def compute_update_mask(self, new_cache_state, Update_flag=False):
    #     if Update_flag:
    #         return np.zeros_like(new_cache_state, dtype=bool)

    #     self.edge_num, total_models = new_cache_state.shape
    #     assert total_models == self.NUM_MODELS

    #     update_mask = np.zeros_like(new_cache_state, dtype=bool)

    #     for node in range(self.edge_num):
    #         cached_models = np.where(new_cache_state[node] == 1)[0]
    #         if len(cached_models) == 0:
    #             continue

    #         # 随机选择至少一个模型
    #         num_selected = np.random.randint(1, len(cached_models) + 1)
    #         selected = np.random.choice(cached_models, num_selected, replace=False)

    #         for m in selected:
    #             update_mask[node, m] = True

    #             # 只在考虑共享时处理依赖关系
    #             if self.Shared_flag and m >= self.NUM_SHARED_MODELS:
    #                 specific_idx = m - self.NUM_SHARED_MODELS
    #                 if specific_idx < self.dependency_matrix.shape[0]:
    #                     shared_deps = np.where(self.dependency_matrix[specific_idx, :] > 0)[0]
    #                     for dep_idx in shared_deps:
    #                         if dep_idx < self.NUM_SHARED_MODELS:
    #                             update_mask[node, dep_idx] = True

    #     return update_mask
    def compute_update_mask(self, new_cache_state, Update_flag=False):
        if Update_flag:
            return np.zeros_like(new_cache_state, dtype=bool)

        self.edge_num, total_models = new_cache_state.shape
        assert total_models == self.NUM_MODELS

        update_mask = np.zeros_like(new_cache_state, dtype=bool)

        for node in range(self.edge_num):
            cached_models = np.where(new_cache_state[node] == 1)[0]
            if len(cached_models) == 0:
                continue

            # 随机选择至少一个模型
            num_selected = np.random.randint(1, len(cached_models) + 1)
            selected = np.random.choice(cached_models, num_selected, replace=False)

            for m in selected:
                update_mask[node, m] = True

                # 移除共享模型的依赖更新逻辑
                # 现在只更新特定模型，不更新共享模型

        return update_mask

    def greedy_optimize(self, t):
        popularity = self.compute_window_popularity(t)
        new_cache_state = self.compute_new_cache_state(popularity)
        Update_flag = self.args.Update_flag
        update_mask = self.compute_update_mask(new_cache_state, Update_flag=Update_flag)
        return new_cache_state, update_mask