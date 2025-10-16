import os
import json
import time
import numpy as np
from collections import defaultdict
from parameters import args_parser
args = args_parser()

class RequestGenerator:
    def __init__(self, NUM_SHARED_MODELS=5, NUM_SPECIFIC_MODELS=100,
                 model_db=None, test_dir='./data/dataset_formation/test',
                 base_alpha=args.base_alpha, d=args.expend, task_length=args.task_length,
                 task_variation_mode="zipf_alpha",  # 新增：任务变异模式
                 min_categories_per_task=3,         # 新增：每个任务最少类别数
                 max_categories_per_task=20,        # 新增：每个任务最多类别数
                 category_overlap_ratio=0.3,        # 新增：任务间类别重叠比例
                 task_complexity_range=(0.5, 2.0)   # 新增：任务复杂度范围
                 ):
        self.test_dir = test_dir
        self.NUM_SHARED_MODELS = NUM_SHARED_MODELS
        self.NUM_SPECIFIC_MODELS = NUM_SPECIFIC_MODELS
        self.base_alpha = base_alpha  # 基础α值
        self.d = d  # 随机变化范围
        self.task_length = task_length  # 任务长度（时隙数）
        self.task_variation_mode = task_variation_mode  # 任务变异模式
        self.min_categories_per_task = min_categories_per_task
        self.max_categories_per_task = max_categories_per_task
        self.category_overlap_ratio = category_overlap_ratio
        self.task_complexity_range = task_complexity_range
        self.image_path_index = defaultdict(list)

        # 统一 model_db 格式为 list of dict
        self.model_db = []
        if isinstance(model_db, dict):
            self.model_db = list(model_db.values())
        elif isinstance(model_db, list):
            for m in model_db:
                if isinstance(m, dict):
                    self.model_db.append(m)
                else:  # 如果只是字符串或数字，转成 dict
                    self.model_db.append({'id': str(m), 'type': 'specific'})
        elif model_db is None:
            self.model_db = []
        else:
            raise ValueError("model_db 必须是 dict、list 或 None")

        # 历史文件名 - 更新以包含基础α、d值和任务长度
        self.history_dir = "requests"
        os.makedirs(self.history_dir, exist_ok=True)
        # 更新文件名以包含更多参数
        param_str = f"base{self.base_alpha}_d{self.d}_T{self.task_length}_mode{self.task_variation_mode}"
        param_str += f"_minCat{self.min_categories_per_task}_maxCat{self.max_categories_per_task}"
        param_str += f"_overlap{self.category_overlap_ratio}_complexity{self.task_complexity_range[0]}-{self.task_complexity_range[1]}"
        self.history_filename = f"{self.history_dir}/request_history_{param_str}_models{self.NUM_SHARED_MODELS+self.NUM_SPECIFIC_MODELS}.json"
        self.matrix_filename = f"{self.history_dir}/request_matrix_{param_str}_models{self.NUM_SHARED_MODELS+self.NUM_SPECIFIC_MODELS}.json"
        self.task_stats_filename = f"{self.history_dir}/task_stats_{param_str}_models{self.NUM_SHARED_MODELS+self.NUM_SPECIFIC_MODELS}.json"

    def get_all_categories(self, root_dir):
        categories = {}
        for group in os.listdir(root_dir):
            group_path = os.path.join(root_dir, group)
            if not os.path.isdir(group_path):
                continue
            sub_dirs = [os.path.join(group_path, item) for item in os.listdir(group_path)
                        if os.path.isdir(os.path.join(group_path, item))]
            if sub_dirs:
                for sub_path in sub_dirs:
                    subcategory = os.path.basename(sub_path)
                    img_paths = [os.path.join(sub_path, f) for f in os.listdir(sub_path)
                                 if f.lower().endswith(('.jpg', '.jpeg'))]
                    if img_paths:
                        categories[subcategory] = img_paths
            else:
                img_paths = [os.path.join(group_path, f) for f in os.listdir(group_path)
                             if f.lower().endswith(('.jpg', '.jpeg'))]
                if img_paths:
                    categories[group] = img_paths
        # ✅ 按类别名排序，保证 Zipf rank 稳定
        return dict(sorted(categories.items()))

    def _sample_zipf(self, k, alpha, size=1):
        """有限集合 Zipf-like 采样，支持变化的α参数"""
        weights = np.array([1.0 / (i + 1) ** alpha for i in range(k)])
        probs = weights / weights.sum()
        return np.random.choice(range(k), size=size, p=probs)
    
    def _create_task_specific_categories(self, all_categories, task_id, total_tasks):
        """为每个任务创建特定的类别子集，确保任务间有差异但也有重叠"""
        all_keys = list(all_categories.keys())
        num_all_categories = len(all_keys)
        
        # 确定这个任务使用的类别数量
        num_categories = np.random.randint(
            self.min_categories_per_task, 
            min(self.max_categories_per_task, num_all_categories) + 1
        )
        
        # 确定重叠的类别数量
        overlap_count = int(num_categories * self.category_overlap_ratio)
        unique_count = num_categories - overlap_count
        
        # 为每个任务分配特定的随机种子，确保可重复性
        rng = np.random.RandomState(task_id)
        
        # 选择重叠的类别（从固定的前N个类别中选择）
        fixed_categories = all_keys[:min(50, num_all_categories)]  # 固定一组类别用于重叠
        overlap_categories = rng.choice(
            fixed_categories, 
            size=min(overlap_count, len(fixed_categories)), 
            replace=False
        ).tolist()
        
        # 选择独特的类别（从所有类别中随机选择）
        remaining_categories = [c for c in all_keys if c not in overlap_categories]
        unique_categories = rng.choice(
            remaining_categories, 
            size=min(unique_count, len(remaining_categories)), 
            replace=False
        ).tolist()
        
        # 合并类别并创建子字典
        task_categories = {}
        for category in overlap_categories + unique_categories:
            task_categories[category] = all_categories[category]
            
        return task_categories

    def generate_zipf_requests_with_timeslots(self, categories, time_slots):
        """生成请求，每个任务（T个时隙）使用不同的配置"""
        all_category_keys = list(categories.keys())
        num_all_categories = len(all_category_keys)

        slot_requests = {}
        total_requests = 0
        task_stats = {}  # 存储每个任务的统计信息
        
        # 计算任务数量
        num_tasks = time_slots // self.task_length
        remainder_slots = time_slots % self.task_length
        
        # 处理完整的任务
        for task_idx in range(num_tasks):
            # 为当前任务创建特定的类别子集
            task_categories = self._create_task_specific_categories(categories, task_idx, num_tasks)
            task_category_keys = list(task_categories.keys())
            num_task_categories = len(task_category_keys)
            
            # 根据选择的模式生成任务参数
            if self.task_variation_mode == "zipf_alpha":
                # 为当前任务生成随机α值：base_alpha + W，其中W是[-d, d]之间的均匀分布
                W = np.random.uniform(-self.d, self.d)
                current_alpha = self.base_alpha + W
                complexity = 1.0  # 默认复杂度
                
            elif self.task_variation_mode == "complexity":
                # 基于复杂度的变异
                complexity = np.random.uniform(*self.task_complexity_range)
                current_alpha = self.base_alpha * complexity
                W = 0  # 不使用W
                
            elif self.task_variation_mode == "mixed":
                # 混合模式：同时使用α和复杂度
                W = np.random.uniform(-self.d, self.d)
                complexity = np.random.uniform(*self.task_complexity_range)
                current_alpha = self.base_alpha * complexity + W
                
            else:
                raise ValueError(f"未知的任务变异模式: {self.task_variation_mode}")
            
            # 当前任务的起始和结束时隙
            start_slot = task_idx * self.task_length
            end_slot = start_slot + self.task_length
            
            # 存储任务统计信息
            task_stats[task_idx] = {
                'start_slot': start_slot,
                'end_slot': end_slot - 1,  # 结束时隙是包含的
                'alpha_value': current_alpha,
                'W_value': W,
                'complexity': complexity,
                'num_categories': num_task_categories,
                'categories': task_category_keys,
                'request_counts': defaultdict(int)
            }
            
            for slot in range(start_slot, end_slot):
                total_requests += 1
                
                # 按 Zipf 分布采样 1 个类别，使用当前任务的α值
                chosen_idx = self._sample_zipf(num_task_categories, current_alpha, size=1)[0]
                category = task_category_keys[chosen_idx]
                img_path = np.random.choice(task_categories[category])
                
                # 更新任务统计
                task_stats[task_idx]['request_counts'][category] += 1

                # 保存结果
                slot_requests[slot] = {
                    'total_requests': 1,
                    'task_id': task_idx,  # 记录任务ID
                    'alpha_value': current_alpha,  # 记录当前任务使用的α值
                    'W_value': W,  # 记录W值
                    'complexity': complexity,  # 记录复杂度
                    'category_counts': {category: 1},
                    'request_sequence': [(category, img_path)]
                }
        
        # 处理剩余的时隙（如果有）
        if remainder_slots > 0:
            # 为剩余时隙创建一个新任务
            task_idx = num_tasks
            task_categories = self._create_task_specific_categories(categories, task_idx, num_tasks + 1)
            task_category_keys = list(task_categories.keys())
            num_task_categories = len(task_category_keys)
            
            # 生成任务参数
            if self.task_variation_mode == "zipf_alpha":
                W = np.random.uniform(-self.d, self.d)
                current_alpha = self.base_alpha + W
                complexity = 1.0
            elif self.task_variation_mode == "complexity":
                complexity = np.random.uniform(*self.task_complexity_range)
                current_alpha = self.base_alpha * complexity
                W = 0
            elif self.task_variation_mode == "mixed":
                W = np.random.uniform(-self.d, self.d)
                complexity = np.random.uniform(*self.task_complexity_range)
                current_alpha = self.base_alpha * complexity + W
            
            start_slot = num_tasks * self.task_length
            end_slot = start_slot + remainder_slots
            
            # 存储任务统计信息
            task_stats[task_idx] = {
                'start_slot': start_slot,
                'end_slot': end_slot - 1,
                'alpha_value': current_alpha,
                'W_value': W,
                'complexity': complexity,
                'num_categories': num_task_categories,
                'categories': task_category_keys,
                'request_counts': defaultdict(int)
            }
            
            for slot in range(start_slot, end_slot):
                total_requests += 1
                
                # 按 Zipf 分布采样 1 个类别，使用当前任务的α值
                chosen_idx = self._sample_zipf(num_task_categories, current_alpha, size=1)[0]
                category = task_category_keys[chosen_idx]
                img_path = np.random.choice(task_categories[category])
                
                # 更新任务统计
                task_stats[task_idx]['request_counts'][category] += 1

                # 保存结果
                slot_requests[slot] = {
                    'total_requests': 1,
                    'task_id': task_idx,  # 记录任务ID
                    'alpha_value': current_alpha,  # 记录当前任务使用的α值
                    'W_value': W,  # 记录W值
                    'complexity': complexity,  # 记录复杂度
                    'category_counts': {category: 1},
                    'request_sequence': [(category, img_path)]
                }

        global_counts = defaultdict(int)
        for slot_data in slot_requests.values():
            for category, count in slot_data['category_counts'].items():
                global_counts[category] += count

        return slot_requests, dict(global_counts), total_requests, task_stats

    def _match_model_id(self, category, model_id_to_category):
        """匹配类别到模型ID"""
        if not model_id_to_category:
            return None
        for model_id, cat in model_id_to_category.items():
            if category.lower() == cat.lower():
                return model_id
        for model_id, cat in model_id_to_category.items():
            if category.lower() in cat.lower():
                return model_id
        return None

    def generate_requests(self, num_nodes, time_slots):
        # 优先从历史文件加载
        if os.path.exists(self.history_filename) and os.path.exists(self.task_stats_filename):
            print(f"检测到历史请求文件 {self.history_filename}，直接加载数据...")
            try:
                with open(self.history_filename, 'r') as f:
                    history_records = json.load(f)
                with open(self.task_stats_filename, 'r') as f:
                    task_stats = json.load(f)
                    
                num_models = self.NUM_SHARED_MODELS + self.NUM_SPECIFIC_MODELS
                requests = np.zeros((time_slots, num_nodes, num_models), dtype=int)
                request_records = defaultdict(list)
                for record in history_records:
                    if record['model_index'] >= num_models:
                        continue
                    key = (record['slot'], record['node'], record['model_index'])
                    requests[record['slot'], record['node'], record['model_index']] += 1
                    request_records[key].append(record['img_path'])
                print(f"成功加载历史请求记录: {len(history_records)} 条记录")
                return requests, request_records, history_records, task_stats
            except Exception as e:
                print(f"加载历史请求文件失败: {e}，将重新生成请求")

        # 没有历史文件或加载失败则生成新请求
        all_categories = self.get_all_categories(self.test_dir)
        slot_requests, _, _, task_stats = self.generate_zipf_requests_with_timeslots(
            all_categories, time_slots
        )

        shared_models = [m for m in self.model_db if m.get('type') == 'shared']
        specific_models = [m for m in self.model_db if m.get('type') == 'specific']
        shared_models.sort(key=lambda x: x['id'])
        specific_models.sort(key=lambda x: x['id'])

        # 建立 model_id → index
        model_id_to_index = {}
        for idx, model in enumerate(shared_models):
            model_id_to_index[model['id']] = idx
        for idx, model in enumerate(specific_models):
            model_id_to_index[model['id']] = self.NUM_SHARED_MODELS + idx

        # 建立 model_id → category
        model_id_to_category = {}
        for model in specific_models:
            filename = model.get('filename', '')
            dataset = model.get('dataset', '')
            category = dataset if dataset else filename.split('_')[0]
            model_id_to_category[model['id']] = category
        for model in shared_models:
            model_id_to_category[model['id']] = model.get('dataset', 'shared_placeholder')

        num_models = self.NUM_SHARED_MODELS + self.NUM_SPECIFIC_MODELS
        requests = np.zeros((time_slots, num_nodes, num_models), dtype=int)
        history_records = []
        request_records = defaultdict(list)

        for slot, slot_data in slot_requests.items():
            if not slot_data['request_sequence']:
                continue
            total_slot_requests = len(slot_data['request_sequence'])
            requests_per_node = [total_slot_requests // num_nodes] * num_nodes
            for i in range(total_slot_requests % num_nodes):
                requests_per_node[i] += 1
            request_idx = 0
            for node in range(num_nodes):
                for _ in range(requests_per_node[node]):
                    if request_idx >= total_slot_requests:
                        break
                    category, img_path = slot_data['request_sequence'][request_idx]
                    request_idx += 1

                    # 匹配模型ID
                    model_id = self._match_model_id(category, model_id_to_category)
                    if model_id not in model_id_to_index:
                        continue  # 跳过找不到的类别

                    model_index = model_id_to_index[model_id]
                    
                    # 只处理专用模型请求
                    if model_index >= self.NUM_SHARED_MODELS:
                        # 更新专用模型请求计数
                        requests[slot, node, model_index] += 1
                        request_records[(slot, node, model_index)].append(img_path)
                        history_records.append({
                            'slot': slot,
                            'node': node,
                            'model_index': int(model_index),
                            'model_id': model_id,
                            'category': category,
                            'img_path': img_path,
                            'task_id': slot_data['task_id'],  # 记录任务ID
                            'alpha_value': slot_data['alpha_value'],  # 记录使用的α值
                            'W_value': slot_data['W_value'],  # 记录W值
                            'complexity': slot_data.get('complexity', 1.0),  # 记录复杂度
                            'timestamp': time.time()
                        })
                    # 不再处理共享模型请求
        
        # 保存所有数据
        with open(self.history_filename, 'w') as f:
            json.dump(history_records, f, indent=2)
        with open(self.matrix_filename, 'w') as f:
            json.dump(requests.tolist(), f, indent=2)
        with open(self.task_stats_filename, 'w') as f:
            # 将defaultdict转换为普通dict以便JSON序列化
            for task_id, stats in task_stats.items():
                if 'request_counts' in stats and isinstance(stats['request_counts'], defaultdict):
                    stats['request_counts'] = dict(stats['request_counts'])
            json.dump(task_stats, f, indent=2)
            
        print(f"生成的新请求已保存到 {self.history_filename}")
        print(f"请求矩阵已保存到 {self.matrix_filename}")
        print(f"任务统计信息已保存到 {self.task_stats_filename}")

        return requests, request_records, history_records, task_stats