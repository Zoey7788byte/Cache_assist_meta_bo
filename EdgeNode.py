import torch.nn as nn
from typing import List, Dict, Tuple, Any
from collections import OrderedDict, defaultdict
from cloud_manager import CloudModelManager
import os
import torch
import heapq
from PIL import Image
import torchvision.transforms as transforms
import traceback
import shutil
import hashlib
from generate_pt.ModelArchitecture import MODEL_CONFIGS, create_model

class EdgeNode:
    def __init__(self, node_id: str, cloud_manager: CloudModelManager, 
                 max_storage: float = 1024.0, cache_dir: str = "./model_data/edge_cache_model"):
        self.node_id = node_id
        self.cloud_manager = cloud_manager
        self.max_storage = max_storage  # 单位：MB
        self.cache_dir = cache_dir
        self.used_storage = 0.0
        self.model_cache = {} #初始化模型字典
        os.makedirs(self.cache_dir, exist_ok=True)

    def Initialize_cache(self): 
        # 清空缓存目录并重置存储使用
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.used_storage = 0
        
        # 收集所有层（共享层和特定层）的信息
        all_layers = defaultdict(lambda: {'count': 0, 'keys': [], 'size': 0, 'type': None})
        
        # 构建层映射并统计信息
        for registry_key, model_info in self.cloud_manager.model_db.items():
            # 处理共享层
            if 'shared_path' in model_info:
                shared_path = model_info['shared_path']
                
                # 初始化文件大小（如果尚未记录）
                if all_layers[shared_path]['size'] == 0:
                    if not os.path.exists(shared_path):
                        print(f"[Warn] Shared path missing: {shared_path}")
                        continue
                        
                    try:
                        file_size = os.path.getsize(shared_path) / (1024 * 1024)  # MB
                        if file_size <= 0:
                            file_size = 1e-9
                        all_layers[shared_path]['size'] = file_size
                        all_layers[shared_path]['type'] = 'shared'
                    except OSError:
                        print(f"[Error] Cannot get size for: {shared_path}")
                        continue
                
                # 更新计数和关联键
                all_layers[shared_path]['count'] += 1
                all_layers[shared_path]['keys'].append(registry_key)
            
            # 处理特定层
            if 'specific_path' in model_info:
                specific_path = model_info['specific_path']
                
                # 初始化文件大小（如果尚未记录）
                if all_layers[specific_path]['size'] == 0:
                    if not os.path.exists(specific_path):
                        print(f"[Warn] Specific path missing: {specific_path}")
                        continue
                        
                    try:
                        file_size = os.path.getsize(specific_path) / (1024 * 1024)  # MB
                        if file_size <= 0:
                            file_size = 1e-9
                        all_layers[specific_path]['size'] = file_size
                        all_layers[specific_path]['type'] = 'specific'
                    except OSError:
                        print(f"[Error] Cannot get size for: {specific_path}")
                        continue
                
                # 特定层计数始终为1（只被一个模型使用）
                all_layers[specific_path]['count'] += 1
                all_layers[specific_path]['keys'].append(registry_key)

        # 生成候选列表（按价值排序）
        candidates = []
        for path, data in all_layers.items():
            # 跳过大小未初始化或无效的条目
            if data['size'] <= 0 or data['type'] is None:
                continue
                
            # 价值公式：使用次数 / 大小
            value = data['count'] / data['size']
            candidates.append({
                'path': path,
                'value': value,
                'count': data['count'],
                'size': data['size'],
                'type': data['type'],
                'keys': data['keys']  # 保存关联的registry_keys
            })
        
        # 按价值降序排序
        candidates.sort(key=lambda x: x['value'], reverse=True)
        
        # 统一缓存所有层
        for candidate in candidates:
            # 检查存储空间
            if self.used_storage + candidate['size'] > self.max_storage:
                layer_type = candidate['type'].capitalize()
                print(f"[Skip] {layer_type} {os.path.basename(candidate['path'])} | "
                    f"Need: {candidate['size']:.1f}MB | "
                    f"Used: {self.used_storage:.1f}/{self.max_storage}MB")
                continue
            
            # 创建缓存路径
            cache_filename = os.path.basename(candidate['path'])
            cache_path = os.path.join(self.cache_dir, cache_filename)
            
            # 安全复制文件
            storage_added = False
            try:
                if not os.path.exists(candidate['path']):
                    print(f"[Error] Source file disappeared: {candidate['path']}")
                    continue
                    
                if not os.path.exists(cache_path):
                    shutil.copy2(candidate['path'], cache_path)
                    
                    if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
                        raise RuntimeError("File copy failed")
                    
                    self.used_storage += candidate['size']
                    storage_added = True
                
                # 更新所有关联模型的缓存信息
                for registry_key in candidate['keys']:
                    if registry_key not in self.model_cache:
                        model_info = self.cloud_manager.model_db[registry_key]
                        self.model_cache[registry_key] = {
                            'id': model_info['id'],
                            'filename': model_info['filename'],
                            'dataset': model_info['dataset'],
                            'model': model_info['model'],
                            'type': model_info['type'],
                            'accuracy': model_info.get('accuracy'),
                            'shared_cache_path': None,
                            'specific_cache_path': None
                        }
                    
                    # 更新对应的缓存路径
                    if candidate['type'] == 'shared':
                        self.model_cache[registry_key]['shared_cache_path'] = cache_path
                    else:  # specific
                        self.model_cache[registry_key]['specific_cache_path'] = cache_path

            except Exception as e:
                layer_type = candidate['type'].capitalize()
                print(f"[Error] Cache {layer_type} failed: {str(e)}")
                if os.path.exists(cache_path):
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass
                if storage_added:
                    self.used_storage -= candidate['size']
        
        print(f"Cache completed | Used: {self.used_storage:.1f}MB/{self.max_storage}MB")

    def get_cache_status(self):
        """获取缓存空间使用情况"""
        from collections import defaultdict
        stats = defaultdict(int)
        for layer_type in ['shared', 'specific']:
            dir_path = os.path.join(self.cache_dir, layer_type)
            if os.path.exists(dir_path):
                stats[layer_type] = sum(
                    os.path.getsize(os.path.join(dir_path, f)) 
                    for f in os.listdir(dir_path)
                ) / (1024*1024)  # MB
        return stats
         
    def prepare_inference(self, image_path: str) -> torch.Tensor:
        """准备推理输入数据"""
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        return transform(img).unsqueeze(0)  # 添加batch维度
  
    def inference(self, request_img: str, data: dict) -> dict:
        if request_img:
            input_tensor = self.prepare_inference(request_img)
            try:
                model_data = data['model_data']
                model_arch = data['model_arch']
                
                # 获取模型在数据库中的完整信息
                registry_key = f"{model_data}|{model_arch}"
                model_info = self.cloud_manager.model_db.get(registry_key)
                if not model_info:
                    raise ValueError(f"Model {registry_key} not found in model database")

                # 基于model_cache计算缓存完整性
                shared_cache_path = os.path.join(self.cache_dir, 'shared', os.path.basename(model_info['shared_path']))
                specific_cache_path = os.path.join(self.cache_dir, 'specific', os.path.basename(model_info['specific_path']))
                
                # 直接从缓存字典获取状态
                shared_cached = shared_cache_path in self.model_cache
                specific_cached = specific_cache_path in self.model_cache
                completeness = (shared_cached + specific_cached) / 2  # 假设模型由两部分组成
                
                data['cache_status'] = "full" if completeness == 1 else "partial"
                
                # 模型组装逻辑（根据缓存元数据优化）
                if completeness == 1:
                    # 从缓存记录中获取模型特征
                    shared_info = self.model_cache[shared_cache_path]
                    specific_info = self.model_cache[specific_cache_path]
                    
                    model = self._load_full_edge_model(
                        shared_path = shared_cache_path,
                        specific_path = specific_cache_path,
                        model_arch = model_arch,
                        specific_metadata = specific_info
                    )
                    data['model_assembly'] = "full_cache"
                else:
                    # 智能组装逻辑：优先使用已缓存的组件
                    cached_components = {
                        'shared': shared_cache_path if shared_cached else None,
                        'specific': specific_cache_path if specific_cached else None
                    }
                    registry_key = f"{model_data}|{model_arch}"
                    model = self._assemble_hybrid_model(cached_components, model_arch, registry_key)
                    data['model_assembly'] = f"hybrid_{completeness*100:.1f}%"

                # 记录缓存使用详情
                data.update({
                    'used_model': f"{model_data}_{model_arch}",
                    'cache_details': {
                        'shared': {
                            'cached': shared_cached,
                            'path': shared_cache_path if shared_cached else None,
                            'accuracy': self.model_cache.get(shared_cache_path, {}).get('accuracy')
                        },
                        'specific': {
                            'cached': specific_cached,
                            'path': specific_cache_path if specific_cached else None,
                            'accuracy': self.model_cache.get(specific_cache_path, {}).get('accuracy')
                        }
                    }
                })
                output = model(input_tensor)
                return data, output
                
            except Exception as e:
                print(f"⚠️ Edge inference failed: {str(e)}")
        else:
            print("推理输入错误，进行下一次处理")
            return None

    def _load_full_edge_model(self, shared_path: str, specific_path: str, model_arch, specific_metadata: str) -> nn.Module:
        # 加载边缘节点缓存的完整模型
        model, config = create_model(model_arch, 2)
        model.shared_layers.load_state_dict(torch.load(shared_path))
        model.specific_layers.load_state_dict(torch.load(specific_path))

        print(f"✅ Loading full cached model: {model_arch}")

        return model

    def _assemble_hybrid_model(self, cached_components, model_arch, registry_key) -> nn.Module:
        """动态组装混合模型（缓存共享层+下载特定层）"""
        print(f"🔄 Assembling hybrid model")
        new_model, config = create_model(model_arch, 2)

        #边缘侧的共享权重存在
        if cached_components['shared']:
            shared_path = cached_components['shared']
        else:
            #TODO:TZY 增加从云端处理获取文件的延迟等信息
            shared_path = self.cloud_manager.model_db[registry_key]['shared_path']

        if cached_components['specific']:
            specific_path = cached_components['specific']
        else:
            #TODO:TZY 增加从云端处理获取文件的延迟等信息
            specific_path = self.cloud_manager.model_db[registry_key]['specific_path']

        
        new_model.shared_layers.load_state_dict(torch.load(shared_path))
        new_model.specific_layers.load_state_dict(torch.load(specific_path))
        
        return new_model
    
    def _cache_specific_layer(self, model_key: str, cloud_path: str):
        """缓存特定层到本地（自动跳过已存在的文件）"""
        specific_filename = os.path.basename(cloud_path)
        local_path = os.path.join(self.cache_dir, "specific", specific_filename)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # 如果本地文件已存在且内容完整，直接跳过
        if os.path.exists(local_path):
            print(f"✅ Specific layer already cached: {local_path}")
        else:
            # 远程路径下载
            if any(cloud_path.startswith(prefix) for prefix in ['http://', 'https://', 's3://']):
                self._download_file(cloud_path, local_path)
            # 本地路径复制（需检查是否相同文件）
            elif os.path.abspath(cloud_path) != os.path.abspath(local_path):
                shutil.copy(cloud_path, local_path)

        # 更新缓存记录
        self.model_cache.setdefault(model_key, {})['specific_path'] = local_path

    def is_model_cached(self, model_id):
        for registry_key, model_info in self.model_cache.items():
            #检查当前条目的'id'是否匹配目标model_id
            if model_info['id'] == model_id:
                # print("边缘节点保存了模型: ",model_id)
                return True
        return False