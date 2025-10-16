import torch.nn as nn
from typing import List, Dict, Tuple
from cloud_manager import CloudModelManager
from EdgeNode import EdgeNode
import os

class CloudEdgeCoordinator:
    """云-边缘协同管理器"""
    def __init__(self, cloud_manager: CloudModelManager):
        self.cloud = cloud_manager
        self.edge_nodes: Dict[str, EdgeNode] = {}
        
    def register_edge_node(self, node_id: str, max_storage) -> EdgeNode:
        """注册边缘节点"""
        if node_id in self.edge_nodes:
            raise ValueError(f"Edge node {node_id} already registered")
            
        edge_node = EdgeNode(node_id, self.cloud, max_storage)
        self.edge_nodes[node_id] = edge_node
        return edge_node
    
    def Hand_edge_nodel_cache_update(self, edge_node_id, selected_models):
        print("selected_models",len(selected_models))

        # 2. 准备数据
        # 获取边缘节点当前缓存的所有模型ID
        cached_model_ids = set()
        for cache_info in self.edge_nodes[edge_node_id].model_cache.values():
            cached_model_ids.add(cache_info['id'])
        
        # 3. 删除不在选定列表中的模型
        removed_count = 0
        registry_keys_to_remove = []
        
        # 首先收集需要删除的注册键
        for registry_key, cache_info in list(self.edge_nodes[edge_node_id].model_cache.items()):
            if cache_info['id'] not in selected_models:
                registry_keys_to_remove.append(registry_key)
        
        # 执行删除操作
        for registry_key in registry_keys_to_remove:
            model_id = self.edge_nodes[edge_node_id].model_cache[registry_key]['id']
            del self.edge_nodes[edge_node_id].model_cache[registry_key]
            removed_count += 1
            # print(f"  已删除模型 {model_id} (注册键: {registry_key})")
        
        print(f"删除完成: 移除了 {removed_count} 个不在选定列表中的模型")
        
        # 4. 添加未缓存的选定模型
        # 获取当前缓存模型ID（删除后）
        current_cached_ids = {cache_info['id'] for cache_info in self.edge_nodes[edge_node_id].model_cache.values()}
        
        # 确定需要添加的模型
        models_to_add = [model_id for model_id in selected_models 
                        if model_id not in current_cached_ids]
        
        added_count = 0
        
        if models_to_add:
            print(f"需要添加 {len(models_to_add)} 个新模型:")
            # 遍历云端数据库查找需要添加的模型
            for registry_key, model_info in self.cloud.model_db.items():
                model_id = model_info['id']
                
                if model_id in models_to_add:
                    # 创建新的缓存条目
                    new_cache_entry = {
                        'id': model_id,
                        'filename': model_info['filename'],
                        'dataset': model_info['dataset'],
                        'model': model_info['model'],
                        'type': model_info['type'],
                        'accuracy': model_info.get('accuracy'),
                        'shared_cached': True,
                        'specific_cached': False,
                        'shared_cache_path': model_info['shared_path']
                    }
                    
                    # 添加到边缘节点的缓存
                    self.edge_nodes[edge_node_id].model_cache[registry_key] = new_cache_entry
                    added_count += 1
                    models_to_add.remove(model_id)  # 从待添加列表中移除
                    
                    # print(f"  已添加模型 {model_id} (注册键: {registry_key})")
        
            # 处理云端数据库中不存在的模型
            if models_to_add:
                print(f"警告: {len(models_to_add)} 个模型在云端数据库不存在: {', '.join(models_to_add)}")
            
            print(f"添加完成: 新增 {added_count} 个模型到缓存")
        else:
            print("无需添加新模型")
        
        # 5. 最终报告
        final_count = len(self.edge_nodes[edge_node_id].model_cache)
        print(f"同步完成! 边缘节点 {edge_node_id} 当前缓存 {final_count} 个模型")
        
        return {
            'removed': removed_count,
            'added': added_count,
            'total': final_count
        }