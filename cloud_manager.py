import traceback
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import json
from dataclasses import dataclass
import hashlib
import numpy as np
import os
from torchvision import models
import re
from generate_pt.ModelArchitecture import MODEL_CONFIGS, create_model
from parameters import args_parser
args = args_parser()

class CloudModelManager:
    def __init__(self, storage_dir: str = "./model_data/cloud_registry_model", weights_dir: str = "./generate_pt/Model_weight_pt"):
        self.storage_dir = storage_dir
        self.model_db = {}
        self.weights_dir = weights_dir
        self.available_models, self.num_shared, self.num_specific = self._scan_available_models()
        self.MODEL_CONFIGS = MODEL_CONFIGS()
        self.model_sizes, self.computation_req = self.init_model_size_and_computation_req()
        self.dependency_matrix = self.get_dependency_matrix()
        self.rho_m_values = self._initialize_rho_m(self.available_models)
        os.makedirs(storage_dir, exist_ok=True)

    def _scan_available_models(self):
        """扫描权重目录，获取可用模型信息，保留每个模型类型中精度最高的模型，并建立ID映射"""
        shared_models = []  # 存储shared模型信息
        specific_models = []    # 存储specific模型信息

        # 处理pretrained模型
        for filename in os.listdir(self.weights_dir):
            if filename.endswith('pretrained.pth'):
                try:
                    # 移除后缀
                    filename_clean = filename.replace('.pth', '')
                    # 移除末尾的'_pretrained'
                    base_name = filename_clean.replace('_pretrained', '')
                    
                    # 分割模型和数据集
                    parts = base_name.split('_')
                    if len(parts) >= 1:
                        model = parts[-1]
                        dataset = '_'.join(parts[:-1]) if len(parts[:-1]) > 0 else ''
                    else:
                        continue
                    filename =  base_name+'_shared.pth'
                    
                    shared_models.append({
                        'filename': filename,
                        'dataset': dataset,
                        'model': model,
                        'type': 'shared',
                        'accuracy': None  # shared模型没有准确率
                    })
                except Exception as e:
                    print(f"解析shared模型失败：{filename}，错误：{e}")
                    continue

        # 处理specific模型
        for filename in os.listdir(self.weights_dir):
            if not filename.endswith('.pth') or filename.endswith("pretrained.pth"):
                continue     
            try:
                # 分割准确率部分前先移除后缀
                filename_clean = filename.replace('.pth', '')
                
                # 分割主体和准确率部分
                if '_acc' not in filename_clean:
                    continue
                    
                main_part, acc_part = filename_clean.split('_acc', 1)
                # 提取准确率数值
                acc_value_str = acc_part.split('_')[0]
                accuracy = float(acc_value_str)
                
                # 解析模型、类型和数据集
                parts = main_part.split('_')
                if len(parts) >= 2:
                    model = parts[-2]
                    type_ = parts[-1]
                    dataset = '_'.join(parts[:-2]) if len(parts[:-2]) > 0 else ''
                else:
                    continue

                specific_models.append({
                    'filename': filename,
                    'dataset': dataset,
                    'model': model,
                    'type': type_,
                    'accuracy': accuracy
                })
            except (ValueError, IndexError) as e:
                print(f"解析specific模型失败：{filename}，错误：{e}")
                continue

        # 按数据集+模型+类型分组，保留最高精度 (只针对specific模型)
        best_specific = {}
        for info in specific_models:
            key = (info['dataset'], info['model'], info['type'])
            if key not in best_specific or info['accuracy'] > best_specific[key]['accuracy']:
                best_specific[key] = info

        # 建立ID映射关系
        all_models = []
        model_id = 0
        
        # 先添加shared模型 (ID 0 到 N-1)
        for info in shared_models:
            info['id'] = model_id
            model_id += 1
            all_models.append(info)
        num_shared = model_id
        
        # 再添加specific模型 (ID从N开始)
        for info in best_specific.values():
            info['id'] = model_id
            model_id += 1
            all_models.append(info)
        num_specific = model_id - num_shared
        return all_models, num_shared, num_specific

    def load_model(self, dataset_name, model_arch, model_type):
        candidates = []
        for model_info in self.available_models:
            if model_info['dataset'] != dataset_name:
                continue
            
            # 构建匹配条件
            name_match = (model_arch is None) or (model_info['model'] == model_arch)
            type_match = (model_type is None) or (model_info['type'] == model_type)
            
            if name_match and type_match:
                candidates.append(model_info)
        
        if not candidates:
            available_models = [f"{m['model']}({m['type']})" for m in self.available_models if m['dataset'] == dataset_name]
            raise ValueError(f"No model found for dataset '{dataset_name}'. Available: {available_models}")
        
        # 即使经过扫描阶段的筛选，这里仍做二次精度验证
        best_model = max(candidates, key=lambda x: x['accuracy'])
        
        # 模型类型校验
        if best_model['model'] not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model architecture: {best_model['model']}. Supported: {list(self.MODEL_CONFIGS.keys())}")
        
        # 加载特定模型的权重
        specific_weight_path = os.path.join(self.weights_dir, best_model['filename'])
        if not os.path.exists(specific_weight_path):
            raise FileNotFoundError(f"Weight file not found: {specific_weight_path}")
        
        #加载对应共享模型的权重
        shared_weight_path = os.path.join(self.weights_dir, best_model['model'] + "_pretrained.pth")
        if not os.path.exists(shared_weight_path):
            raise FileNotFoundError(f"shared Weight file not found: {shared_weight_path}")
        
        model, config = create_model(best_model['model'], 2) #微调模型的创建，这里默认是5
        model.shared_layers.load_state_dict(torch.load(shared_weight_path, map_location=torch.device('cpu')))
        # model.specific_layers.load_state_dict(torch.load(specific_weight_path, map_location=torch.device('cpu')))
        model.specific_layers.load_state_dict(
            torch.load(
                specific_weight_path,
                map_location=torch.device('cpu'),
                weights_only=False  # Required for legacy .tar format
            )
        )
        return model, config

    def register_split_model(self, model_info, model, model_dataset, model_arch, model_type, file_name, accuracy):
        """注册模型并提取共享层，同时保存完整的模型信息"""
        os.makedirs(f"{self.storage_dir}/shared", exist_ok=True)
        os.makedirs(f"{self.storage_dir}/specific", exist_ok=True)
        
        shared_path = f"{self.storage_dir}/shared/{model_arch}_shared.pth"
        specific_path = f"{self.storage_dir}/specific/{model_dataset}_{model_arch}_{model_type}_acc{accuracy}.pth"
        
        orig_weight_path = os.path.join(self.weights_dir, file_name)
        if not os.path.exists(orig_weight_path):
            raise FileNotFoundError(f"Source weight file missing: {orig_weight_path}")
        try:    
            if os.path.exists(shared_path) and os.path.exists(specific_path):
                print(f"Shared and specific layers for {model_dataset}_{model_arch}_{model_type} already exist. Skipping saving.")
            else:
                # 保存前确保目录存在
                os.makedirs(os.path.dirname(shared_path), exist_ok=True)
                os.makedirs(os.path.dirname(specific_path), exist_ok=True)
                torch.save(model.shared_layers.state_dict(), shared_path)
                torch.save(model.specific_layers.state_dict(), specific_path)
        except Exception as e:
            for path in [shared_path, specific_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            raise RuntimeError(f"Model splitting failed: {str(e)}") from e
        
        registry_key = f"{model_dataset}|{model_arch}"
        # 保存完整的模型信息到数据库
        self.model_db[registry_key] = {
            'id': model_info['id'],
            'filename': file_name,
            'dataset': model_dataset,
            'model': model_arch,
            'type': model_type,
            'accuracy': accuracy,
            'shared_path': shared_path,
            'specific_path': specific_path
        }

    def register_trained_models(self):
        """注册所有训练好的模型到云服务，保存完整模型信息"""
        success_count = 0
        skip_count = 0
        for model_info in self.available_models:
            dataset = model_info['dataset']
            model_arch = model_info['model']
            model_type = model_info['type']
            filename = model_info['filename']
            accuracy = model_info['accuracy']
            model_id = model_info['id']
            
            # 生成标准路径
            shared_path = f"{self.storage_dir}/shared/{model_arch}_shared.pth"
            specific_path = f"{self.storage_dir}/specific/{dataset}_{model_arch}_{model_type}_acc{accuracy}.pth"
            registry_key = f"{dataset}|{model_arch}"
            
            # 关键改进：无论文件是否存在，都需确保注册信息写入
            if registry_key in self.model_db:
                # 模型已注册，跳过处理
                print(f"模型已存在且已注册，跳过: {registry_key}")
                skip_count += 1
                continue
            
            if os.path.exists(shared_path) and os.path.exists(specific_path):
                # 文件存在但未注册，补充完整信息
                self.model_db[registry_key] = {
                    'id': model_id,
                    'filename': filename,
                    'dataset': dataset,
                    'model': model_arch,
                    'type': model_type,
                    'accuracy': accuracy,
                    'shared_path': shared_path,
                    'specific_path': specific_path,
                }
                # print(f"检测到已存在模型文件，补充注册: {registry_key}")
                skip_count += 1
                continue
            
            try:
                # 前置校验
                if model_arch not in self.MODEL_CONFIGS:
                    print(f"跳过不支持模型: {model_arch}")
                    continue

                if model_type == 'shared':
                    # 预训练模型特殊处理
                    shared_path = os.path.join(self.storage_dir,'shared', filename)
                    self.model_db[registry_key] = {
                        'id': model_id,
                        'filename': filename,
                        'dataset': dataset,
                        'model': model_arch,
                        'type': model_type,
                        'accuracy': accuracy,
                        'shared_path': shared_path
                    }
                    print(f"注册预训练模型: {registry_key}")
                    success_count += 1
                    continue
                
                # 加载模型进行验证
                model, config = self.load_model(dataset, model_arch, model_type)

                # 注册拆分后的模型（内部会自动保存文件）
                self.register_split_model(
                    model_info,  # 传递完整的模型信息
                    model,
                    model_dataset=dataset,
                    model_arch=model_arch,
                    model_type=model_type,
                    file_name=filename,
                    accuracy=accuracy
                )
                
                print(f"注册成功: {dataset}_{model_arch}_{model_type} (ID: {model_id}, 准确率: {accuracy:.2f})")
                success_count += 1
                
            except Exception as e:
                error_msg = f"注册 {filename} (ID: {model_id}) 失败: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                continue

        print(f"\n注册完成。成功数: {success_count}/{len(self.available_models)}, 跳过数: {skip_count}/{len(self.available_models)}")

    # 提取特定模型架构名称
    def extract_arch_from_specific(self, name):
        parts = name.split('_')
        for i in range(len(parts)):
            if parts[i] == 'specific':
                return parts[i - 1]  # 架构名在specific前一项
        return None
        
    def get_dependency_matrix(self):
        weights_dir = './model_data/cloud_registry_model/'
        shared_dir = os.path.join(weights_dir, 'shared')
        specific_dir = os.path.join(weights_dir, 'specific')

        # 对文件列表进行排序以确保一致顺序
        shared_models = sorted([f for f in os.listdir(shared_dir) if f.endswith('.pth')])
        specific_models = sorted([f for f in os.listdir(specific_dir) if f.endswith('.pth')])

        # 提取共享模型架构名称
        shared_archs = [f.replace('_shared.pth', '') for f in shared_models]
        
        specific_archs = [self.extract_arch_from_specific(f) for f in specific_models]

        num_specific = len(specific_models)
        num_shared = len(shared_models)

        dependency_matrix = np.zeros((num_specific, num_shared))

        # 构建依赖矩阵
        for i, spec_arch in enumerate(specific_archs):
            if spec_arch is None:
                print(f"[Warn] Failed to extract architecture from: {specific_models[i]}")
                continue
                
            for j, shared_arch in enumerate(shared_archs):
                if spec_arch == shared_arch:
                    dependency_matrix[i, j] = 1
                    # print(f"Specific model {i} ({specific_models[i]}) depends on shared model {j} ({shared_models[j]})")
                    break  # 一般一对一依赖
        
        return dependency_matrix

    def init_model_size(self):
        """使用现有模型信息初始化模型大小，智能处理文件名格式"""
        # 初始化模型大小数组（大小为总模型数量）
        model_sizes = np.zeros(len(self.available_models))
        print("\n===== 使用ID信息初始化模型大小 =====")
        
        # 遍历所有可用模型
        for model_info in self.available_models:
            model_id = model_info['id']
            model_type = model_info['type']
            filename = model_info['filename']
            
            # 添加ID有效性检查
            if model_id < 0 or model_id >= len(model_sizes):
                print(f"错误: 无效模型ID {model_id}，超出范围 [0, {len(model_sizes)-1}]")
                continue
                
            # 尝试多种路径获取方式
            possible_paths = []
            
            # 1. 优先检查原始权重路径（所有模型类型）
            weights_path = os.path.join(self.storage_dir, filename)
            possible_paths.append(weights_path)
            
            # 2. 特定模型路径处理
            if model_type != 'shared':
                # 智能处理精度格式：移除多余的0
                clean_filename = filename
                if '_acc' in filename:
                    try:
                        base_part, rest = filename.split('_acc', 1)
                        acc_value, ext = os.path.splitext(rest)
                        # 转换为浮点数再格式化为字符串，移除多余的0
                        normalized_acc = f"{float(acc_value):.2f}".rstrip('0').rstrip('.')
                        clean_filename = f"{base_part}_acc{normalized_acc}{ext}"
                    except Exception as e:
                        print(f"精度格式化失败: {filename}, 错误: {e}")
                
                # 构建标准特定路径
                specific_path = os.path.join(self.storage_dir, "specific", clean_filename)
                possible_paths.append(specific_path)
                
                # 添加原始文件名作为后备路径
                possible_paths.append(os.path.join(self.storage_dir, "specific", filename))
            
            # 3. 检查注册的共享路径（如果存在）
            if model_type == 'shared' and 'filename' in model_info:
                shared_path = os.path.join(self.storage_dir, "shared", filename)
                possible_paths.append(shared_path)
            
            # 查找存在的文件路径
            found_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_path = path
                    break
            
            # 计算文件大小
            if found_path:
                # 获取文件大小（MB）
                size = os.path.getsize(found_path) / (1024 * 1024)
                model_sizes[model_id] = size
                # print(f"模型 ID:{model_id} ({model_type}, {os.path.basename(found_path)}) 大小: {size:.2f} MB")
            else:
                # 文件不存在，使用默认值
                default_size = np.random.uniform(5, 10)
                model_sizes[model_id] = default_size
                print(f"警告: 模型 ID:{model_id} ({model_type}, {filename}) 文件不存在（尝试路径: {', '.join(possible_paths)}），使用默认大小 {default_size:.2f} MB")
        
        return model_sizes

    def init_model_size_and_computation_req(self):
        """使用现有模型信息初始化模型大小和计算需求，智能处理文件名格式"""
        # 初始化模型大小和计算需求数组（大小为总模型数量）
        model_sizes = np.zeros(len(self.available_models))
        computation_reqs = np.zeros(len(self.available_models))
        
        print("\n===== 使用ID信息初始化模型大小和计算需求 =====")
        
        # 遍历所有可用模型
        for model_info in self.available_models:
            model_id = model_info['id']
            model_type = model_info['type']
            filename = model_info['filename']
            
            # 添加ID有效性检查
            if model_id < 0 or model_id >= len(model_sizes):
                print(f"错误: 无效模型ID {model_id}，超出范围 [0, {len(model_sizes)-1}]")
                continue
                
            # 尝试多种路径获取方式
            possible_paths = []
            
            # 1. 优先检查原始权重路径（所有模型类型）
            weights_path = os.path.join(self.storage_dir, filename)
            possible_paths.append(weights_path)
            
            # 2. 特定模型路径处理
            if model_type != 'shared':
                # 智能处理精度格式：移除多余的0
                clean_filename = filename
                if '_acc' in filename:
                    try:
                        base_part, rest = filename.split('_acc', 1)
                        acc_value, ext = os.path.splitext(rest)
                        # 转换为浮点数再格式化为字符串，移除多余的0
                        normalized_acc = f"{float(acc_value):.2f}".rstrip('0').rstrip('.')
                        clean_filename = f"{base_part}_acc{normalized_acc}{ext}"
                    except Exception as e:
                        print(f"精度格式化失败: {filename}, 错误: {e}")
                
                # 构建标准特定路径
                specific_path = os.path.join(self.storage_dir, "specific", clean_filename)
                possible_paths.append(specific_path)
                
                # 添加原始文件名作为后备路径
                possible_paths.append(os.path.join(self.storage_dir, "specific", filename))
            
            # 3. 检查注册的共享路径（如果存在）
            if model_type == 'shared' and 'filename' in model_info:
                shared_path = os.path.join(self.storage_dir, "shared", filename)
                possible_paths.append(shared_path)
            
            # 查找存在的文件路径
            found_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_path = path
                    break
            
            # 计算文件大小
            if found_path:
                # 获取文件大小（MB）
                size = os.path.getsize(found_path) / (1024 * 1024)
                model_sizes[model_id] = size
                
                # 基于模型大小和类型智能生成计算需求
                if model_type == 'shared':
                    # 共享模型通常更轻量，计算需求较低
                    computation_reqs[model_id] = max(0.5, min(1.5, size * 0.1 + np.random.uniform(-0.2, 0.2)))
                else:
                    # 特定模型可能有更高计算需求
                    computation_reqs[model_id] = max(0.7, min(2.0, size * 0.15 + np.random.uniform(-0.3, 0.3)))
                
                # print(f"模型 ID:{model_id} ({model_type}, {os.path.basename(found_path)}) "
                #     f"大小: {size:.2f} MB | 计算需求: {computation_reqs[model_id]:.2f}")
            else:
                # 文件不存在，使用默认值
                default_size = np.random.uniform(5, 10)
                model_sizes[model_id] = default_size
                
                # 根据模型类型生成默认计算需求
                if model_type == 'shared':
                    computation_reqs[model_id] = np.random.uniform(0.5, 1.2)
                else:
                    computation_reqs[model_id] = np.random.uniform(0.8, 2.0)
                
                print(f"警告: 模型 ID:{model_id} ({model_type}, {filename}) 文件不存在（尝试路径: {', '.join(possible_paths)}），"
                    f"使用默认大小 {default_size:.2f} MB 和计算需求 {computation_reqs[model_id]:.2f}")
        
        return model_sizes, computation_reqs
    
    def _initialize_rho_m(self, available_models):
        # 预定义模型计算强度映射表（单位：FLOPs per MB）
        MODEL_COMPLEXITY_TABLE = {
            'mobilenet': 3e8,
            'resnet18': 1.8e9,
            'resnet50': 4e9,
            'vit-small': 7e9,
            'bert-small': 1e10
        }
        # 初始化计算强度数组（大小为总模型数量）
        rho_m_list = [1e9] * len(available_models)  # 默认值1e9
        
        print("\n===== 使用模型信息初始化计算强度 rho_m =====")
        
        # 遍历所有可用模型
        for model_info in available_models:
            model_id = model_info['id']
            model_name = model_info['model'].lower()  # 使用模型名称字段
            
            # 添加ID有效性检查
            if model_id < 0 or model_id >= len(rho_m_list):
                print(f"错误: 无效模型ID {model_id}，超出范围 [0, {len(rho_m_list)-1}]")
                continue
                
            # 尝试匹配预定义模型类型
            matched = False
            for model_key in MODEL_COMPLEXITY_TABLE:
                if model_key in model_name:
                    rho_m = MODEL_COMPLEXITY_TABLE[model_key]
                    rho_m_list[model_id] = rho_m
                    print(f"模型 ID:{model_id} ({model_name}, {model_info['type']}) -> 识别为 {model_key}，rho_m = {rho_m:.1e} FLOPs/MB")
                    matched = True
                    break
            
            # 如果未匹配到预定义类型，使用默认值
            if not matched:
                print(f"警告 模型 ID:{model_id} ({model_name}, {model_info['type']}) -> 未识别类型，使用默认值 rho_m = 1e9 FLOPs/MB")
        
        return rho_m_list
    
    def get_model_classes_model_paths(self):
        # 获取所有 filename
        model_classes = [item['filename'] for item in self.available_models]

        # 获取所有 dataset
        model_paths = [item['dataset'] for item in self.available_models]
        return model_classes, model_paths