import torch
import torch.nn as nn
from collections import OrderedDict
from ModelArchitecture import create_model

def get_model_layers(model):
    """递归获取模型的所有层"""
    layers = []
    
    def _get_layers(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # 如果是基本层类型，直接添加
            if isinstance(child, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, 
                                nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d,
                                nn.Flatten)):
                layers.append((full_name, child))
            # 如果是Sequential或ModuleList，递归展开
            elif isinstance(child, (nn.Sequential, nn.ModuleList)):
                _get_layers(child, full_name)
            # 对于其他复杂模块，也递归展开
            else:
                _get_layers(child, full_name)
    
    _get_layers(model)
    return layers

def compare_layers(layers1, layers2):
    """比较两个层列表的相似性"""
    if not layers1 or not layers2:
        return 0
    
    # 使用较短的层列表长度作为分母，确保百分比有意义
    min_length = min(len(layers1), len(layers2))
    identical_count = 0
    
    for i in range(min_length):
        layer1 = layers1[i][1]  # 获取层对象
        layer2 = layers2[i][1]  # 获取层对象
        
        # 检查层类型是否相同
        if type(layer1) != type(layer2):
            continue
            
        # 根据层类型检查关键参数
        if isinstance(layer1, nn.Conv2d):
            if (layer1.in_channels == layer2.in_channels and
                layer1.out_channels == layer2.out_channels and
                layer1.kernel_size == layer2.kernel_size and
                layer1.stride == layer2.stride and
                layer1.padding == layer2.padding):
                identical_count += 1
                
        elif isinstance(layer1, nn.Linear):
            if (layer1.in_features == layer2.in_features and
                layer1.out_features == layer2.out_features):
                identical_count += 1
                
        elif isinstance(layer1, nn.BatchNorm2d):
            if layer1.num_features == layer2.num_features:
                identical_count += 1
                
        elif isinstance(layer1, (nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Flatten)):
            # 这些层通常没有需要比较的参数，或者参数固定
            identical_count += 1
    
    return (identical_count / min_length) * 100

def get_relationship(model1, model2, model_families):
    """确定两个模型的关系类型"""
    if model1 == model2:
        return "Same Model"
    
    # 检查是否属于同一家族
    for family_name, family_models in model_families.items():
        if model1 in family_models and model2 in family_models:
            return "Same Family"
    
    # 检查是否有相似的主干
    if ('resnet' in model1 and 'resnet' in model2) or \
       ('vgg' in model1 and 'vgg' in model2) or \
       ('mobilenet' in model1 and 'mobilenet' in model2):
        return "Similar Backbone"
    
    return "Different Family"

def analyze_all_model_pairs():
    """分析所有模型对之间的相似性"""
    # 定义所有要测试的模型
    models_to_test = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg16', 'mobilenet']
    
    # 定义模型家族
    model_families = {
        'resnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
        'vgg': ['vgg16'],
        'mobilenet': ['mobilenet']
    }
    
    models = {}
    model_layers = {}
    
    # 创建所有模型并提取层信息
    print("Creating models and extracting layers...")
    for model_name in models_to_test:
        model, _ = create_model(model_name, num_classes=10)
        models[model_name] = model
        layers = get_model_layers(model)
        model_layers[model_name] = layers
        print(f"  {model_name}: {len(layers)} layers")
    
    # 分析所有模型对
    print("\n" + "="*80)
    print("COMPLETE MODEL PAIRWISE SIMILARITY ANALYSIS")
    print("="*80)
    
    # 创建相似性矩阵
    similarity_matrix = {}
    relationship_matrix = {}
    
    for i, model1 in enumerate(models_to_test):
        similarity_matrix[model1] = {}
        relationship_matrix[model1] = {}
        
        for model2 in models_to_test:
            if model1 == model2:
                similarity = 100.0  # 相同模型总是100%
            else:
                similarity = compare_layers(model_layers[model1], model_layers[model2])
            
            relationship = get_relationship(model1, model2, model_families)
            
            similarity_matrix[model1][model2] = similarity
            relationship_matrix[model1][model2] = relationship
    
    # 打印相似性矩阵
    print("\nSIMILARITY MATRIX (Percentage of Architecturally Identical Layers)")
    print("-" * 100)
    
    # 表头
    header = "Model".ljust(15)
    for model in models_to_test:
        header += model.ljust(12)
    print(header)
    print("-" * 100)
    
    # 矩阵内容
    for model1 in models_to_test:
        row = model1.ljust(15)
        for model2 in models_to_test:
            similarity = similarity_matrix[model1][model2]
            row += f"{similarity:.1f}%".ljust(12)
        print(row)
    
    # 按关系类型分组显示结果
    print("\n" + "="*80)
    print("ANALYSIS BY RELATIONSHIP TYPE")
    print("="*80)
    
    relationship_types = ["Same Model", "Same Family", "Similar Backbone", "Different Family"]
    
    for rel_type in relationship_types:
        print(f"\n--- {rel_type.upper()} ---")
        
        for model1 in models_to_test:
            matches = []
            for model2 in models_to_test:
                if relationship_matrix[model1][model2] == rel_type and model1 != model2:
                    similarity = similarity_matrix[model1][model2]
                    matches.append(f"{model2}({similarity:.1f}%)")
            
            if matches:
                print(f"{model1}: {', '.join(matches)}")
    
    # 计算每种关系的平均相似性
    print("\n" + "="*80)
    print("AVERAGE SIMILARITY BY RELATIONSHIP TYPE")
    print("="*80)
    
    for rel_type in relationship_types:
        similarities = []
        for model1 in models_to_test:
            for model2 in models_to_test:
                if relationship_matrix[model1][model2] == rel_type:
                    similarities.append(similarity_matrix[model1][model2])
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            print(f"{rel_type}: {avg_similarity:.1f}% (based on {len(similarities)} comparisons)")
    
    # 生成与原图类似的输出格式
    print("\n" + "="*80)
    print("OUTPUT IN ORIGINAL FIGURE FORMAT")
    print("="*80)
    
    # 为每个模型显示四种关系类型的百分比
    for model in models_to_test:
        print(f"\n{model}")
        
        # 计算每种关系的平均相似性
        for rel_type in relationship_types:
            similarities = []
            for other_model in models_to_test:
                if relationship_matrix[model][other_model] == rel_type:
                    similarities.append(similarity_matrix[model][other_model])
            
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                print(f"  {rel_type}: {avg_similarity:.1f}%")
            else:
                print(f"  {rel_type}: N/A")
    
    return similarity_matrix, relationship_matrix

def count_layer_types(model):
    """统计模型中各类型层的数量"""
    layer_counts = {}
    
    def _count_layers(module):
        for child in module.children():
            layer_type = type(child).__name__
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
            _count_layers(child)
    
    _count_layers(model)
    return layer_counts

if __name__ == "__main__":
    # 运行完整的模型对分析
    similarity_matrix, relationship_matrix = analyze_all_model_pairs()
    
    # 可选：显示每个模型的层统计
    print("\n" + "="*80)
    print("LAYER TYPE STATISTICS FOR EACH MODEL")
    print("="*80)
    
    models_to_test = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg16', 'mobilenet']
    for model_name in models_to_test:
        model, _ = create_model(model_name, num_classes=10)
        counts = count_layer_types(model)
        total_layers = sum(counts.values())
        print(f"{model_name}: {total_layers} total layers - {counts}")
