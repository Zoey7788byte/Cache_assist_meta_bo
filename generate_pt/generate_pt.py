import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import json
from tqdm import tqdm
import random
from ModelArchitecture import SplitModel, create_model, MODEL_CONFIGS
import glob


# 基础配置
BASE_DATA_PATH = "./Power_aware_parameter_sharing_cache/data/dataset_formation/"
OUTPUT_DIR = "./Power_aware_parameter_sharing_cache/generate_pt/Model_weight_pt/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def check_model_exists(model_name, subclass=None):  # 修改参数名为subclass
    """检查模型文件是否已存在"""
    if subclass is None:
        # 检查预训练模型
        pretrain_path = os.path.join(OUTPUT_DIR, f"{model_name}_pretrained.pth")
        matching_files = glob.glob(pretrain_path)
        return len(matching_files) > 0
    else:
        # 检查微调模型 - 改为按子类检查
        finetune_path = os.path.join(OUTPUT_DIR, f"{subclass}_{model_name}_specific*.pth")
        matching_files = glob.glob(finetune_path)
        return len(matching_files) > 0

class CIFAR100Dataset(Dataset):
    """支持子类级别加载的数据集"""
    def __init__(self, root, split='train', mode='pretrain', target_subclass=None):  # 参数改为target_subclass
        with open(os.path.join(root, 'hierarchy.json')) as f:
            self.metadata = json.load(f)
        
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                               std=[0.2675, 0.2565, 0.2761])
        ])
        
        if mode == 'pretrain':
            self._load_all_classes(root, split)
        else:
            # 直接加载指定子类
            self._load_subclass(root, split, target_subclass)

    def _load_subclass(self, root, split, subclass_name):
        """加载指定子类的数据"""
        found = False
        for superclass in self.metadata['hierarchy']:
            if subclass_name in self.metadata['hierarchy'][superclass]:
                found = True
                subclass_dir = os.path.join(root, split, superclass, subclass_name)
                break
        
        if not found:
            raise ValueError(f"Subclass not found: {subclass_name}")
        
        if not os.path.isdir(subclass_dir):
            raise ValueError(f"Directory not found: {subclass_dir}")
        
        # 加载目标子类样本（正样本）
        for img_name in os.listdir(subclass_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subclass_dir, img_name)
                self.samples.append((img_path, 1))  # 正样本标签为1
        
        # 加载其他子类样本作为负样本
        other_samples = []
        for superclass in self.metadata['hierarchy']:
            for subclass in self.metadata['hierarchy'][superclass]:
                if subclass == subclass_name:
                    continue
                other_dir = os.path.join(root, split, superclass, subclass)
                if not os.path.isdir(other_dir):
                    continue
                for img_name in os.listdir(other_dir):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(other_dir, img_name)
                        other_samples.append((img_path, 0))  # 负样本标签为0
        
        # 平衡正负样本（可选，根据需求调整比例）
        random.shuffle(other_samples)
        self.samples += other_samples[:len(self.samples)]  # 保持1:1比例

    def _load_all_classes(self, root, split):
        """加载所有类别数据"""
        for superclass in self.metadata['hierarchy']:
            for subclass in self.metadata['hierarchy'][superclass]:
                subclass_dir = os.path.join(root, split, superclass, subclass)
                if not os.path.isdir(subclass_dir):
                    continue
                
                for img_name in os.listdir(subclass_dir):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(subclass_dir, img_name)
                        label = self.metadata['hierarchy'][superclass].index(subclass)
                        self.samples.append((img_path, label))

    def _load_superclass(self, root, split, superclass):
        """加载指定超类的数据"""
        if superclass not in self.metadata['hierarchy']:
            raise ValueError(f"Invalid superclass: {superclass}")
        
        subclasses = self.metadata['hierarchy'][superclass]
        self.class_to_idx = {name:i for i,name in enumerate(subclasses)}
        
        superclass_dir = os.path.join(root, split, superclass)
        if not os.path.isdir(superclass_dir):
            raise ValueError(f"Directory not found: {superclass_dir}")
        
        for subclass in subclasses:
            subclass_dir = os.path.join(superclass_dir, subclass)
            if not os.path.isdir(subclass_dir):
                continue
                
            for img_name in os.listdir(subclass_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(subclass_dir, img_name)
                    label = self.class_to_idx[subclass]
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), label

def train_model(model, dataloaders, optimizer, criterion, num_epochs, phase='pretrain', model_name=None, target_superclass=None):
    """通用的训练函数"""
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(dataloaders['train'], desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  
            running_loss += loss.item()
        
        # 验证
        val_acc = evaluate_model(model, dataloaders['test'])
        print(f'Loss: {running_loss/len(dataloaders["train"]):.4f} Acc: {val_acc:.2f}')
        
         # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            if phase == 'pretrain':
                save_path = os.path.join(OUTPUT_DIR, f"{model_name}_pretrained.pth")
                torch.save(model.shared_layers.state_dict(), save_path)
            else:
                save_path = os.path.join(OUTPUT_DIR, f"{target_superclass}_{model_name}_specific_acc{best_acc:.2f}.pth")
                torch.save(model.specific_layers.state_dict(), save_path)
    return best_acc 

def evaluate_model(model, dataloader):
    """模型评估函数"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def pretrain(model_name='resnet18', num_epochs=100):
    if check_model_exists(model_name):
        return

    model, config = create_model(model_name, 100)
    model = model.to(DEVICE)
    
    # 添加数据增强
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((config['input_size'], config['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])
    
    train_set = CIFAR100Dataset(BASE_DATA_PATH, 'train')
    train_set.transform = transform
    val_set = CIFAR100Dataset(BASE_DATA_PATH, 'test')  # 实际应使用验证集
    val_set.transform = transform
    
    dataloaders = {
        'train': DataLoader(train_set, batch_size=128, shuffle=True),
        'val': DataLoader(val_set, batch_size=128)  # 验证集
    }
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(dataloaders['train']):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            running_loss += loss.item()
        
        # 验证
        val_acc = evaluate_model(model, dataloaders['val'])
        print(f'Epoch {epoch+1}/{num_epochs} | Loss: {running_loss/len(dataloaders["train"]):.4f} | Acc: {val_acc:.4f}')
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(OUTPUT_DIR, f"{model_name}_pretrained.pth")
            torch.save(model.shared_layers.state_dict(), save_path)
    
    print(f"{model_name} pretraining finished. Best accuracy: {best_acc:.4f}")


def finetune(model_name, subclass_name, num_epochs=50):  # 参数改为subclass_name
    # 检查模型是否已存在（按子类检查）
    if check_model_exists(model_name, subclass_name):
        print(f"{subclass_name}_{model_name} finetuned model already exists. Skipping...")
        return
    
    # 检查预训练模型是否存在
    pretrained_path = os.path.join(OUTPUT_DIR, f"{model_name}_pretrained.pth")
    if not os.path.exists(pretrained_path):
        print(f"Pretrained model {model_name} not found. Please run pretraining first.")
        return
    
    # 创建二分类模型（输出维度为2）
    model, config = create_model(model_name, 2)  # 改为二分类
    model = model.to(DEVICE)
    model.shared_layers.load_state_dict(torch.load(pretrained_path))
    
    # 冻结共享层
    for param in model.shared_layers.parameters():
        param.requires_grad = False
    
    # 调整输入尺寸
    transform = transforms.Compose([
        transforms.Resize((config['input_size'], config['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])
    
    # 加载指定子类数据（含负样本）
    train_set = CIFAR100Dataset(BASE_DATA_PATH, 'train', 'finetune', subclass_name)
    train_set.transform = transform
    test_set = CIFAR100Dataset(BASE_DATA_PATH, 'test', 'finetune', subclass_name)
    test_set.transform = transform
    
    dataloaders = {
        'train': DataLoader(train_set, batch_size=64, shuffle=True),
        'test': DataLoader(test_set, batch_size=64)
    }
    
    optimizer = optim.Adam(model.specific_layers.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_acc = train_model(model, dataloaders, optimizer, criterion, num_epochs,
                         phase='finetune', model_name=model_name,
                         target_superclass=subclass_name)  # 改为subclass_name
    
    print(f"{subclass_name} finetuning finished. Best accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    models_name = {'resnet18','vgg16', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'mobilenet'}

    # 基础预训练（保持不变）
    print("="*50)
    print("Starting pretraining phase...")
    for model_name in models_name:
        print(f"\nProcessing {model_name}...")
        pretrain(model_name, num_epochs=100)
    
    # 下游微调 - 改为遍历所有子类
    print("\n" + "="*50)
    print("Starting finetuning phase...")
    with open(os.path.join(BASE_DATA_PATH, 'hierarchy.json')) as f:
        metadata = json.load(f)
    
    # 收集所有子类
    all_subclasses = []
    for superclass in metadata['hierarchy']:
        all_subclasses.extend(metadata['hierarchy'][superclass])
    
    # 对每个子类进行微调
    for subclass in all_subclasses:
        for model_name in models_name:
            print(f"\nProcessing {subclass} with {model_name}...")
            finetune(model_name, subclass, num_epochs=50)  # 传入子类名
