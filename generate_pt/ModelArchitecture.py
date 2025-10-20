
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

class SplitModel(nn.Module):
    def __init__(self, split_blocks, split_point):
        super().__init__()
        self.split_blocks = nn.ModuleList(split_blocks)
        self.split_point = split_point

    def forward(self, x, split=-1):
        if split == -1:
            for block in self.split_blocks:
                x = block(x)
        elif split == 0:
            for block in self.split_blocks[:self.split_point+1]:
                x = block(x)
        elif split == 1:
            for block in self.split_blocks[self.split_point+1:]:
                x = block(x)
        return x

    @property
    def shared_layers(self):
        return nn.Sequential(*self.split_blocks[:self.split_point+1])
    
    @property
    def specific_layers(self):
        return nn.Sequential(*self.split_blocks[self.split_point+1:])

def MODEL_CONFIGS():
    return {
        # ResNet 系列
        'resnet18': {
            'constructor': models.resnet18,
            'modifier': lambda m, nc: setattr(m, 'fc', nn.Linear(512, nc)),
            'split_blocks': lambda m: [
                nn.Sequential(m.conv1, m.bn1, nn.ReLU(), m.maxpool),
                m.layer1, m.layer2, m.layer3, m.layer4,
                nn.Sequential(m.avgpool, nn.Flatten(), m.fc)
            ],
            'split_point': 2,
            'input_size': 32
        },
        'resnet34': {
            'constructor': models.resnet34,
            'modifier': lambda m, nc: setattr(m, 'fc', nn.Linear(512, nc)),
            'split_blocks': lambda m: [
                nn.Sequential(m.conv1, m.bn1, nn.ReLU(), m.maxpool),
                m.layer1, m.layer2, m.layer3, m.layer4,
                nn.Sequential(m.avgpool, nn.Flatten(), m.fc)
            ],
            'split_point': 2,
            'input_size': 32
        },
        'resnet50': {
            'constructor': models.resnet50,
            'modifier': lambda m, nc: setattr(m, 'fc', nn.Linear(2048, nc)),
            'split_blocks': lambda m: [
                nn.Sequential(m.conv1, m.bn1, nn.ReLU(), m.maxpool),
                m.layer1, m.layer2, m.layer3, m.layer4,
                nn.Sequential(m.avgpool, nn.Flatten(), m.fc)
            ],
            'split_point': 2,
            'input_size': 32
        },
        'resnet101': {
            'constructor': models.resnet101,
            'modifier': lambda m, nc: setattr(m, 'fc', nn.Linear(2048, nc)),
            'split_blocks': lambda m: [
                nn.Sequential(m.conv1, m.bn1, nn.ReLU(), m.maxpool),
                m.layer1, m.layer2, m.layer3, m.layer4,
                nn.Sequential(m.avgpool, nn.Flatten(), m.fc)
            ],
            'split_point': 2,
            'input_size': 32
        },
        'resnet152': {
            'constructor': models.resnet152,
            'modifier': lambda m, nc: setattr(m, 'fc', nn.Linear(2048, nc)),
            'split_blocks': lambda m: [
                nn.Sequential(m.conv1, m.bn1, nn.ReLU(), m.maxpool),
                m.layer1, m.layer2, m.layer3, m.layer4,
                nn.Sequential(m.avgpool, nn.Flatten(), m.fc)
            ],
            'split_point': 2,
            'input_size': 32
        },
        # VGG
        'vgg16': {
            'constructor': models.vgg16,
            'modifier': lambda m, nc: setattr(m.classifier[6], 'out_features', nc),
            'split_blocks': lambda m: [
                m.features,
                nn.Sequential(nn.AdaptiveAvgPool2d((7,7)), nn.Flatten(), m.classifier)
            ],
            'split_point': 0,
            'input_size': 32
        },
        # MobileNet v2
        'mobilenet': {
            'constructor': models.mobilenet_v2,
            'modifier': lambda m, nc: setattr(m.classifier[1], 'out_features', nc),
            'split_blocks': lambda m: [
                m.features,
                nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), 
                nn.Flatten(), 
                m.classifier)
            ],
            'split_point': 0,
            'input_size': 96  # 原始为224，缩小以适应CIFAR
        }
    }

def create_model(model_name, num_classes):
    config = MODEL_CONFIGS()[model_name]
    model = config['constructor'](pretrained=False)
    
    config['modifier'](model, num_classes)
    split_blocks = config['split_blocks'](model)
    return SplitModel(split_blocks, config['split_point']), config
