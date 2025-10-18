import os
import json
import pickle
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from tqdm import tqdm
import cv2
import os, ssl, certifi

# 设置环境变量
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# 设置全局 SSL context
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

from torchvision.datasets import CIFAR100
import random

def load_cifar100_hierarchy(data_root='./data'):
    """通过解析元数据及训练数据获取层级关系"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载元数据
    meta_path = os.path.join(script_dir, 'cifar-100-python', 'meta')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    
    # 解析标签名称
    coarse_labels = [name.decode('utf-8').replace(' ', '_') 
                    for name in meta[b'coarse_label_names']]
    fine_labels = [name.decode('utf-8').replace(' ', '_') 
                  for name in meta[b'fine_label_names']]
    
    # 关键修正：从训练数据中提取细到粗的映射关系
    train_file = os.path.join(script_dir, 'cifar-100-python', 'train')
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f, encoding='bytes')
    
    # 构建映射字典：确保每个细类索引对应唯一的粗类索引
    fine_to_coarse = {}
    for fine_idx, coarse_idx in zip(train_data[b'fine_labels'], train_data[b'coarse_labels']):
        if fine_idx not in fine_to_coarse:
            fine_to_coarse[fine_idx] = coarse_idx
    
    # 转换为列表格式，索引为细类编号，值为粗类编号
    fine_to_coarse_list = [fine_to_coarse[i] for i in range(len(fine_labels))]
    
    return {
        'coarse_labels': coarse_labels,
        'fine_labels': fine_labels,
        'fine_to_coarse': fine_to_coarse_list
    }

# ================== 19种缺陷生成函数 ==================
def gaussian_noise(img, severity):
    """高斯噪声"""
    sigma = [0.04, 0.06, 0.08, 0.10, 0.12][severity-1]
    img_np = np.array(img) / 255.0
    noise = np.random.normal(0, sigma, img_np.shape)
    noisy_img = np.clip(img_np + noise, 0, 1) * 255
    return Image.fromarray(noisy_img.astype(np.uint8))

def impulse_noise(img, severity):
    """脉冲噪声"""
    ratios = [0.01, 0.03, 0.05, 0.07, 0.09][severity-1]
    img_np = np.array(img)
    mask = np.random.choice([0, 1, 2], size=img_np.shape[:2], 
                          p=[ratios/2, ratios/2, 1-ratios])
    img_np[mask == 0] = 0    # 黑色像素
    img_np[mask == 1] = 255  # 白色像素
    return Image.fromarray(img_np)

def shot_noise(img, severity):
    """散粒噪声"""
    lam = [5, 10, 20, 30, 40][severity-1]
    img_np = np.array(img) / 255.0
    noise = np.random.poisson(lam * img_np) / lam
    noisy_img = np.clip(noise, 0, 1) * 255
    return Image.fromarray(noisy_img.astype(np.uint8))

def gaussian_blur(img, severity):
    """高斯模糊"""
    kernel_sizes = [1, 3, 5, 7, 9][severity-1]
    return img.filter(ImageFilter.GaussianBlur(kernel_sizes))

def motion_blur(img, severity):
    """运动模糊"""
    angles = [0, 45, 90, 135, 180][severity-1]
    kernel_size = 9
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size//2, :] = 1
    M = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angles, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / kernel.sum()
    
    img_np = np.array(img)
    blurred = cv2.filter2D(img_np, -1, kernel)
    return Image.fromarray(blurred)

def defocus_blur(img, severity):
    """散焦模糊"""
    radius = [0.1, 0.5, 1.0, 1.5, 2.0][severity-1]
    return img.filter(ImageFilter.GaussianBlur(radius * 5))

def fog_effect(img, severity):
    """雾模拟"""
    density = [0.5, 1.0, 1.5, 2.0, 2.5][severity-1]
    img_np = np.array(img)
    h, w, _ = img_np.shape
    fog = np.ones((h, w, 3), dtype=np.uint8) * 200
    fog = cv2.GaussianBlur(fog, (101, 101), 0)
    fog = (fog * density).astype(np.uint8)
    foggy_img = cv2.addWeighted(img_np, 1 - density/3, fog, density/3, 0)
    return Image.fromarray(foggy_img)

def snow_effect(img, severity):
    """雪模拟"""
    densities = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    
    # 创建雪花
    snow = np.zeros((h, w), dtype=np.float32)
    for _ in range(int(1000 * densities)):
        x, y = random.randint(0, w-1), random.randint(0, h-1)
        size = random.randint(1, 3)
        cv2.circle(snow, (x, y), size, (1), -1)
    
    # 添加模糊效果
    snow = cv2.GaussianBlur(snow, (5, 5), 0)
    snow = np.stack([snow]*3, axis=-1)
    snowy_img = np.clip(img_np + snow * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(snowy_img)

def frost_effect(img, severity):
    """霜模拟"""
    opacities = [0.3, 0.4, 0.5, 0.6, 0.7][severity-1]
    
    # 创建霜纹理
    frost = Image.new('RGB', img.size, (200, 220, 240))
    frost = frost.filter(ImageFilter.GaussianBlur(3))
    
    # 混合霜纹理
    return Image.blend(img, frost, opacities)

def contrast_adjust(img, severity):
    """对比度调整"""
    factors = [0.4, 0.6, 0.8, 1.2, 1.5][severity-1]
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factors)

def brightness_adjust(img, severity):
    """亮度调整"""
    deltas = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(1 + deltas)

def pixelate(img, severity):
    """像素化"""
    factors = [2, 4, 6, 8, 10][severity-1]
    small = img.resize((img.width//factors, img.height//factors), Image.NEAREST)
    return small.resize(img.size, Image.NEAREST)

def jpeg_compress(img, severity):
    """JPEG压缩失真"""
    qualities = [10, 20, 30, 40, 50][severity-1]
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=qualities)
    buffer.seek(0)
    return Image.open(buffer)

def elastic_transform(img, severity):
    """弹性变换"""
    alpha = [1, 2, 3, 4, 5][severity-1] * img.width
    sigma = [0.05, 0.06, 0.07, 0.08, 0.09][severity-1] * img.width
    random_state = np.random.RandomState(seed=42)
    
    shape = img.size[1], img.size[0]
    dx = random_state.rand(*shape) * 2 - 1
    dy = random_state.rand(*shape) * 2 - 1
    
    dx = cv2.GaussianBlur(dx, (17, 17), sigma)
    dy = cv2.GaussianBlur(dy, (17, 17), sigma)
    dx = (dx * alpha).astype(np.float32)
    dy = (dy * alpha).astype(np.float32)
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    img_np = np.array(img)
    distorted = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR)
    return Image.fromarray(distorted)

# ====== 额外的4种缺陷 ======
def rotation(img, severity):
    """旋转"""
    angles = [5, 10, 15, 20, 25][severity-1]
    return img.rotate(angles, fillcolor=(128, 128, 128))

def shear(img, severity):
    """剪切变形"""
    shears = [0.05, 0.1, 0.15, 0.2, 0.25][severity-1]
    return img.transform(
        img.size, 
        Image.AFFINE, 
        (1, shears, 0, 0, 1, 0),
        fillcolor=(128, 128, 128)
    )

def color_shift(img, severity):
    """颜色偏移"""
    shifts = [10, 20, 30, 40, 50][severity-1]
    img_np = np.array(img)
    # 随机选择通道偏移
    channel = random.randint(0, 2)
    img_np[:, :, channel] = np.clip(img_np[:, :, channel] + shifts, 0, 255)
    return Image.fromarray(img_np)

def lens_distortion(img, severity):
    """透镜畸变"""
    distortions = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    
    # 创建畸变映射
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            # 归一化坐标
            x = (j - w/2) / (w/2)
            y = (i - h/2) / (h/2)
            
            # 径向畸变
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            r_distorted = r * (1 + distortions * r**2)
            
            # 转换回像素坐标
            x_distorted = r_distorted * np.cos(theta)
            y_distorted = r_distorted * np.sin(theta)
            
            map_x[i, j] = (x_distorted + 1) * w/2
            map_y[i, j] = (y_distorted + 1) * h/2
    
    # 应用畸变
    distorted = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR)
    return Image.fromarray(distorted)

# 所有缺陷类型字典
CORRUPTIONS = {
    # 15种常见缺陷
    'gaussian_noise': gaussian_noise,
    'impulse_noise': impulse_noise,
    'shot_noise': shot_noise,
    'gaussian_blur': gaussian_blur,
    'motion_blur': motion_blur,
    'defocus_blur': defocus_blur,
    'fog_effect': fog_effect,
    'snow_effect': snow_effect,
    'frost_effect': frost_effect,
    'contrast_adjust': contrast_adjust,
    'brightness_adjust': brightness_adjust,
    'pixelate': pixelate,
    'jpeg_compress': jpeg_compress,
    'elastic_transform': elastic_transform,
    
    # 4种额外缺陷
    'rotation': rotation,
    'shear': shear,
    'color_shift': color_shift,
    'lens_distortion': lens_distortion
}

def convert_cifar100_to_gemel_format(output_dir="datasets"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载层级关系
    hierarchy = load_cifar100_hierarchy()
    coarse_labels = hierarchy['coarse_labels']
    fine_labels = hierarchy['fine_labels']
    fine_to_coarse = hierarchy['fine_to_coarse']
    
    # 创建目录结构
    def create_dirs(split):
        for coarse_idx, coarse_name in enumerate(coarse_labels):
            # 创建超类目录
            coarse_dir = os.path.join(output_dir, split, coarse_name)
            os.makedirs(coarse_dir, exist_ok=True)
            
            # 创建对应的子类目录
            for fine_idx, fine_name in enumerate(fine_labels):
                if fine_to_coarse[fine_idx] == coarse_idx:
                    os.makedirs(os.path.join(coarse_dir, fine_name), exist_ok=True)
    
    create_dirs("train")
    create_dirs("test")
    
    # 处理数据集
    def process_split(train=True):
        dataset = CIFAR100(root='./data', train=True, download=True)

        split_name = "train" if train else "test"
        
        print(f"Processing {split_name} set...")
        for idx in tqdm(range(len(dataset))):
            img, fine_idx = dataset[idx]
            
            # 转换图像格式
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            
            # 获取对应标签
            coarse_idx = fine_to_coarse[fine_idx]
            coarse_name = coarse_labels[coarse_idx]
            fine_name = fine_labels[fine_idx]
            
            # 保存路径
            save_dir = os.path.join(output_dir, split_name, coarse_name, fine_name)
            
            # 保存原始图像
            img.save(os.path.join(save_dir, f"{idx}.jpg"))
            
            # 仅对测试集添加19种缺陷
            if not train:
                for corruption_name, corruption_func in CORRUPTIONS.items():
                    for severity in range(1, 6):  # 5个严重等级
                        try:
                            # 应用缺陷
                            corrupted_img = corruption_func(img.copy(), severity)
                            
                            # 保存缺陷图像
                            filename = f"{idx}_{corruption_name}_s{severity}.jpg"
                            corrupted_img.save(os.path.join(save_dir, filename))
                        except Exception as e:
                            print(f"Error applying {corruption_name} level {severity} to image {idx}: {e}")
    
    process_split(train=True)
    process_split(train=False)
    
    # 生成元数据
    metadata = {
        "hierarchy": {coarse: [] for coarse in coarse_labels},
        "statistics": {
            "superclasses": len(coarse_labels),
            "subclasses": len(fine_labels),
            "samples": {
                "train": 50000,
                "test": 10000 * (1 + len(CORRUPTIONS) * 5)  # 原始图像 + 19缺陷×5等级
            }
        },
        "corruptions": {
            "count": len(CORRUPTIONS),
            "types": list(CORRUPTIONS.keys()),
            "severity_levels": 5
        }
    }
    
    for fine_idx in range(len(fine_labels)):
        coarse_name = coarse_labels[fine_to_coarse[fine_idx]]
        metadata["hierarchy"][coarse_name].append(fine_labels[fine_idx])
    
    with open(os.path.join(output_dir, "hierarchy.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("转换成功！验证目录结构：")
    print(f"{output_dir}/")
    print("├── train/")
    print("│   ├── aquatic_mammals/")
    print("│   │   ├── beaver/")
    print("│   │   ├── dolphin/ ")
    print("│   │   └── ...")
    print("├── test/")
    print("│   ├── aquatic_mammals/")
    print("│   │   ├── beaver/")
    print("│   │   │   ├── 0.jpg                    # 原始图像")
    print("│   │   │   ├── 0_gaussian_noise_s1.jpg   # 缺陷图像")
    print("│   │   │   ├── 0_gaussian_noise_s2.jpg")
    print("│   │   │   └── ... (共10000×19×5+10000=960,000张测试图像)")
    print("└── hierarchy.json")
    print(f"已添加{len(CORRUPTIONS)}种缺陷类型，每种有5个严重等级")

if __name__ == "__main__":
    import io  # 用于JPEG压缩
    convert_cifar100_to_gemel_format(
        output_dir="dataset_formation/"
    )