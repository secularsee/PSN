# CUDA_VISIBLE_DEVICES=0 python compute_sensitivity_vgg16.py
import os
from collections import defaultdict

# 导入 argparse
import argparse
import multiprocessing
import random
import time
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from tqdm import tqdm
import copy
from collections.abc import Mapping, Sequence
import pickle
import gc
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear
from torchvision import datasets, transforms
import random

# 1. 定义 VGG16 模型（不变）
class VGG16(nn.Module):
    # ... (VGG16 定义不变) ...
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def zscore(x, eps=1e-9):
    # ... (zscore 函数不变) ...
    mu = x.mean()
    std = x.std() + eps
    return (x - mu) / std

def main():

    # ----------------------------------------------------
    parser = argparse.ArgumentParser(description="VGG16 Layer Sensitivity Analysis via Auto-LiRPA")
    
    # 鲁棒性/边界参数
    parser.add_argument('--eps', type=float, default=8/255, help='Lp-norm perturbation radius (epsilon). Default: 8/255.')
    parser.add_argument('--norm', type=str, default='inf', help='Lp-norm type (e.g., inf or 2). Default: inf.')
    
    # 数据加载参数
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for data loader. Default: 1.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of random training samples to analyze. Default: 1000.')
    parser.add_argument('--max_batches', type=int, default=1000, help='Maximum number of batches to process. Default: 1000.')
    
    # 路径和设备参数
    parser.add_argument('--save_path', type=str, 
                        default='sensitivity_scores_vgg16_transform_random_norm/sensitivity_scores_allwise_vgg16.pkl',
                        help='Path to save the computed sensitivity scores.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use. Use -1 for CPU.')

    args = parser.parse_args()

    # ----------------------------------------------------
    # 🚀 使用参数值
    # ----------------------------------------------------
    
    # 将 norm 转换为 float/numpy inf
    norm = float(args.norm) if args.norm != 'inf' else np.inf
    
    sensitivity_save_path = args.save_path
    os.makedirs(os.path.dirname(sensitivity_save_path), exist_ok=True)
    print("Saving to:", os.path.abspath(sensitivity_save_path))

    # 设备设置
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # 模型加载
    model = VGG16().to(device)
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'pretrained/vgg16_cifar10_best.pth'), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval() # 确保模型处于评估模式

    # 数据集准备
    random.seed(42)
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        normalize])

    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    # 仅抽取 --num_samples 个样本
    indices = random.sample(range(len(train_data)), args.num_samples) 
    subset = torch.utils.data.Subset(train_data, indices)

    train_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size, # 使用参数化的 batch_size
        shuffle=False
    )

    # 均值和标准差设置 (用于边界计算)
    train_loader.mean = torch.tensor([0.4914, 0.4822, 0.4465])
    train_loader.std = torch.tensor([0.2023, 0.1994, 0.2010])

    # auto-LiRPA 模型封装
    # 注意: 即使 batch_size=1，也需要确保 tensor 维度正确，这里使用 next(iter(train_loader))[0] 来获取一个样本的形状
    initial_input, _ = next(iter(train_loader))
    lirpa_model = BoundedModule(model, initial_input.to(device), device=device)
    lirpa_model.eval()

    # 辅助函数 (find_next_node, register_hooks) - 保持不变
    def find_next_node(node_name):
        # ... (函数体不变) ...
        next_node = None
        for node in all_nodes:
            if len(node.inputs) != 0:
                if any(inp.name == node_name for inp in node.inputs):
                    next_node = node.name
                    break
        return next_node

    # 注册 hook（如果需要调试）
    # register_hooks(model) 

    # A 矩阵需求准备 - 保持不变
    required_A = defaultdict(set)
    all_nodes = lirpa_model.nodes()
    begin = lirpa_model.input_name[0]
    final = lirpa_model.final_node()
    for node in all_nodes:
        if node.name not in lirpa_model.root_names:
            required_A[node.name].add(begin)
        required_A[final.name].add(node.name)
    layer_order = [node.name for node in all_nodes]

    sensitivities = defaultdict(list)
    # 使用参数化的 max_batches
    for batch_idx, (image, _) in enumerate(tqdm(train_loader, desc="处理中")):
        if batch_idx >= args.max_batches:
            break

        # 边界计算逻辑 (保持不变，但使用 args.eps 和参数化的 norm)
        
        # 将 image 移动到 device
        image_device = image.to(device).to(torch.float32)

        if norm == np.inf:
            std_gpu = train_loader.std.to(device).view(1, -1, 1, 1)
            mean_gpu = train_loader.mean.to(device).view(1, -1, 1, 1)
            
            data_max = torch.reshape((1. - mean_gpu) / std_gpu, (1, -1, 1, 1))
            data_min = torch.reshape((0. - mean_gpu) / std_gpu, (1, -1, 1, 1))
            
            # 使用参数化的 eps
            data_ub = torch.min(image_device + (args.eps / std_gpu), data_max) 
            data_lb = torch.max(image_device - (args.eps / std_gpu), data_min)
        else:
            data_ub = data_lb = image_device 

        # 创建 PerturbationLpNorm
        ptb = PerturbationLpNorm(norm=norm, eps=args.eps, x_L=data_lb, x_U=data_ub)
        image = BoundedTensor(image_device, ptb) 

        # ... (计算 bounds 和 A 矩阵) ...
        with torch.no_grad():
            lb, ub, A_dict = lirpa_model.compute_bounds(x=(image,), method='backward', return_A=True, needed_A_dict=required_A)
            torch.cuda.empty_cache()
            intermediate_bounds = lirpa_model.save_intermediate()
            torch.cuda.empty_cache()

        # ... (敏感度得分计算逻辑保持不变) ...
        try:
            for i, layer_name in enumerate(layer_order):
                if i + 1 >= len(layer_order):
                    continue
                next_root = find_next_node(layer_name)
                
                if layer_name not in A_dict or begin not in A_dict[layer_name]:
                    continue
                if final.name not in A_dict or next_root not in A_dict[final.name]:
                    continue

                lA_before = A_dict[layer_name][begin]['lA'].flatten(2).to('cpu').norm(dim=2)
                uA_before = A_dict[layer_name][begin]['uA'].flatten(2).to('cpu').norm(dim=2)
                lA_after = A_dict[final.name][next_root]['lA'][A_dict[final.name][next_root]['input_to_idx'][layer_name]].flatten(2).to('cpu').norm(dim=1)
                uA_after = A_dict[final.name][next_root]['uA'][A_dict[final.name][next_root]['input_to_idx'][layer_name]].flatten(2).to('cpu').norm(dim=1)

                z1 = zscore(lA_before + uA_before)
                z2 = zscore(lA_after + uA_after)
                score = torch.sqrt(torch.clamp(z1 * z2, min=0.0) + 1e-9)

                sensitivities[layer_name].append(score.cpu())
        except Exception as e:
            print(f"Batch {batch_idx} 处理时出错: {e}")

        torch.cuda.empty_cache()
        gc.collect()

    # 结果保存
    final_scores = {layer: torch.cat(score_list, dim=0) for layer, score_list in sensitivities.items()}
    with open(sensitivity_save_path, 'wb') as f:
        pickle.dump(final_scores, f)

    print("敏感度计算完成并已保存。")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()