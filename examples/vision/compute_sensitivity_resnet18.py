# CUDA_VISIBLE_DEVICES=1 python compute_sensitivity_resnet18.py
import os
from collections import defaultdict

# 新增 argparse 导入，用于处理命令行参数
import argparse 
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
from torchvision import datasets, transforms
import random
import numpy as np
import multiprocessing # 保持原有导入

def zscore(x, eps=1e-9):
# ... (zscore 函数不变) ...
    mu = x.mean()
    std = x.std() + eps
    return (x - mu) / std

# ResNet 模块定义 (BasicBlock, Bottleneck, ResNet, ResNet18 不变)

class BasicBlock(nn.Module):
    # ... (BasicBlock 定义不变) ...
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    # ... (Bottleneck 定义不变) ...
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    # ... (ResNet 定义不变) ...
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def ResNet18(in_planes=64):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_planes=in_planes)


def main():
    # ----------------------------------------------------
    # 💥 新增: 命令行参数解析 (解决硬编码问题)
    # ----------------------------------------------------
    import argparse
    parser = argparse.ArgumentParser(description="ResNet18 Layer Sensitivity Analysis via Auto-LiRPA")
    
    # 鲁棒性/边界参数
    parser.add_argument('--eps', type=float, default=8/255, help='Lp-norm perturbation radius (epsilon). Default: 8/255.')
    parser.add_argument('--norm', type=str, default='inf', help='Lp-norm type (e.g., inf or 2). Default: inf.')
    
    # 数据加载参数
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader. Default: 1.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of random training samples to analyze. Default: 1000.')
    parser.add_argument('--max_batches', type=int, default=1, help='Maximum number of batches to process. Default: 1.')
    
    # 路径和设备参数
    parser.add_argument('--save_path', type=str, 
                        default='sensitivity_scores_new_resnet18_transform_random_norm/sensitivity_scores_allwise_resnet18.pkl',
                        help='Path to save the computed sensitivity scores.')
    parser.add_argument('--checkpoint_path', type=str, 
                        default='saved_models/resnet18_epoch179_acc92.97.pth',
                        help='Path to the model checkpoint file.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use. Use -1 for CPU.')

    args = parser.parse_args()

    # ----------------------------------------------------
    # 🚀 使用参数值替换硬编码
    # ----------------------------------------------------
    
    # 扰动参数
    eps = args.eps
    norm = float(args.norm) if args.norm != 'inf' else np.inf

    # 路径
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
    model = ResNet18(in_planes=64)
    # 使用参数化的 checkpoint_path
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu') 
    
    # 新建一个dict，去掉module.
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉 'module.' 7个字符
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device).eval() # 确保模型被移动到正确的设备并处于评估模式


    random.seed(42)

    # 数据集准备 (标准化和增强不变)
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),
        normalize])


    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    # 使用参数化的 num_samples
    indices = random.sample(range(len(train_data)), args.num_samples) 
    subset = torch.utils.data.Subset(train_data, indices)

    train_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size, # 使用参数化的 batch_size
        shuffle=False 
    )

    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    train_loader.mean = test_loader.mean = torch.tensor([0.4914, 0.4822, 0.4465])
    train_loader.std = test_loader.std = torch.tensor([0.2023, 0.1994, 0.2010])

    # lirpa_model 初始化 (获取 batch_size 的第一个样本的形状)
    initial_input, _ = next(iter(train_loader))
    lirpa_model = BoundedModule(model, initial_input.to(device), device=device)
    lirpa_model.eval()


    # 辅助函数 (register_hooks, find_next_node) - 保持不变
    def register_hooks(model):
        for name, module in model.named_modules():
            def hook_fn(m, inp, out, layer_name=name): 
                print(f"{layer_name}: input={inp[0].shape}, output={out.shape}")
            module.register_forward_hook(hook_fn)

    # register_hooks(model) # 默认注释，需要时启用

    def find_next_node(node_name):
        next_node = None
        for node in all_nodes:
            if len(node.inputs) != 0:
                if any(inp.name == node_name for inp in node.inputs):
                    next_node = node.name
                    break
        return next_node

    # A 矩阵需求准备 - 保持不变
    # 注意: ptb = PerturbationLpNorm(norm=norm, eps=eps) 被移到循环内以适应不同的 data_lb/ub
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
    for batch_idx, (image, _) in enumerate(tqdm(train_loader, desc="processing")):
        if batch_idx >= args.max_batches:
            break
            
        # 边界计算逻辑 (使用参数化的 eps 和 norm)
        if norm == np.inf:
            std_gpu = train_loader.std.to(device).view(1, -1, 1, 1)
            mean_gpu = train_loader.mean.to(device).view(1, -1, 1, 1)
            
            data_max = torch.reshape((1. - mean_gpu) / std_gpu, (1, -1, 1, 1))
            data_min = torch.reshape((0. - mean_gpu) / std_gpu, (1, -1, 1, 1))
            
            image_device = image.to(device).to(torch.float32) 
            
            # 使用 args.eps
            data_ub = torch.min(image_device + (args.eps / std_gpu), data_max) 
            data_lb = torch.max(image_device - (args.eps / std_gpu), data_min)
        else:
            image_device = image.to(device).to(torch.float32)
            data_ub = data_lb = image_device 

        # 创建 PerturbationLpNorm (使用参数化的 eps 和 norm)
        ptb = PerturbationLpNorm(norm=norm, eps=args.eps, x_L=data_lb, x_U=data_ub)
        
        # 将 image 转换为 BoundedTensor
        image = BoundedTensor(image_device, ptb) 
        
        # ... (计算 bounds 和 A 矩阵) ...
        with torch.no_grad():
            lb, ub, A_dict = lirpa_model.compute_bounds(x=(image,), method='backward', return_A=True, needed_A_dict=required_A)
            torch.cuda.empty_cache()
            intermediate_bounds = lirpa_model.save_intermediate()

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
            print(f"Batch {batch_idx} ERROR: {e}")

        torch.cuda.empty_cache()
        gc.collect()

    # 结果保存
    final_scores = {layer: torch.cat(score_list, dim=0) for layer, score_list in sensitivities.items()}
    with open(sensitivity_save_path, 'wb') as f:
        pickle.dump(final_scores, f)

    print("saved")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()