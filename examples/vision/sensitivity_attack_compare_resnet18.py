#python sensitivity_attack_compare_resnet18.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import pickle
import os
import pandas as pd
from torchvision import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1. 模型结构修改，支持 mask_dict
# -----------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, mask_dict=None, mode="noise", strength=0.8, layer_prefix=""):
        out = self.conv1(x)
        if mask_dict and f"{layer_prefix}.conv1" in mask_dict:
            out = modify_activation_vectorized(out, mask_dict[f"{layer_prefix}.conv1"], mode, strength, f"{layer_prefix}.conv1")
        out = self.bn1(out)
        if mask_dict and f"{layer_prefix}.bn1" in mask_dict:
            out = modify_activation_vectorized(out, mask_dict[f"{layer_prefix}.bn1"], mode, strength, f"{layer_prefix}.bn1")
        out = F.relu(out)
        if mask_dict and f"{layer_prefix}.relu" in mask_dict:
            out = modify_activation_vectorized(out, mask_dict[f"{layer_prefix}.relu"], mode, strength, f"{layer_prefix}.relu")

        residual = self.shortcut(x)
        out2 = self.conv2(out)
        if mask_dict and f"{layer_prefix}.conv2" in mask_dict:
            out2 = modify_activation_vectorized(out2, mask_dict[f"{layer_prefix}.conv2"], mode, strength, f"{layer_prefix}.conv2")
        out2 = self.bn2(out2)
        if mask_dict and f"{layer_prefix}.bn2" in mask_dict:
            out2 = modify_activation_vectorized(out2, mask_dict[f"{layer_prefix}.bn2"], mode, strength, f"{layer_prefix}.bn2")

        out = out2 + residual
        if mask_dict and f"{layer_prefix}.add" in mask_dict:
            out = modify_activation_vectorized(out, mask_dict[f"{layer_prefix}.add"], mode, strength, f"{layer_prefix}.add")
        out = F.relu(out)
        if mask_dict and f"{layer_prefix}.relu_out" in mask_dict:
            out = modify_activation_vectorized(out, mask_dict[f"{layer_prefix}.relu_out"], mode, strength, f"{layer_prefix}.relu_out")
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, in_planes, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(in_planes*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for i, s in enumerate([stride]+[1]*(num_blocks-1)):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes*block.expansion
        return nn.ModuleList(layers)

    def forward(self, x, mask_dict=None, mode="noise", strength=0.8):
        out = self.conv1(x)
        if mask_dict and "conv1" in mask_dict:
            out = modify_activation_vectorized(out, mask_dict["conv1"], mode, strength, "conv1")
        out = self.bn1(out)
        if mask_dict and "bn1" in mask_dict:
            out = modify_activation_vectorized(out, mask_dict["bn1"], mode, strength, "bn1")
        out = F.relu(out)
        if mask_dict and "relu" in mask_dict:
            out = modify_activation_vectorized(out, mask_dict["relu"], mode, strength, "relu")

        for i, block in enumerate(self.layer1):
            out = block(out, mask_dict, mode, strength, f"layer1.{i}")
        for i, block in enumerate(self.layer2):
            out = block(out, mask_dict, mode, strength, f"layer2.{i}")
        for i, block in enumerate(self.layer3):
            out = block(out, mask_dict, mode, strength, f"layer3.{i}")
        for i, block in enumerate(self.layer4):
            out = block(out, mask_dict, mode, strength, f"layer4.{i}")

        out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        if mask_dict and "linear" in mask_dict:
            out = modify_activation_vectorized(out, mask_dict["linear"], mode, strength, "linear")
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def modify_activation_vectorized(output, mask, mode="noise", strength=0.8, layer_name=None):
    out = output.clone()
    B = out.size(0)
    if len(out.shape)==4 and len(mask)>0:  # conv
        c_list, h_list, w_list = zip(*mask)
        c_idx = torch.tensor(c_list, device=out.device)
        h_idx = torch.tensor(h_list, device=out.device)
        w_idx = torch.tensor(w_list, device=out.device)
        if mode=="noise":
            noise = torch.randn((B,len(mask)), device=out.device)*strength
            for b in range(B):
                out[b,c_idx,h_idx,w_idx] += noise[b]
        elif mode=="zero":
            for b in range(B):
                out[b,c_idx,h_idx,w_idx] = 0
        elif mode=="scale":
            for b in range(B):
                out[b,c_idx,h_idx,w_idx] *= strength
    elif len(out.shape)==2 and len(mask)>0:  # fc
        idx = torch.tensor([i if isinstance(i,int) else i[0] for i in mask], device=out.device)
        if mode=="noise":
            noise = torch.randn((B,len(idx)), device=out.device)*strength
            out[:,idx] += noise
        elif mode=="zero":
            out[:,idx] = 0
        elif mode=="scale":
            out[:,idx] *= strength

    return out

# -----------------------------
# 3. 全局 top-k 转换
# -----------------------------
def convert_flat_to_positions_global(global_topk_dict, conv_shape):
    pos_dict = {}
    for layer_name, flat_idxs in global_topk_dict.items():
        if layer_name not in conv_shape:
            print(f"警告: 层 {layer_name} 不在 conv_shape 中，跳过")
            continue
        C,H,W = conv_shape[layer_name]
        pos_list=[]
        for flat_idx in flat_idxs:
            if flat_idx>=C*H*W:
                continue
            c = flat_idx // (H*W)
            rem = flat_idx % (H*W)
            h = rem // W
            w = rem % W
            pos_list.append((c,h,w))
        pos_dict[layer_name]=pos_list
    return pos_dict

def block_forward_with_mask(model, x, mask_dict, mode="noise", strength=0.8):
    out = x
    for name, module in model.named_modules():
        # 跳过顶层模块（Sequential 等）
        if len(list(module.children())) > 0:
            continue

        out = module(out)
        if name in mask_dict:
            out = modify_activation_vectorized(out, mask_dict[name], mode=mode, strength=strength, layer_name=name)
    return out


def evaluate_with_activation_mod(model, data_loader, mask_sensitive, mask_random,
                                 mode="noise", strength=0.8, device="cuda"):
    model.eval()
    correct_clean, correct_sensitive, correct_random = 0,0,0
    total=0

    # baseline
    for x,y in data_loader:
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(x)
        correct_clean += (out.argmax(1)==y).sum().item()
        total += y.size(0)

    # 修改敏感神经元
    for x,y in data_loader:
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(x, mask_dict=mask_sensitive, mode=mode, strength=strength)
        correct_sensitive += (out.argmax(1)==y).sum().item()

    # 修改随机神经元
    for x,y in data_loader:
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(x, mask_dict=mask_random, mode=mode, strength=strength)
        correct_random += (out.argmax(1)==y).sum().item()

    return correct_clean/total, correct_sensitive/total, correct_random/total

def apply_mask_in_block(out, mask_s, mask_r, *layer_names):
    """
    对一组层名依次尝试修改激活
    """
    for ln in layer_names:
        out = modify_activation_vectorized(out, mask_s.get(ln, []), mode="noise", strength=0.8)
        out = modify_activation_vectorized(out, mask_r.get(ln, []), mode="noise", strength=0.8)
    return out


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = "CIFAR10"
    arc = "resnet18"

    percent = 0.01 # 选择前 10% 的敏感神经元
    sensitivity_path = f'sensitivity_scores_percentile_global_resnet18_allwise_eps0.031_new/global_topk_neurons_resnet18_top{percent}.pkl'
    epsilon = 8/255
    alpha = 2/255
    iters = 10
    mode = 'abs'
    rand_init = 'normal'
    conv_shape = {
        # stem
        "conv1":  (64, 32, 32),
        "bn1":    (64, 32, 32),
        "relu":   (64, 32, 32),

        # layer1 (64 channels, stride=1)
        "layer1.0.conv1": (64, 32, 32),
        "layer1.0.bn1":   (64, 32, 32),
        "layer1.0.relu":  (64, 32, 32),
        "layer1.0.conv2": (64, 32, 32),
        "layer1.0.bn2":   (64, 32, 32),
        "layer1.0.add":   (64, 32, 32),
        "layer1.0.relu_out": (64, 32, 32),

        "layer1.1.conv1": (64, 32, 32),
        "layer1.1.bn1":   (64, 32, 32),
        "layer1.1.relu":  (64, 32, 32),
        "layer1.1.conv2": (64, 32, 32),
        "layer1.1.bn2":   (64, 32, 32),
        "layer1.1.add":   (64, 32, 32),
        "layer1.1.relu_out": (64, 32, 32),

        # layer2 (128 channels, stride=2)
        "layer2.0.conv1": (128, 16, 16),
        "layer2.0.bn1":   (128, 16, 16),
        "layer2.0.relu":  (128, 16, 16),
        "layer2.0.conv2": (128, 16, 16),
        "layer2.0.bn2":   (128, 16, 16),
        "layer2.0.shortcut.0": (128, 16, 16),
        "layer2.0.shortcut.1": (128, 16, 16),
        "layer2.0.add":   (128, 16, 16),
        "layer2.0.relu_out": (128, 16, 16),

        "layer2.1.conv1": (128, 16, 16),
        "layer2.1.bn1":   (128, 16, 16),
        "layer2.1.relu":  (128, 16, 16),
        "layer2.1.conv2": (128, 16, 16),
        "layer2.1.bn2":   (128, 16, 16),
        "layer2.1.add":   (128, 16, 16),
        "layer2.1.relu_out": (128, 16, 16),

        # layer3 (256 channels, stride=2)
        "layer3.0.conv1": (256, 8, 8),
        "layer3.0.bn1":   (256, 8, 8),
        "layer3.0.relu":  (256, 8, 8),
        "layer3.0.conv2": (256, 8, 8),
        "layer3.0.bn2":   (256, 8, 8),
        "layer3.0.shortcut.0": (256, 8, 8),
        "layer3.0.shortcut.1": (256, 8, 8),
        "layer3.0.add":   (256, 8, 8),
        "layer3.0.relu_out": (256, 8, 8),

        "layer3.1.conv1": (256, 8, 8),
        "layer3.1.bn1":   (256, 8, 8),
        "layer3.1.relu":  (256, 8, 8),
        "layer3.1.conv2": (256, 8, 8),
        "layer3.1.bn2":   (256, 8, 8),
        "layer3.1.add":   (256, 8, 8),
        "layer3.1.relu_out": (256, 8, 8),

        # layer4 (512 channels, stride=2)
        "layer4.0.conv1": (512, 4, 4),
        "layer4.0.bn1":   (512, 4, 4),
        "layer4.0.relu":  (512, 4, 4),
        "layer4.0.conv2": (512, 4, 4),
        "layer4.0.bn2":   (512, 4, 4),
        "layer4.0.shortcut.0": (512, 4, 4),
        "layer4.0.shortcut.1": (512, 4, 4),
        "layer4.0.add":   (512, 4, 4),
        "layer4.0.relu_out": (512, 4, 4),

        "layer4.1.conv1": (512, 4, 4),
        "layer4.1.bn1":   (512, 4, 4),
        "layer4.1.relu":  (512, 4, 4),                 
        "layer4.1.conv2": (512, 4, 4),
        "layer4.1.bn2":   (512, 4, 4),
        "layer4.1.add":   (512, 4, 4),
        "layer4.1.relu_out": (512, 4, 4),

        # head
        "avgpool": (512, 1, 1),
        "flatten": (512, 1, 1),     
    }







    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010)),
    ])
    # -------- 加载模型 --------

    batch_size = 128

    checkpoint_path = "saved_models/resnet18_epoch179_acc92.97.pth"  # 你的模型权重文件路径
    checkpoint = torch.load(checkpoint_path)
    # 新建一个dict，去掉module.
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  
        else:
            new_state_dict[k] = v

    model = ResNet18()
    model.load_state_dict(new_state_dict)
    model.to(device).eval()
    #for name, module in model.named_modules():
        #print(name, module)
    # ====== 只加载最敏感神经元，并构造 mask_sensitive 和 mask_rest_sampled ======
    with open(sensitivity_path, 'rb') as f:
        sensitive_neurons_raw = pickle.load(f)


    mask_sensitive = convert_flat_to_positions_global(sensitive_neurons_raw, conv_shape)


    all_pos_dict = {}
    for layer, (C, H, W) in conv_shape.items():
        pos_list = []
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    pos_list.append((c, h, w))
        all_pos_dict[layer] = pos_list


    mask_rest = {}
    for layer, all_pos in all_pos_dict.items():
        sens_pos = set(mask_sensitive.get(layer, []))
        rest_pos = list(set(all_pos) - sens_pos)
        mask_rest[layer] = rest_pos


    mask_rest_sampled = {}
    for layer in mask_sensitive:
        sens_pos = mask_sensitive[layer]
        rest_pos = mask_rest.get(layer, [])
        if len(rest_pos) >= len(sens_pos):
            mask_rest_sampled[layer] = random.sample(rest_pos, len(sens_pos))
        else:
            mask_rest_sampled[layer] = rest_pos



    # ====== 数据集加载 ======
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 定义敏感神经元比例列表
    percents = [0.01, 0.02, 0.05, 0.1,0.2,0.3]  # 可自行调整


    results = []

    target_layers = list(conv_shape.keys())

    for percent in percents:
        sensitivity_path = f'sensitivity_scores_percentile_global_resnet18_allwise_eps0.031_new_correct/global_topk_neurons_resnet18_top{percent}.pkl'
        with open(sensitivity_path, 'rb') as f:
            sensitive_neurons_raw = pickle.load(f)

        mask_sensitive = convert_flat_to_positions_global(sensitive_neurons_raw, conv_shape)

        # 生成 mask_rest_sampled
        mask_rest = {}
        for layer, all_pos in all_pos_dict.items():
            sens_pos = set(mask_sensitive.get(layer, []))
            rest_pos = list(set(all_pos) - sens_pos)
            mask_rest[layer] = rest_pos

        mask_rest_sampled = {}
        for layer in mask_sensitive:
            sens_pos = mask_sensitive[layer]
            rest_pos = mask_rest.get(layer, [])
            if len(rest_pos) >= len(sens_pos):
                mask_rest_sampled[layer] = random.sample(rest_pos, len(sens_pos))
            else:
                mask_rest_sampled[layer] = rest_pos

        mask_sensitive_layer = {k: v for k, v in mask_sensitive.items() if k in target_layers}
        mask_rest_sampled_layer = {k: v for k, v in mask_rest_sampled.items() if k in target_layers}

        # 只计算 noise 模式
        acc_clean, acc_sensitive, acc_random = evaluate_with_activation_mod(
            model, test_loader,
            mask_sensitive_layer,
            mask_rest_sampled_layer,
            mode="noise",
            strength=0.8,
            device=device
        )

        results.append({
            "percent": percent,
            "acc_clean": acc_clean*100,
            "acc_sensitive": acc_sensitive*100,
            "acc_random": acc_random*100,
            "drop_sensitive": (acc_clean-acc_sensitive)*100,
            "drop_random": (acc_clean-acc_random)*100
        })

    # 保存 CSV 前，将数值转为百分比并保留两位小数
    for r in results:
        r["acc_clean"] = round(r["acc_clean"], 2)
        r["acc_sensitive"] = round(r["acc_sensitive"], 2)
        r["acc_random"] = round(r["acc_random"], 2)
        r["drop_sensitive"] = round(r["drop_sensitive"], 2)
        r["drop_random"] = round(r["drop_random"], 2)

    # 保存 CSV
    save_dir = "attack_compare_resnet18_correct_pdf_full_test_2_notitle"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "sensitivity_attack_noise.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"结果已保存到 {csv_path}")

    # 绘制折线图
    plt.figure(figsize=(6,4))
    plt.plot(df['percent'], df['acc_clean'], marker='o', label='Clean Accuracy')
    plt.plot(df['percent'], df['acc_sensitive'], marker='s', label='Sensitive Modified Accuracy')
    plt.plot(df['percent'], df['acc_random'], marker='^', label='Random Modified Accuracy')
    plt.xlabel("Propagation-Sensitive Neuron Ratio (%)", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    #plt.title("resnet18 Accuracy vs Propagation-Sensitive Neuron Ratio")
    #plt.xticks(df['percent'])
    plt.xticks(df['percent'], [f"{int(p*100)}%" for p in df['percent']], rotation=45)

    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
# 同时保存 PNG 和 PDF
    plt.savefig(os.path.join(save_dir, "accuracy_vs_ratio_noise.png"), dpi=300)  # 光栅图
    plt.savefig(os.path.join(save_dir, "accuracy_vs_ratio_noise.pdf"), format='pdf')  # 矢量图
    plt.show()


if __name__ == "__main__":
    main()
