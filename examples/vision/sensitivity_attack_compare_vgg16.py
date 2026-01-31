#python sensitivity_attack_vgg16.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import pickle
import os
import pandas as pd
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
# 定义模型结构
class VGG16(nn.Module):
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




def convert_flat_to_positions_global(global_topk_dict, conv_shape):
    pos_dict = {}
    for layer_name, flat_idxs in global_topk_dict.items():
        if layer_name not in conv_shape:
            print(f"警告: 层 {layer_name} 不在 conv_shape 中，跳过")
            continue
        C, H, W = conv_shape[layer_name]
        pos_list = []
        for flat_idx in flat_idxs:
            if flat_idx >= C * H * W:
                print(f"警告: Layer {layer_name} 的 flat_idx={flat_idx} 超出范围，卷积形状是 {C}x{H}x{W}={C*H*W}")
                continue
            c = flat_idx // (H * W)
            rem = flat_idx % (H * W)
            h = rem // W
            w = rem % W
            pos_list.append((c, h, w))
        pos_dict[layer_name] = pos_list
    return pos_dict




def modify_activation_vectorized(output, mask, mode="noise", strength=0.8):
    out = output.clone()
    B = out.size(0)

    if len(out.shape) == 4 and len(mask) > 0:  # 卷积层
        # 拆成 c,h,w
        c_list, h_list, w_list = zip(*mask)
        c_idx = torch.tensor(c_list, device=out.device)
        h_idx = torch.tensor(h_list, device=out.device)
        w_idx = torch.tensor(w_list, device=out.device)

        if mode == "noise":
            # (B, num_neurons)
            noise = torch.randn((B, len(mask)), device=out.device) * strength
            # batch 维度广播，按神经元索引赋值
            for b in range(B):
                out[b, c_idx, h_idx, w_idx] += noise[b]
        elif mode == "zero":
            for b in range(B):
                out[b, c_idx, h_idx, w_idx] = 0
        elif mode == "scale":
            for b in range(B):
                out[b, c_idx, h_idx, w_idx] *= strength

    elif len(out.shape) == 2 and len(mask) > 0:  # 全连接层
        idx = torch.tensor([i if isinstance(i,int) else i[0] for i in mask], device=out.device)
        if mode == "noise":
            noise = torch.randn((B, len(idx)), device=out.device) * strength
            out[:, idx] += noise
        elif mode == "zero":
            out[:, idx] = 0
        elif mode == "scale":
            out[:, idx] *= strength

    return out



def register_activation_modifiers_vectorized(model, mask_sensitive, mask_random, mode="noise", strength=0.8):
    hooks = []

    for layer_name, layer in model.named_modules():
        if layer_name in mask_sensitive or layer_name in mask_random:
            def hook_fn(m, i, o, ln=layer_name):
                if ln in mask_sensitive:
                    o = modify_activation_vectorized(o, mask_sensitive[ln], mode=mode, strength=strength)
                if ln in mask_random:
                    o = modify_activation_vectorized(o, mask_random[ln], mode=mode, strength=strength)
                return o
            hooks.append(layer.register_forward_hook(hook_fn))
    return hooks



def evaluate_with_activation_mod(model, data_loader, mask_sensitive, mask_random,
                                 mode="noise", strength=0.8, device="cuda"):
    model.eval()
    correct_clean, correct_sensitive, correct_random = 0, 0, 0
    total = 0

    # baseline
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(x)
        correct_clean += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    # 修改敏感神经元
    hooks = register_activation_modifiers_vectorized(model, mask_sensitive, {}, mode=mode, strength=strength)

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(x)
        correct_sensitive += (out.argmax(1) == y).sum().item()
    for h in hooks: h.remove()

    # 修改随机神经元
    hooks = register_activation_modifiers_vectorized(model, {}, mask_random, mode=mode, strength=strength)

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(x)
        correct_random += (out.argmax(1) == y).sum().item()
    for h in hooks: h.remove()

    return (correct_clean/total, correct_sensitive/total, correct_random/total)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = "CIFAR10"
    arc = "vgg16"

    percent = 0.01 # 选择前 10% 的敏感神经元
    sensitivity_path = f'sensitivity_scores_percentile_global_vgg16_allwise_eps0.031_new/global_topk_neurons_vgg16_top{percent}.pkl'
    epsilon = 8/255
    alpha = 2/255
    iters = 10
    mode = 'abs'
    rand_init = 'normal'

    conv_shape = {
        'features.0': (64, 32, 32),
        'features.1': (64, 32, 32),
        'features.2': (64, 32, 32),
        'features.3': (64, 32, 32),
        'features.4': (64, 32, 32),
        'features.5': (64, 32, 32),
        'features.6': (64, 16, 16),
        'features.7': (128, 16, 16),
        'features.8': (128, 16, 16),
        'features.9': (128, 16, 16),
        'features.10': (128, 16, 16),
        'features.11': (128, 16, 16),
        'features.12': (128, 16, 16),
        'features.13': (128, 8, 8),
        'features.14': (256, 8, 8),
        'features.15': (256, 8, 8),
        'features.16': (256, 8, 8),
        'features.17': (256, 8, 8),
        'features.18': (256, 8, 8),
        'features.19': (256, 8, 8),
        'features.20': (256, 8, 8),
        'features.21': (256, 8, 8),
        'features.22': (256, 8, 8),
        'features.23': (256, 4, 4),
        'features.24': (512, 4, 4),
        'features.25': (512, 4, 4),
        'features.26': (512, 4, 4),
        'features.27': (512, 4, 4),
        'features.28': (512, 4, 4),
        'features.29': (512, 4, 4),
        'features.30': (512, 4, 4),
        'features.31': (512, 4, 4),
        'features.32': (512, 4, 4),
        'features.33': (512, 2, 2),
        'features.34': (512, 2, 2),
        'features.35': (512, 2, 2),
        'features.36': (512, 2, 2),
        'features.37': (512, 2, 2),
        'features.38': (512, 2, 2),
        'features.39': (512, 2, 2),
        'features.40': (512, 2, 2),
        'features.41': (512, 2, 2),
        'features.42': (512, 2, 2),
        'features.43': (512, 1, 1),
        'avgpool': (512, 1, 1),   # AdaptiveAvgPool2d输出
        'flatten': (512, 1, 1),   # flatten后通常维度转换，保持同样尺寸以便索引
        'classifier.0': (4096, 1, 1),  # Linear层输出
        'classifier.1': (4096, 1, 1),
        'classifier.2': (4096, 1, 1),
        'classifier.3': (4096, 1, 1),
        'classifier.4': (4096, 1, 1),
        'classifier.5': (4096, 1, 1),    
    }






    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010)),
    ])
    # -------- 加载模型 --------

    batch_size = 128
    model = VGG16()
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'pretrained/vgg16_cifar10_best.pth'), map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    # ====== 只加载最敏感神经元，并构造 mask_sensitive 和 mask_rest_sampled ======
    with open(sensitivity_path, 'rb') as f:
        sensitive_neurons_raw = pickle.load(f)

    # 这里改成调用全局top-k的转换函数，不再用频率阈值筛选
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

    target_layers = [k for k in conv_shape.keys() if k.startswith("features.") or k.startswith("classifier.")]



    for percent in percents:
        sensitivity_path = f'sensitivity_scores_percentile_global_vgg16_allwise_eps0.031_new/global_topk_neurons_vgg16_top{percent}.pkl'
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
    save_dir = "attack_compare_vgg16_pdf_2_notitle"
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
    #plt.title("VGG16 Accuracy vs Propagation-Sensitive Neuron Ratio")
    #plt.xticks(df['percent'])
    plt.xticks(df['percent'], [f"{int(p*100)}%" for p in df['percent']], rotation=45)

    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # 保存为矢量图 PDF
# 同时保存 PNG 和 PDF
    plt.savefig(os.path.join(save_dir, "accuracy_vs_ratio_noise.png"), dpi=300)  # 光栅图
    plt.savefig(os.path.join(save_dir, "accuracy_vs_ratio_noise.pdf"), format='pdf')  # 矢量图
    plt.show()


if __name__ == "__main__":
    main()
