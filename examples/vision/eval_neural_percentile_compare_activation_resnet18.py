
# python eval_neural_percentile_compare_activation_resnet18.py
import torch
import torch.nn as nn
import pickle


from torch.utils.data import DataLoader
import os
from functools import partial
import torch.optim as optim
from torchvision import datasets, transforms
from functools import partial
import random
import pandas as pd
import numpy as np
from collections import defaultdict
import csv
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
# ——关键：使用 TrueType 字体嵌入，避免 PDF Type3 乱码——
matplotlib.rcParams['pdf.fonttype'] = 42       # PDF 中嵌入 TrueType
matplotlib.rcParams['ps.fonttype'] = 42        # EPS/PS 同理
matplotlib.rcParams['svg.fonttype'] = 'none'   # SVG 保持可编辑文本
matplotlib.rcParams['axes.unicode_minus'] = False

# 统一使用 Matplotlib 自带且跨平台的 DejaVu Sans（包含在 Matplotlib 里）
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
# 定义模型结构
class BasicBlock(nn.Module):
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



def compare_sensitive_activations(model, mask_dict, clean_x, adv_x, device='cuda', chunk_size=128):
    """
    Chunked forward that manually saves intermediate tensors (conv/bn/relu/add/relu_out/avgpool/flatten)
    for each chunk, moves the saved tensors to CPU and accumulates them. Returns per-layer mean abs diff
    for the coordinates in mask_dict.
    """
    model.eval()
    device = torch.device(device)

    layer_names = list(mask_dict.keys())

    N = clean_x.shape[0]

    # we'll accumulate on CPU to save GPU memory: each entry is a list of cpu tensors per chunk
    inter_clean_list = defaultdict(list)
    inter_adv_list = defaultdict(list)

    def forward_chunk(x_chunk):
        # x_chunk 在 device 上
        out = model.conv1(x_chunk)
        out = model.bn1(out)
        local = {}
        if 'conv1' in layer_names:
            local['conv1'] = out
        out = F.relu(out)
        if 'relu' in layer_names:
            local['relu'] = out

        # 遍历层
        for li, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4]):
            for bi, block in enumerate(layer):
                identity = block.shortcut(out)

                out = block.conv1(out)
                out = block.bn1(out)
                name_conv1 = f"layer{li+1}.{bi}.conv1"
                if name_conv1 in layer_names:
                    local[name_conv1] = out

                out = F.relu(out)
                name_relu = f"layer{li+1}.{bi}.relu"
                if name_relu in layer_names:
                    local[name_relu] = out

                out = block.conv2(out)
                out = block.bn2(out)
                name_conv2 = f"layer{li+1}.{bi}.conv2"
                if name_conv2 in layer_names:
                    local[name_conv2] = out

                out = out + identity
                name_add = f"layer{li+1}.{bi}.add"
                if name_add in layer_names:
                    local[name_add] = out

                out = F.relu(out)
                name_relu_out = f"layer{li+1}.{bi}.relu_out"
                if name_relu_out in layer_names:
                    local[name_relu_out] = out

        # 使用 F.adaptive_avg_pool2d 替代 model.avgpool
        out = F.adaptive_avg_pool2d(out, (1, 1))
        if 'avgpool' in layer_names:
            local['avgpool'] = out

        # 显式 flatten
        flat = torch.flatten(out, 1)
        if 'flatten' in layer_names:
            local['flatten'] = flat

        return local


    # process in chunks to avoid OOM
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        x_chunk = clean_x[start:end].to(device, non_blocking=True)
        with torch.no_grad():
            saved = forward_chunk(x_chunk)
        # move saved tensors to CPU and store per layer
        for k, t in saved.items():
            inter_clean_list[k].append(t.detach().cpu())
        torch.cuda.empty_cache()

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        x_chunk = adv_x[start:end].to(device, non_blocking=True)
        with torch.no_grad():
            saved = forward_chunk(x_chunk)
        for k, t in saved.items():
            inter_adv_list[k].append(t.detach().cpu())
        torch.cuda.empty_cache()

    # concatenate per-layer along batch dim
    inter_clean = {}
    inter_adv = {}
    for k, parts in inter_clean_list.items():
        inter_clean[k] = torch.cat(parts, dim=0)
    for k, parts in inter_adv_list.items():
        inter_adv[k] = torch.cat(parts, dim=0)

    # compute differences
    layer_abs_sum = defaultdict(float)
    layer_neuron_count = {ln: len(neurons) for ln, neurons in mask_dict.items()}
    for layer_name, neuron_list in mask_dict.items():
        clean_feat = inter_clean.get(layer_name, None)
        adv_feat = inter_adv.get(layer_name, None)
        if clean_feat is None or adv_feat is None:
            continue
        # feature shapes can be [N, C, H, W] or [N, C] (flatten)
        if clean_feat.ndim == 4:
            N_local = clean_feat.shape[0]
            for (c, h, w) in neuron_list:
                try:
                    clean_vals = clean_feat[:, c, h, w]
                    adv_vals = adv_feat[:, c, h, w]
                    layer_abs_sum[layer_name] += torch.abs(clean_vals - adv_vals).sum().item()
                except Exception:
                    pass
        elif clean_feat.ndim == 2:
            # flatten case: neuron coords given as (c,0,0) previously; treat c as index
            for (c, _, _) in neuron_list:
                try:
                    clean_vals = clean_feat[:, c]
                    adv_vals = adv_feat[:, c]
                    layer_abs_sum[layer_name] += torch.abs(clean_vals - adv_vals).sum().item()
                except Exception:
                    pass
        else:
            continue

    # get mean diff per neuron
    layer_differences = {}
    N_total = float(N)
    for layer_name, abs_sum in layer_abs_sum.items():
        k = layer_neuron_count.get(layer_name, 0)
        if k > 0 and N_total > 0:
            layer_differences[layer_name] = abs_sum / (k * N_total)
        else:
            layer_differences[layer_name] = 0.0

    return layer_differences


def add_noise_to_images(images, epsilon, noise_type='uniform'):
    if noise_type == 'uniform':
        noise = (torch.rand_like(images) - 0.5) * 2 * epsilon
    elif noise_type == 'gaussian':
        noise = torch.randn_like(images) * epsilon
    else:
        raise ValueError(f"未知噪声类型: {noise_type}")
    
   
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=images.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010], device=images.device).view(1, 3, 1, 1)
    
    images_px = images * std + mean
    noisy_images_px = images_px + noise
    noisy_images_px = torch.clamp(noisy_images_px, 0.0, 1.0)
    noisy_images_norm = (noisy_images_px - mean) / std
    return noisy_images_norm


def generate_adversarial_pixel_space(model, images_norm, labels, eps, alpha, iters, random_start=True):
    """
    在原始像素空间 [0,1] 上做 L_inf PGD。
    - images_norm: DataLoader 输出的已 Normalize 张量 (B,3,H,W)
    - 返回：与 model 输入格式一致的已 Normalize 对抗样本 (B,3,H,W)
    """
    device = images_norm.device
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)

    images_px = images_norm * std + mean
    
    if random_start:
        x_adv = images_px + torch.empty_like(images_px).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    else:
        x_adv = images_px.clone().detach()

    x_adv.requires_grad = True
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(iters):
        x_adv_norm = (x_adv - mean) / std
        outputs = model(x_adv_norm)
        loss = loss_fn(outputs, labels)

        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.detach_()
            x_adv.grad.zero_()
        loss.backward()
        grad = x_adv.grad.data

        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, images_px + eps), images_px - eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
        x_adv.requires_grad = True

    x_adv_norm_final = (x_adv - mean) / std
    return x_adv_norm_final
# =========================================================================


def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def layer_to_block(layer_name: str) -> str:
    if layer_name.startswith("conv1") or layer_name.startswith("bn1") or layer_name.startswith("relu"):
        return "Stem"
    elif layer_name.startswith("layer1"):
        return "Layer1"
    elif layer_name.startswith("layer2"):
        return "Layer2"
    elif layer_name.startswith("layer3"):
        return "Layer3"
    elif layer_name.startswith("layer4"):
        return "Layer4"
    elif layer_name.startswith("avgpool") or layer_name.startswith("flatten") or layer_name.startswith("fc"):
        return "Head"
    else:
        return "Other"


def main():
    import pandas as pd
    import numpy as np
    from collections import defaultdict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epsilon = 8/255  # 使用标准的像素空间epsilon
    alpha = 2/255    # 同样使用标准的alpha
    iters = 40       # 迭代次数可以保持
    topk_list = [0.01]
    save_dir = "eval_neural_percentile_activation_resnet18_multi_topk_fast_alllayer_correct2_norm_trend"
    os.makedirs(save_dir, exist_ok=True)

    # ====== 模型加载 ======
    checkpoint_path = "saved_models/resnet18_epoch179_acc92.97.pth"  # 你的模型权重文件路径
    checkpoint = torch.load(checkpoint_path)
    # 新建一个dict，去掉module.
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉 'module.' 7个字符
        else:
            new_state_dict[k] = v

    model = ResNet18()
    model.load_state_dict(new_state_dict)
    model.to(device).eval()

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
        "layer4.1.relu":  (512, 4, 4),                  #模型层名缺add,relu,relu_out
        "layer4.1.conv2": (512, 4, 4),
        "layer4.1.bn2":   (512, 4, 4),
        "layer4.1.add":   (512, 4, 4),
        "layer4.1.relu_out": (512, 4, 4),

        # head
        "avgpool": (512, 1, 1),
        "flatten": (512, 1, 1),     # forward 里的函数，不是 nn.Module
        #"fc":      (10,),      # 分类输出
    }


# ====== 数据集加载 ======
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    clean_x_list, y_list = [], []
    for x, y in test_loader:
        clean_x_list.append(x)
        y_list.append(y)
    clean_x = torch.cat(clean_x_list).to(device)
    y = torch.cat(y_list).to(device)


# ====== 生成对抗样本 (使用修正后的函数) ======
    print("Generating noisy images...")
    adv_x_noise = add_noise_to_images(clean_x, epsilon=epsilon, noise_type='uniform')
    print("Generating PGD adversarial images...")
    adv_x_pgd = generate_adversarial_pixel_space(model, clean_x, y, eps=epsilon, alpha=alpha, iters=iters)
    print("Adversarial images generated.")
    all_rows = []
    all_block_rows = []
    all_topk_layers = []

    for topk in topk_list:
        print(f"\n===== Processing top-{topk}% =====")
        global_topk_path = f'sensitivity_scores_percentile_global_resnet18_allwise_eps0.031_new_correct_transform_random_norm/global_topk_neurons_resnet18_top{topk}.pkl'

        with open(global_topk_path, 'rb') as f:
            global_topk = pickle.load(f)

        mask_dict = {}
        rest_dict = {}
        for layer_name, flat_indices in global_topk.items():
            C,H,W = conv_shape.get(layer_name,(0,1,1))
            positions = [(idx//(H*W), (idx%(H*W))//W, (idx%(H*W))%W) for idx in flat_indices]
            mask_dict[layer_name] = positions

        for layer, (C,H,W) in conv_shape.items():
            full_neurons = set((c,h,w) for c in range(C) for h in range(H) for w in range(W))
            mask_neurons = set(mask_dict.get(layer, []))
            rest_neurons = list(full_neurons - mask_neurons)
            sens_count = len(mask_dict.get(layer, []))
            if sens_count > 0 and len(rest_neurons) >= sens_count:
                sampled_rest = random.sample(rest_neurons, sens_count)
            else:
                sampled_rest = rest_neurons
            rest_dict[layer] = sampled_rest

        # ====== 计算激活差异 ======
        # move tensors to device in compare_sensitive_activations internally (chunked)
        diff_noise_sensitive = compare_sensitive_activations(model, mask_dict, clean_x, adv_x_noise, device=device, chunk_size=128)
        diff_noise_rest      = compare_sensitive_activations(model, rest_dict, clean_x, adv_x_noise, device=device, chunk_size=128)
        diff_pgd_sensitive   = compare_sensitive_activations(model, mask_dict, clean_x, adv_x_pgd,   device=device, chunk_size=128)
        diff_pgd_rest        = compare_sensitive_activations(model, rest_dict, clean_x, adv_x_pgd,   device=device, chunk_size=128)

        # ====== 逐层保存 ======
        for layer in diff_noise_sensitive.keys():
            ns = diff_noise_sensitive.get(layer,0.0)
            nr = diff_noise_rest.get(layer,0.0)
            ps = diff_pgd_sensitive.get(layer,0.0)
            pr = diff_pgd_rest.get(layer,0.0)
            all_rows.append({
                "layer": layer,
                "topk": topk,
                "noise_sens": round(ns,3),
                "noise_rand": round(nr,3),
                "noise_ratio": round(ns/nr,2) if nr>1e-12 else None,
                "pgd_sens": round(ps,3),
                "pgd_rand": round(pr,3),
                "pgd_ratio": round(ps/pr,2) if pr>1e-12 else None,
                "pgd_delta": round(ps-pr,3),
                "noise_delta": round(ns-nr,3),
            })

        # ====== Block 聚合 ======
        agg = defaultdict(list)
        for r in all_rows:
            if r["topk"]==topk:
                agg[layer_to_block(r["layer"])].append(r)

        for block, items in agg.items():
            if not items: continue
            def mean_safe(key):
                vals = [x[key] for x in items if x[key] is not None and np.isfinite(x[key])]
                return round(float(np.mean(vals)),3) if vals else None
            all_block_rows.append({
                "topk": topk,
                "block": block,
                "noise_sens_mean":  mean_safe("noise_sens"),
                "noise_rand_mean":  mean_safe("noise_rand"),
                "noise_ratio_mean": mean_safe("noise_ratio"),
                "pgd_sens_mean":    mean_safe("pgd_sens"),
                "pgd_rand_mean":    mean_safe("pgd_rand"),
                "pgd_ratio_mean":   mean_safe("pgd_ratio"),
                "pgd_delta_mean":   mean_safe("pgd_delta"),
                "noise_delta_mean": mean_safe("noise_delta"),
                "n_layers": len(items),
            })

        # ====== Top-5 层选取 ======
        items_top5 = sorted([r for r in all_rows if r["topk"]==topk], key=lambda r: r["pgd_delta"], reverse=True)[:5]
        all_topk_layers.extend(items_top5)

    # ====== 保存 per-layer CSV ======
    pd.DataFrame(all_rows).to_csv(os.path.join(save_dir,"per_layer_diffs_multi_topk.csv"),index=False)
    pd.DataFrame(all_topk_layers).to_csv(os.path.join(save_dir,"top5_layers_multi_topk.csv"),index=False)

    # ====== Block CSV 处理 ======
    block_df = pd.DataFrame(all_block_rows)

    block_df = block_df[block_df['block'] != 'Head']

    block_df = block_df.drop(columns=['pgd_delta_mean','noise_delta_mean','n_layers'])
    block_df.to_csv(os.path.join(save_dir,"block_summary_multi_topk.csv"), index=False)
    print(f"多 top-k CSV 已保存到 {save_dir} 下")




    def plot_layerwise_trend(df, target_topk=0.01):
        """
        绘制指定 Top-k 下，Activation Difference 随 Network Block 变化的趋势图。
        X轴: Block (Stem -> Layer1 -> Layer2 -> Layer3 -> Layer4)
        Y轴: Activation Difference
        """
        df_plot = df[df['topk'] == target_topk].copy()

        df_plot = df_plot[df_plot['block'] != 'Head']
        

        custom_order = {'Stem':0, 'Layer1':1, 'Layer2':2, 'Layer3':3, 'Layer4':4}
        

        df_plot = df_plot[df_plot['block'].isin(custom_order.keys())]
        
        # 排序
        df_plot['block_rank'] = df_plot['block'].map(custom_order)
        df_plot = df_plot.sort_values('block_rank')
        
        blocks = df_plot['block'].tolist()
        
        # 设置绘图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- 子图 1: Random Noise (Random Uniform) ---
        ax = axes[0]
        # 红色实线表示敏感神经元，灰色虚线表示随机神经元
        ax.plot(blocks, df_plot['noise_sens_mean'], marker='o', color='#d62728', linewidth=2, label='PSN (Sensitive)')
        ax.plot(blocks, df_plot['noise_rand_mean'], marker='s', color='#7f7f7f', linestyle='--', linewidth=2, label='Random Neurons')
        
        ax.set_title(f"Random Noise Perturbation (Top-{target_topk*100}%)", fontsize=14, fontweight='bold')
        ax.set_xlabel("ResNet Block (Depth)", fontsize=12)
        ax.set_ylabel("Mean Activation Deviation", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=11)
        
        # --- 子图 2: PGD Adversarial Attack ---
        ax = axes[1]
        ax.plot(blocks, df_plot['pgd_sens_mean'], marker='o', color='#d62728', linewidth=2, label='PSN (Sensitive)')
        ax.plot(blocks, df_plot['pgd_rand_mean'], marker='s', color='#7f7f7f', linestyle='--', linewidth=2, label='Random Neurons')
        
        ax.set_title(f"PGD Adversarial Attack (Top-{target_topk*100}%)", fontsize=14, fontweight='bold')
        ax.set_xlabel("ResNet Block (Depth)", fontsize=12)
        ax.set_ylabel("Mean Activation Deviation", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        # 保存图片
        filename_pdf = os.path.join(save_dir, f"layerwise_trend_top{target_topk}_resnet18.pdf")
        filename_png = os.path.join(save_dir, f"layerwise_trend_top{target_topk}_resnet18.png")
        plt.savefig(filename_pdf)
        plt.savefig(filename_png, dpi=300)
        print(f"ResNet18 逐层趋势图已保存: {filename_png}")

    plot_layerwise_trend(block_df, target_topk=0.01)





if __name__ == '__main__':
    main()
