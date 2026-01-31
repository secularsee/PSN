#python eval_neural_percentile_compare_activation_vgg16.py
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

# ——关键：使用 TrueType 字体嵌入，避免 PDF Type3 乱码——
matplotlib.rcParams['pdf.fonttype'] = 42       # PDF 中嵌入 TrueType
matplotlib.rcParams['ps.fonttype'] = 42        # EPS/PS 同理
matplotlib.rcParams['svg.fonttype'] = 'none'   # SVG 保持可编辑文本
matplotlib.rcParams['axes.unicode_minus'] = False

# 统一使用 Matplotlib 自带且跨平台的 DejaVu Sans（包含在 Matplotlib 里）
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
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



def compare_sensitive_activations(model, mask_dict, clean_x, adv_x):
    intermediate_clean = {}
    intermediate_adv = {}

    def register_hooks(model, store_dict, layer_names):
        def hook_fn(module, input, output, layer_name, store_dict):
            # 输出直接转CPU存储，避免显存占用
            store_dict[layer_name] = output.detach().cpu().clone()
        hooks = []
        for name, module in model.named_modules():
            #print("name:")
            #print(name)
            if name in layer_names:
                hooks.append(
                    module.register_forward_hook(
                        partial(hook_fn, layer_name=name, store_dict=store_dict)
                    )
                )
        return hooks

    layer_names = set(mask_dict.keys())
    if 'flatten' in layer_names:
        layer_names.add('avgpool')

    # 捕获 clean 输入的中间特征，输入数据转到GPU，输出转CPU存储
    hooks = register_hooks(model, intermediate_clean, layer_names)
    _ = model(clean_x)
    for hook in hooks:
        hook.remove()

    hooks = register_hooks(model, intermediate_adv, layer_names)
    _ = model(adv_x)
    for hook in hooks:
        hook.remove()

    # 手动构造 flatten，已经在CPU上
    if 'flatten' in mask_dict:
        if 'avgpool' in intermediate_clean and 'avgpool' in intermediate_adv:
            intermediate_clean['flatten'] = torch.flatten(intermediate_clean['avgpool'], 1)
            intermediate_adv['flatten'] = torch.flatten(intermediate_adv['avgpool'], 1)
        else:
            print("[Warning] cannot compute 'flatten' because 'avgpool' not captured.")

    # 计算差异，均在CPU上
    layer_differences = {}
    for layer_name, neuron_list in mask_dict.items():
        clean_feat = intermediate_clean.get(layer_name)
        adv_feat = intermediate_adv.get(layer_name)
        if clean_feat is None or adv_feat is None:
            print(f"[Warning] layer {layer_name} missing in model outputs.")
            continue

        diffs = []
        for (c, h, w) in neuron_list:
            try:
                if clean_feat.ndim == 4:
                    clean_vals = clean_feat[:, c, h, w]
                    adv_vals = adv_feat[:, c, h, w]
                elif clean_feat.ndim == 2:
                    clean_vals = clean_feat[:, c]
                    adv_vals = adv_feat[:, c]
                else:
                    continue
                diff = (clean_vals - adv_vals).abs().mean().item()
                diffs.append(diff)
            except Exception as e:
                print(f"[Skip] Layer {layer_name} neuron {(c,h,w)}: {e}")

        avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
        layer_differences[layer_name] = avg_diff

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

# ==================== 新增：严谨的、在像素空间攻击的PGD函数 ====================
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

def layer_to_block(name: str) -> str:
    if name.startswith("features."):
        idx = int(name.split(".")[1])
        if 0 <= idx <= 5: return "B1"
        if 6 <= idx <= 12: return "B2"
        if 13 <= idx <= 23: return "B3"
        if 24 <= idx <= 33: return "B4"
        if 34 <= idx <= 43: return "B5"
        return "Features_Other"
    if name.startswith("classifier."):
        return "FC"
    if name in ("avgpool","flatten"):
        return "Pool/Flat"
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
    save_dir = "eval_neural_percentile_activation_vgg16_multi_topk_fast_2_norm_trend"
    os.makedirs(save_dir, exist_ok=True)

    # ====== 模型加载 ======
    model = VGG16()
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'pretrained/vgg16_cifar10_best.pth'), map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device).eval()

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
        global_topk_path = f'sensitivity_scores_percentile_global_vgg16_allwise_eps0.031_new_transform_random_norm/global_topk_neurons_vgg16_top{topk}.pkl'
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
        diff_noise_sensitive = compare_sensitive_activations(model, mask_dict, clean_x, adv_x_noise)
        diff_noise_rest = compare_sensitive_activations(model, rest_dict, clean_x, adv_x_noise)
        diff_pgd_sensitive = compare_sensitive_activations(model, mask_dict, clean_x, adv_x_pgd)
        diff_pgd_rest = compare_sensitive_activations(model, rest_dict, clean_x, adv_x_pgd)

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


    block_df = pd.DataFrame(all_block_rows)

    block_df = block_df[block_df['block'] != 'Pool/Flat']

    block_df = block_df.drop(columns=['pgd_delta_mean','noise_delta_mean','n_layers'])
    block_df.to_csv(os.path.join(save_dir,"block_summary_multi_topk.csv"), index=False)
    print(f"多 top-k CSV 已保存到 {save_dir} 下")


    def plot_layerwise_trend(df, target_topk=0.01):
        """
        绘制指定 Top-k 下，Activation Difference 随 Network Block 变化的趋势图。
        X轴: Block (B1 -> FC)
        Y轴: Activation Difference
        """
        df_plot = df[df['topk'] == target_topk].copy()
        df_plot = df_plot[df_plot['block'] != 'Pool/Flat']
        
        # 确保 Block 顺序正确
        # VGG16: B1 -> B2 -> B3 -> B4 -> B5 -> FC
        custom_order = {'B1':0, 'B2':1, 'B3':2, 'B4':3, 'B5':4, 'FC':5}
        df_plot['block_rank'] = df_plot['block'].map(custom_order)
        df_plot = df_plot.sort_values('block_rank')
        
        blocks = df_plot['block'].tolist()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- 子图 1: Random Noise (Random Uniform) ---
        ax = axes[0]
        ax.plot(blocks, df_plot['noise_sens_mean'], marker='o', color='#d62728', linewidth=2, label='PSN (Sensitive)')
        ax.plot(blocks, df_plot['noise_rand_mean'], marker='s', color='#7f7f7f', linestyle='--', linewidth=2, label='Random Neurons')
        
        ax.set_title(f"Random Noise Perturbation (Top-{target_topk*100}%)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Network Block (Depth)", fontsize=12)
        ax.set_ylabel("Mean Activation Deviation", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=11)
        
        # --- 子图 2: PGD Attack ---
        ax = axes[1]
        ax.plot(blocks, df_plot['pgd_sens_mean'], marker='o', color='#d62728', linewidth=2, label='PSN (Sensitive)')
        ax.plot(blocks, df_plot['pgd_rand_mean'], marker='s', color='#7f7f7f', linestyle='--', linewidth=2, label='Random Neurons')
        
        ax.set_title(f"PGD Adversarial Attack (Top-{target_topk*100}%)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Network Block (Depth)", fontsize=12)
        ax.set_ylabel("Mean Activation Deviation", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        # 保存图片
        filename_pdf = os.path.join(save_dir, f"layerwise_trend_top{target_topk}_vgg16.pdf")
        filename_png = os.path.join(save_dir, f"layerwise_trend_top{target_topk}_vgg16.png")
        plt.savefig(filename_pdf)
        plt.savefig(filename_png, dpi=300)
        print(f"逐层趋势图已保存: {filename_png}")

    # 调用绘图函数 (重点关注 Top-1%)
    plot_layerwise_trend(block_df, target_topk=0.01)



if __name__ == '__main__':
    main()
