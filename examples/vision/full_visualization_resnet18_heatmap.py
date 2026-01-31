# python full_visualization_resnet18_heatmap.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from torchvision import datasets
import matplotlib.cm as cm
# ---------------------------
# VGG16 模型定义
# ---------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        self.add = nn.Identity()  # 卷积输出与 shortcut 相加包装成 Module
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.add(out + self.shortcut(x))
        out = self.relu_out(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, in_planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, in_planes*8, num_blocks[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_planes*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

# ---------------------------
# 配置
# ---------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_top_neurons = 3       # 每层敏感神经元和普通神经元各取3个
num_top_images = 5        # 每个神经元激活最高前5张图

threshold = 0.4           # Grad-CAM 激活阈值
alpha = 0.3               # 非关注区域透明度

global_top_neurons_path = 'sensitivity_scores_percentile_global_resnet18_allwise_eps0.031_new_norm/global_top3_sensitive_and_vanilla.pkl'
neuron_activation_path = 'neuron_activation_full_norm/neuron_activation_resnet18_alllayers_full.pkl'
dataset_root = './data'

# ---------------------------
# 加载敏感神经元和激活值
# ---------------------------
with open(global_top_neurons_path, 'rb') as f:
    global_top_neurons = pickle.load(f)

with open(neuron_activation_path, 'rb') as f:
    neuron_activation = pickle.load(f)

# ---------------------------
# CIFAR-10 数据集
# ---------------------------
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.CIFAR10(dataset_root, train=False, download=True, transform=transform)

# ---------------------------
# 加载模型
# ---------------------------
model = ResNet18()
checkpoint = torch.load("saved_models/resnet18_epoch179_acc92.97.pth")
new_state_dict = {}
for k, v in checkpoint.items():
    new_state_dict[k[7:] if k.startswith('module.') else k] = v
model.load_state_dict(new_state_dict)
model = model.to(device)
model.eval()

# ---------------------------
# PGD攻击
# ---------------------------
def pgd_attack(model, images, labels, eps=0.031):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    adv_images = images + eps * images.grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images.detach()
# ---------------------------
# 多步 PGD 生成对抗样本
# ---------------------------
# ---------------------------
# 多步 PGD 生成对抗样本 (修正版)
# ---------------------------
def pgd_attack_multi(model, images, labels, eps=8/255, alpha=2/255, steps=40):
    """
    多步 PGD 攻击 - 修正版
    - 在像素空间 [0,1] 上进行扰动和裁剪。
    - 在计算梯度时，将图像归一化后再送入模型。
    """

    
    # 新增：定义用于归一化的 mean 和 std
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)

    # images 输入已经是 [0,1] 像素空间，无需反归一化
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    adv_images = images.clone().detach() # adv_images 也始终保持在 [0,1]

    for _ in range(steps):
        adv_images.requires_grad = True
        

        normalized_adv = (adv_images - mean) / std
        outputs = model(normalized_adv)
        
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        # 更新和投影操作仍在 [0,1] 像素空间进行
        adv_images = adv_images.detach() + alpha * adv_images.grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images


# ---------------------------
def gradcam_mask_transparent(model, image, neuron_idx, layer):
    # --- 新增：定义用于归一化的 mean 和 std ---
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)

    image = image.unsqueeze(0).to(device)
    activations, gradients = [], []

    def forward_hook(module, input, output):
        activations.append(output)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_backward_hook(backward_hook)


    normalized_image = (image - mean) / std
    output = model(normalized_image)

    if activations[0].dim() == 4:
        C, H, W = activations[0].shape[1:4]
        c = neuron_idx // (H * W)
        rem = neuron_idx % (H * W)
        hh = rem // W
        ww = rem % W
        target = activations[0][0, c, hh, ww]
    else:
        target = activations[0][0, neuron_idx]

    model.zero_grad()
    target.backward(retain_graph=True)

    act = activations[0].detach()
    grad = gradients[0].detach()

    handle_f.remove()
    handle_b.remove()

    if act.dim() == 4:
        weights = grad.mean(dim=(2,3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam[0,0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap = cm.jet(cam)[..., :3]
        

        img_np = np.transpose(image[0].detach().cpu().numpy(), (1, 2, 0))

        superimposed_img = heatmap * 0.4 + img_np * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 1)
    else: 
        img_np = np.transpose(image[0].detach().cpu().numpy(), (1,2,0))
        superimposed_img = img_np

    return superimposed_img


def visualize_top_neurons(global_top_neurons, neuron_activation, dataset, model,
                          save_dir='visualization_resnet18_gradcam_transparent_attack3_combine_pdf_norm_heatmap',
                          top_n_images=5):
    os.makedirs(save_dir, exist_ok=True)

    for layer_name, neuron_dict in global_top_neurons.items():
        layer_modules = dict(model.named_modules())
        if layer_name not in layer_modules:
            print(f"[Warning] 层 {layer_name} 不在模型中，跳过")
            continue
        layer_module = layer_modules[layer_name]
        act_scores_full = neuron_activation[layer_name]

        for label in ['sensitive', 'vanilla']:
            neuron_list = neuron_dict[label]

            for neuron_idx in neuron_list:
                top_imgs_idx = np.argsort(act_scores_full[neuron_idx, :])[-top_n_images:]

                # 每列一组，5 列，每列上下两张图
                fig, axes = plt.subplots(2, top_n_images, figsize=(2*top_n_images, 4))  # 2行 top_n_images列

                for col, img_idx in enumerate(top_imgs_idx):
                    img, target = dataset[img_idx]

                    adv_img = pgd_attack_multi(
                        model, img.unsqueeze(0), torch.tensor([target]),
                        eps=8/255, alpha=2/255, steps=40
                    )[0]

                    masked_benign = gradcam_mask_transparent(model, img, neuron_idx, layer_module)
                    masked_adv = gradcam_mask_transparent(model, adv_img, neuron_idx, layer_module)

                    # 上行：干净图
                    axes[0, col].imshow(masked_benign)
                    axes[0, col].axis('off')

                    # 下行：对抗图
                    axes[1, col].imshow(masked_adv)
                    axes[1, col].axis('off')

                # 调整布局，保存整张大图
                plt.tight_layout()

                # 保存 PNG（方便预览）
                png_path = os.path.join(save_dir, f'{layer_name}_{label}_neuron{neuron_idx}.png')
                plt.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=300)

                # 保存 PDF（矢量图，适合论文）
                pdf_path = os.path.join(save_dir, f'{layer_name}_{label}_neuron{neuron_idx}.pdf')
                plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0, format='pdf')

                plt.close(fig)
# ---------------------------
# 调用
# ---------------------------
visualize_top_neurons(global_top_neurons, neuron_activation, test_dataset, model, top_n_images=num_top_images)
