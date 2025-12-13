# python compute_sensitivity_neurons_percentile_resnet18.py

import pickle
import torch
import os

eps = 0.031
sensitivity_save_path = f'sensitivity_scores_new_resnet18_transform_random_norm/sensitivity_scores_allwise_resnet18.pkl'
percent = 0.001 # 选择前 10% 的敏感神经元
#top_k = 9
output_path = f'sensitivity_scores_percentile_global_resnet18_allwise_eps0.031_new_correct_transform_random_norm/global_topk_neurons_resnet18_top{percent}.pkl'

# 层名映射字典，根据你模型和数据实际情况调整
# 例子：将原key映射到目标key（字符串数字）
layer_name_map = {
    # stem
    "/input": "conv1",
    "/124": "bn1",
    "/input-4": "relu",

    # layer1.0
    "/input-8": "layer1.0.conv1",
    "/129": "layer1.0.bn1",
    "/input-12": "layer1.0.relu",
    "/input-16": "layer1.0.conv2",
    "/134": "layer1.0.bn2",
    "/137": "layer1.0.add",
    "/input-20": "layer1.0.relu_out",

    # layer1.1
    "/input-24": "layer1.1.conv1",
    "/140": "layer1.1.bn1",
    "/input-28": "layer1.1.relu",
    "/input-32": "layer1.1.conv2",
    "/145": "layer1.1.bn2",
    "/148": "layer1.1.add",
    "/input-36": "layer1.1.relu_out",

    # layer2.0 (带下采样的shortcut)
    "/input-40": "layer2.0.conv1",
    "/151": "layer2.0.bn1",
    "/input-44": "layer2.0.relu",
    "/input-48": "layer2.0.conv2",
    "/156": "layer2.0.bn2",
    "/input-52": "layer2.0.shortcut.0",
    "/160": "layer2.0.shortcut.1",
    "/163": "layer2.0.add",
    "/input-56": "layer2.0.relu_out",

    # layer2.1
    "/input-60": "layer2.1.conv1",
    "/166": "layer2.1.bn1",
    "/input-64": "layer2.1.relu",
    "/input-68": "layer2.1.conv2",
    "/171": "layer2.1.bn2",
    "/174": "layer2.1.add",
    "/input-72": "layer2.1.relu_out",

    # layer3.0 (带下采样的shortcut)
    "/input-76": "layer3.0.conv1",
    "/177": "layer3.0.bn1",
    "/input-80": "layer3.0.relu",
    "/input-84": "layer3.0.conv2",
    "/182": "layer3.0.bn2",
    "/input-88": "layer3.0.shortcut.0",
    "/186": "layer3.0.shortcut.1",
    "/189": "layer3.0.add",
    "/input-92": "layer3.0.relu_out",

    # layer3.1
    "/input-96": "layer3.1.conv1",
    "/192": "layer3.1.bn1",
    "/input-100": "layer3.1.relu",
    "/input-104": "layer3.1.conv2",
    "/197": "layer3.1.bn2",
    "/200": "layer3.1.add",
    "/input-108": "layer3.1.relu_out",

    # layer4.0 (带下采样的shortcut)
    "/input-112": "layer4.0.conv1",
    "/203": "layer4.0.bn1",
    "/input-116": "layer4.0.relu",
    "/input-120": "layer4.0.conv2",
    "/208": "layer4.0.bn2",
    "/input-124": "layer4.0.shortcut.0",
    "/212": "layer4.0.shortcut.1",
    "/215": "layer4.0.add",
    "/input-128": "layer4.0.relu_out",

    # layer4.1
    "/input-132": "layer4.1.conv1",
    "/218": "layer4.1.bn1",
    "/input-136": "layer4.1.relu",
    "/input-140": "layer4.1.conv2",
    "/223": "layer4.1.bn2",
    "/226": "layer4.1.add",
    "/227": "layer4.1.relu_out",

    # classifier（函数式算子在原始模型里不是Module，但LiRPA里有节点）
    "/228": "avgpool",
    "/229": "flatten",
    #"/230": "fc",
}


with open(sensitivity_save_path, 'rb') as f:
    sensitivity_scores = pickle.load(f)

#print("Before stopping")
#breakpoint()  # 代码将在这一行停止，进入调试模式
#print("After stopping") 

global_topk_neurons = {}

for orig_layer_name, scores in sensitivity_scores.items():
    layer_name = layer_name_map.get(orig_layer_name)
    if layer_name is None:
        print(f"[Warning] 层名 {orig_layer_name} 未在映射表中，跳过")
        continue

    assert scores.ndim == 2
    avg_scores = scores.mean(dim=0)
    
    num_neurons = avg_scores.shape[0]
    top_k = max(1, int(num_neurons * percent))  # 至少保留1个神经元

    values, indices = torch.sort(avg_scores, descending=True)
    topk_indices = indices[:top_k].tolist()
    global_topk_neurons[layer_name] = topk_indices


os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'wb') as f:
    pickle.dump(global_topk_neurons, f)

print(f"✅ 已保存每层 top-{percent} 全局敏感神经元编号到 {output_path}")

