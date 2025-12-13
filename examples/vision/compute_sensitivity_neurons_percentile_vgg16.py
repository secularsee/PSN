# python compute_sensitivity_neurons_percentile_vgg16.py

import pickle
import torch
import os

eps = 0.031
sensitivity_save_path = f'sensitivity_scores_vgg16_transform_random_norm/sensitivity_scores_allwise_vgg16.pkl'
percent = 0.01 # 选择前 10% 的敏感神经元
#top_k = 9
output_path = f'sensitivity_scores_percentile_global_vgg16_allwise_eps0.031_new_transform_random_norm/global_topk_neurons_vgg16_top{percent}.pkl'

# 层名映射字典，根据你模型和数据实际情况调整
# 例子：将原key映射到目标key（字符串数字）
layer_name_map = {
    "/input": "features.0",
    "/input-4": "features.1",
    "/102": "features.2",
    "/input-8": "features.3",
    "/input-12": "features.4",
    "/107": "features.5",    
    "/input-16": "features.6",
    "/input-20": "features.7",
    "/input-24": "features.8",
    "/113": "features.9",
    "/input-28": "features.10",
    "/input-32": "features.11",
    "/118": "features.12",    
    "/input-36": "features.13",
    "/input-40": "features.14",
    "/input-44": "features.15",
    "/124": "features.16",
    "/input-48": "features.17",
    "/input-52": "features.18",
    "/129": "features.19",
    "/input-56": "features.20",
    "/134": "features.21",
    "/input-60": "features.22",    
    "/input-64": "features.23",
    "/input-68": "features.24",
    "/input-72": "features.25",
    "/140": "features.26",
    "/input-76": "features.27",
    "/input-80": "features.28",
    "/145": "features.29",
    "/input-84": "features.30",
    "/input-88": "features.31",
    "/150": "features.32",
    "/input-92": "features.33",
    "/input-96": "features.34",
    "/input-100": "features.35",
    "/156": "features.36",
    "/input-104": "features.37",
    "/input-108": "features.38",
    "/161": "features.39",
    "/input-112": "features.40",
    "/input-116": "features.41",
    "/166": "features.42",
    "/input-120": "features.43",
    "/168": "avgpool",
    "/169": "flatten",
    "/input-124": "classifier.0",
    "/171": "classifier.1",
    "/174": "classifier.2",
    "/input-128": "classifier.3",
    "/177": "classifier.4",
    "/180": "classifier.5",
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

