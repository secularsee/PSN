# PSN

Official implementation of the paper: **"Propagation-Sensitive Neurons: A Verification-Based Method for Interpreting and Enhancing Adversarial Robustness"**.

## Installation

### Requirements
* **Python**: 3.11+
* **PyTorch**: 2.0+

### Install Auto-LiRPA
To install the `auto-lirpa` toolkit, run the following command in the root directory:

```bash
pip install .
```

### Usage


1. Navigate to the working directory:
   ```bash
   cd examples/vision

2. Prepare Pre-trained Models:
   Download or train the VGG16 and ResNet18 models. Place the model checkpoints into the `pretrained_models` folder.

3. Calculate Propagation Sensitivity Scores:
   Run the following scripts to compute the neuron-wise propagation sensitivity scores for VGG16 and ResNet18:

   ```bash
   # For VGG16
   python compute_sensitivity_vgg16.py

   # For ResNet18
   python compute_sensitivity_resnet18.py
4. Identify Top-k% Sensitive Neurons:
   Run the scripts to identify the top-k% propagation-sensitive neurons for each layer of the VGG16 and ResNet18 models:

   ```bash
   # For VGG16
   python compute_sensitivity_neurons_percentile_vgg16.py

   # For ResNet18
   python compute_sensitivity_neurons_percentile_resnet18.py

### Analysis Experiment

1. Evaluating the the activation deviations of PSNs:
   Run this script to evaluate the activation deviations of PSNs in VGG16 and ResNet18 models under noise and adversarial attacks:

   ```bash
   # For VGG16
   python eval_neural_percentile_compare_activation_vgg16.py

   # For ResNet18
   python eval_neural_percentile_compare_activation_resnet18.py

2. Visualize the focus regions of PSNs:
   Run this script to visualize the focus regions of PSNs in VGG16 and ResNet18 models under adversarial attacks:

   ```bash
   # For VGG16
   python full_visualization_vgg16_heatmap.py

   # For ResNet18
   python full_visualization_resnet18_heatmap.py

3. Evaluate the impact of attacking PSNs:

   Run this script to evaluate the impact of attacking PSNs in VGG16 and ResNet18 models on prediction results:

   ```bash
   # For VGG16
   python sensitivity_attack_compare_vgg16.py

   # For ResNet18
   python sensitivity_attack_compare_resnet18.py



