# PSN
code for paper"Propagation-Sensitive Neurons: A Verification-Based Method for Interpreting and Enhancing Adversarial Robustness"
## Installation

### Requirements
* Python: 3.11+
* PyTorch: 2.0+

### Install Auto-LiRPA
To install the `auto-lirpa` toolkit, run the following command in the root directory:

```bash
pip install .

1. Navigate to the working directory:
   ```bash
   cd examples/vision

2. Prepare Pre-trained Models:
   Download or train the VGG16 and ResNet18 models. Place the model checkpoints into the `pretrained_models` folder.

3. **Calculate Propagation Sensitivity Scores**:
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
