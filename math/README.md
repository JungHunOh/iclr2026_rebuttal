# Training Configuration System

This directory contains a YAML-based configuration system for running training experiments with different models and methods.

## Structure

```
configs/
├── gemma_base.yaml
├── gemma_loraga.yaml
├── gemma_lorapro.yaml
├── gemma_odlora.yaml
├── llama3_base.yaml
├── llama3_loraga.yaml
├── llama3_lorapro.yaml
└── llama3_odlora.yaml
```

## Usage

### Basic Usage

Run training with a specific config file:

```bash
python3 run.py configs/gemma_base.yaml --gpu 0
```

### Multi-GPU Usage

Specify multiple GPUs:

```bash
python3 run.py configs/llama3_base.yaml --gpu 0,1,2,3
```

### Configuration Files

Each YAML config file contains all the necessary parameters for training:

- **model**: Model identifier (gemma, llama3)
- **base_model**: HuggingFace model path
- **method**: Training method (base, loraga, lorapro, odlora)
- **bs**: Batch size
- **lr**: Learning rate
- **mini_bs**: Per-device batch size
- **r**: List of LoRA rank values to run
- **scale**: LoRA alpha scaling factor (fixed at 4)
- **max_steps**: Maximum training steps (-1 for epoch-based)
- **target_modules**: LoRA target modules

### Example Config File

```yaml
model: gemma
base_model: google/gemma-2b
method: base
bs: 32
dataset: metamath100k
dataset_name: MetaMathQA-395K
lr: 2e-5
mini_bs: 16
epoch: 1
seed: 1
target_modules: q_proj k_proj v_proj down_proj up_proj o_proj gate_proj
target_modules_name: qkvodownupgate
r: [8, 32]
scale: 4
max_steps: -1
lora_dropout: 0.05
```

### Output

Training outputs will be saved to:
```
./trained_models/{model}_{dataset}_epoch{epoch}_bs{bs}_r{r}_scale{scale}_lr{lr}_seed{seed}_{method}_{target_modules_name}/
```

Each output directory contains:
- **log.txt**: Training logs
- Model checkpoints and adapters

### Requirements

Make sure you have PyYAML installed:

```bash
pip install PyYAML
```

### Customization

To create a new configuration:

1. Copy an existing config file
2. Modify the parameters as needed
3. Run with `python3 run.py your_config.yaml --gpu 0`
