# KL-Flow: Flow Matching for Text Generation

Training flow matching models for text generation using standard tools (HuggingFace Transformers, PyTorch, Muon optimizer).

## Quick Start

### 1. Installation

```bash
pip install -r requirements_dataset.txt
```

Or install manually:
```bash
pip install torch datasets transformers accelerate omegaconf tensorboard
pip install git+https://github.com/KellerJordan/Muon.git
```

### 2. Download Dataset

Download the dataset specified in your config:

```bash
python download_dataset.py configs/config_tinystories_unconditional.yaml
```

This will download TinyStories from HuggingFace Hub and save it locally to `./data/TinyStories/`.

After download, update your config file to point to the local path:
```yaml
data:
    dataset_path: "./data/TinyStories"  # Update to local path
```

### 3. Training

#### Unconditional Training (TinyStories)

**Single GPU:**
```bash
python train_fm.py configs/config_tinystories_unconditional.yaml
```

**Multi-GPU (8 GPUs):**
```bash
torchrun --nproc_per_node=8 train_fm.py configs/config_tinystories_unconditional.yaml
```

#### Conditional Training (Instruction Following)

**Single GPU:**
```bash
python download_dataset.py configs/config_conditional_alpaca.yaml
python train_fm.py configs/config_conditional_alpaca.yaml
```

**Multi-GPU:**
```bash
torchrun --nproc_per_node=8 train_fm.py configs/config_conditional_alpaca.yaml
```

### 4. Monitor Training

View training logs with TensorBoard:

```bash
tensorboard --logdir logs/tiny_stories_fm/logs
```

Then open http://localhost:6006 in your browser.

## Configuration

### Available Flow Matching Types

Change `fm.type` in config to use different flow matching methods:
- `Logit` - Logit-based flow matching (default, recommended)
- `LogitEntropy` - Logit with entropy regularization
- `LogitUpdated` - Updated logit flow
- `GPT` - Standard autoregressive GPT
- `OneShot` - One-shot generation
- `LogitMask` - Logit with mask sampling
- `DFM` - Discrete Flow Matching (set max_t=1.0)
- `Sphere` - Sphere-based flow
- `Dirichlet` - Dirichlet flow

### Config Structure

```yaml
data:
    dataset_path: "roneneldan/TinyStories"  # HF dataset or local path
    tokenizer_name: "gpt2"                   # HF tokenizer
    sequence_length: 512
    condition: False                         # True for prompt/response
    text_field: "text"                       # Field name for text
    prompt_field: "instruction"              # For conditional training
    response_field: "output"                 # For conditional training

model:
    type: "FlowMatchingTransformer"          # or "GPTCausal"
    vocab_size: 50304                        # From tokenizer
    n_embd: 768                              # Embedding dimension
    n_layer: 12                              # Number of transformer layers
    n_head: 12                               # Number of attention heads
    dropout: 0.1                             # Dropout probability
    max_seq_len: 2048                        # Maximum sequence length

training_config:
    num_tokens_to_train: 1.0                 # Billions of tokens
    batch_size: 128                          # Global batch size
    device_batch_size: 16                    # Per-device batch size
    
optimizer:
    muon_learning_rate: 0.002                # For 2D matrices
    embed_learning_rate: 0.00036             # For embeddings/biases
```

## Features

- ✅ **Standard Tools**: HuggingFace Trainer, PyTorch DataLoader
- ✅ **Muon Optimizer**: Fast training with hybrid Muon+Adam optimizer
- ✅ **Dynamic Tokenization**: Tokenize during batch formation
- ✅ **Special Tokens**: Automatic BOS, EOS, PAD, MASK tokens
- ✅ **Conditional Training**: Prompt/response instruction following
- ✅ **TensorBoard**: Built-in logging and monitoring
- ✅ **Multi-GPU**: Automatic DDP with torchrun
- ✅ **Two-Level Configs**: Clean experiment configs with sensible defaults

## Configuration System

KL-Flow uses a **two-level configuration system** for easy experiment management:

### Default Config (`configs/default_config.yaml`)
Contains sensible defaults for all parameters (model architecture, optimizer, FM settings, etc.). You typically don't need to modify this.

### Experiment Configs
Only specify what's unique to your experiment. See `configs/template_experiment.yaml` for a template.

**Minimal experiment config:**
```yaml
training_config:
    run_name: "my_experiment"

data:
    dataset_path: "my/dataset"
```

All other parameters are loaded from defaults and can be selectively overridden.

**For more details:** See [CONFIG_SYSTEM.md](CONFIG_SYSTEM.md) for complete documentation.

## Model Architectures

The project includes two transformer architectures built with standard PyTorch modules:

### 1. Flow Matching Transformer (`models/flow_matching_transformer.py`)
- **Bidirectional attention** for flow matching training
- **Time conditioning** via sinusoidal timestep embeddings
- **Standard PyTorch modules** for maintainability
- Use for: Flow matching, discrete diffusion models

### 2. GPT Causal (`models/gpt_causal.py`)
- **Causal attention** for autoregressive modeling
- **Rotary Position Embeddings (RoPE)** for position encoding
- **Autoregressive generation** support
- Use for: Standard language modeling, text generation

Both models feature:
- RMS normalization
- Squared ReLU activation
- Zero-initialized output projections
- Flexible configuration via constructor

See [`models/README.md`](models/README.md) for detailed documentation and usage examples.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU
- 16GB+ VRAM recommended

## Inference

To generate samples for unconditional and conditional tasks use:

```bash
CUDA_VISIBLE_DEVICES=DEVICE_IDS python inference_fm.py ./configs/config_tinystories_unconditional.yaml
```

**Automatic checkpoint and output detection:**
- Checkpoint is auto-detected from `logs/{run_name}/` directory
- Results are saved to `logs/{run_name}/inference/`
- To use a specific checkpoint, add to your config:
```yaml
inference:
    checkpoint: "./path/to/checkpoint.pt"
```

For infilling use:

```bash
CUDA_VISIBLE_DEVICES=DEVICE_IDS python inference_fm_infilling.py ./configs/config_tinystories_unconditional.yaml POSTFIX
```

Results are saved to `logs/{run_name}/inference/`

## Evaluation

All evaluation scripts automatically:
- Load inference results from `logs/{run_name}/inference/`
- Save evaluation results to `logs/{run_name}/eval_*.pt`

### Unconditional evaluation

```bash
CUDA_VISIBLE_DEVICES=DEVICE_IDS python eval_unconditional.py ./configs/config_tinystories_unconditional.yaml
```

Results saved to: `logs/{run_name}/eval_unconditional.pt`

### Conditional evaluation

```bash
CUDA_VISIBLE_DEVICES=DEVICE_IDS python eval_conditional.py ./configs/config_conditional_alpaca.yaml
```

Results saved to: `logs/{run_name}/eval_conditional.pt`

### Infilling evaluation

```bash
CUDA_VISIBLE_DEVICES=DEVICE_IDS python eval_infilling.py ./configs/config_tinystories_unconditional.yaml
```

Results saved to: `logs/{run_name}/eval_infilling.pt`

## Directory Structure

All experiment artifacts are organized under `logs/{run_name}/`:

```
logs/
└── {run_name}/                    # Your experiment name
    ├── checkpoint-1000/           # HuggingFace Trainer checkpoints
    │   ├── pytorch_model.bin
    │   └── ...
    ├── ckpt.pt                    # Final model checkpoint
    ├── inference/                 # Inference results
    │   ├── 0.pt
    │   ├── 1.pt
    │   └── ...
    ├── eval_unconditional.pt      # Evaluation results (if run)
    ├── eval_conditional.pt        # Evaluation results (if run)
    ├── eval_infilling.pt          # Evaluation results (if run)
    └── logs/                      # TensorBoard training logs
        └── events.out.tfevents.*
```

This unified structure makes it easy to:
- Find all files related to an experiment
- Clean up experiments (just delete the directory)
- Share experiment results (zip the directory)
- Compare different experiments side-by-side

## Example Configs

### Unconditional Training
- `configs/config_tinystories_unconditional.yaml` - TinyStories with GPT-2 tokenizer

### Conditional Training  
- `configs/config_conditional_alpaca.yaml` - Alpaca instruction following

## Usage Examples

### Train on Custom Dataset

1. **Create config file** (e.g., `configs/my_dataset.yaml`):
```yaml
data:
    dataset_path: "your/hf-dataset"
    tokenizer_name: "gpt2"
    sequence_length: 512
    condition: False
    text_field: "text"
```

2. **Download dataset**:
```bash
python download_dataset.py configs/my_dataset.yaml
```

3. **Update config** with local path and **train**:
```bash
torchrun --nproc_per_node=8 train_fm.py configs/my_dataset.yaml
```

### Conditional Training on Custom Dataset

1. **Create config** with prompt/response fields:
```yaml
data:
    dataset_path: "your/instruction-dataset"
    tokenizer_name: "gpt2"
    condition: True
    prompt_field: "instruction"
    response_field: "output"
```

2. **Download and train**:
```bash
python download_dataset.py configs/my_dataset.yaml
torchrun --nproc_per_node=8 train_fm.py configs/my_dataset.yaml
```

## Citation

If you use this code, please cite:
```bibtex
@misc{klflow2024,
  title={KL-Flow: Flow Matching for Text Generation},
  author={Your Name},
  year={2024},
}
```