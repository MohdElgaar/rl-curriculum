# RL Curriculum - IF-RLVR Training Setup

This repository contains scripts and configuration for training language models using Instruction Following Reinforcement Learning with Verifiable Rewards (IF-RLVR) based on [IFBench](https://arxiv.org/pdf/2507.02833) and [open-instruct](https://github.com/allenai/open-instruct).

## 📁 Project Structure

```
rl_curriculum/
├── scripts/                    # Training and setup scripts
│   ├── train_if_rlvr.sh       # Main IF-RLVR training script (multi-GPU)
│   ├── train_if_rlvr_single_gpu.sh  # Single GPU debugging script
│   ├── setup_docker.sh        # Docker image build script
│   └── run_docker.sh          # Run training in Docker
├── configs/                    # Custom training configurations
├── outputs/                    # Training outputs and checkpoints
├── open-instruct/             # Open-instruct repository
├── IFBench/                   # IFBench evaluation suite
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- **Hardware**: NVIDIA GPU(s) with CUDA support (8 GPUs recommended for full training, 1 GPU for testing)
- **Software**: 
  - Docker (for containerized training)
  - OR Python 3.10+ with uv package manager (for native training)
  - Git

### Option 1: Training with Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   cd /home/mohamed/rl_curriculum
   bash scripts/setup_docker.sh
   ```

2. **Run training:**
   ```bash
   # Set environment variables (optional)
   export WANDB_API_KEY="your_wandb_key"  # For experiment tracking
   export HF_TOKEN="your_hf_token"        # For downloading models
   
   # Run training
   bash scripts/run_docker.sh
   ```

### Option 2: Native Training with uv

1. **Install uv package manager:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies:**
   ```bash
   cd /home/mohamed/rl_curriculum/open-instruct
   uv sync
   ```

3. **Run training:**
   ```bash
   # For multi-GPU training (8 GPUs)
   cd /home/mohamed/rl_curriculum/open-instruct
   bash ../scripts/train_if_rlvr.sh
   
   # OR for single GPU testing
   bash ../scripts/train_if_rlvr_single_gpu.sh
   ```

## 🎯 Training Scripts

### Multi-GPU Training (`train_if_rlvr.sh`)

Full-scale training on 8 GPUs with the following defaults:
- **Model**: `allenai/Llama-3.1-Tulu-3-8B-DPO`
- **Dataset**: `allenai/IF_multi_constraints_upto5`
- **Episodes**: 2,000,000
- **GPUs**: 8
- **Batch Settings**: 48 unique prompts × 16 samples per prompt

**Customize via environment variables:**
```bash
# Example: Train with different model and fewer episodes
export MODEL_NAME="allenai/OLMo-2-1124-7B-DPO"
export TOTAL_EPISODES=500000
export EXP_NAME="my_custom_experiment"
bash scripts/train_if_rlvr.sh
```

**Available configuration variables:**
- `EXP_NAME`: Experiment name (default: `if_rlvr_tulu3_8b_grpo`)
- `MODEL_NAME`: Base model to train (default: `allenai/Llama-3.1-Tulu-3-8B-DPO`)
- `TRAIN_DATASET`: Training dataset (default: `allenai/IF_multi_constraints_upto5`)
- `NUM_GPUS`: Number of GPUs (default: `8`)
- `LEARNING_RATE`: Learning rate (default: `5e-7`)
- `TOTAL_EPISODES`: Total training episodes (default: `2000000`)
- `TEMPERATURE`: Sampling temperature (default: `1.0`)
- `BETA`: KL penalty coefficient (default: `0.01`)
- `OUTPUT_DIR`: Output directory (default: `../outputs/${EXP_NAME}`)

### Single GPU Training (`train_if_rlvr_single_gpu.sh`)

Lightweight training for debugging and testing:
- **Model**: `Qwen/Qwen3-0.6B` (smaller for faster iteration)
- **Dataset**: `allenai/IF_multi_constraints_upto5`
- **Episodes**: 200
- **GPU**: 1

**Usage:**
```bash
cd /home/mohamed/rl_curriculum/open-instruct
bash ../scripts/train_if_rlvr_single_gpu.sh
```

## 🔧 Advanced Configuration

### Custom Training Parameters

Edit the training scripts or override via environment variables:

```bash
# Example: Custom learning rate and model
export LEARNING_RATE=1e-6
export MODEL_NAME="meta-llama/Llama-3.1-8B"
export NUM_UNIQUE_PROMPTS=64
export NUM_SAMPLES_PER_PROMPT=32
bash scripts/train_if_rlvr.sh
```

### Using Custom Datasets

To train on your own dataset:

1. Format your dataset following the IF-RLVR structure (see `allenai/IF_multi_constraints_upto5` on HuggingFace)
2. Upload to HuggingFace or use local path
3. Set the dataset name:
   ```bash
   export TRAIN_DATASET="your_username/your_dataset"
   bash scripts/train_if_rlvr.sh
   ```

### Monitoring Training

The scripts automatically enable Weights & Biases tracking with `--with_tracking`. Set your API key:

```bash
export WANDB_API_KEY="your_key_here"
```

Or disable tracking by removing `--with_tracking` from the training script.

## 📊 Evaluation

### Using IFBench

After training, evaluate your model on IFBench:

1. **Generate model outputs:**
   ```bash
   cd /home/mohamed/rl_curriculum/IFBench
   # Generate outputs with your trained model
   # (You'll need to adapt this to your inference setup)
   ```

2. **Run evaluation:**
   ```bash
   python3 -m run_eval \
       --input_data=data/IFBench_test.jsonl \
       --input_response_data=your_model_outputs.jsonl \
       --output_dir=eval
   ```

See the [IFBench README](IFBench/README.md) for more details.

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

- Reduce `NUM_UNIQUE_PROMPTS` or `NUM_SAMPLES_PER_PROMPT`
- Decrease `MAX_TOKEN_LENGTH` or `RESPONSE_LENGTH`
- Enable `--gradient_checkpointing` (already enabled by default)
- Use smaller model for testing

### Docker Issues

- **GPU not accessible**: Make sure nvidia-docker is installed
  ```bash
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update && sudo apt-get install -y nvidia-docker2
  sudo systemctl restart docker
  ```

- **Build fails**: Check Docker has enough disk space and memory

### Training Crashes

- Check GPU memory with `nvidia-smi`
- Review logs in the output directory
- Try single GPU script first for debugging

## 📚 Resources

- **IFBench Paper**: [Generalizing Verifiable Instruction Following](https://arxiv.org/pdf/2507.02833)
- **Open-Instruct**: [GitHub Repository](https://github.com/allenai/open-instruct)
- **Tulu 3 Paper**: [Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)
- **Training Data**: [IF_multi_constraints_upto5](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5)
- **IFBench Test Data**: [IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test)

## 🎓 Citation

If you use this code or the IFBench dataset, please cite:

```bibtex
@misc{pyatkin2025generalizing,
   title={Generalizing Verifiable Instruction Following}, 
   author={Valentina Pyatkin and Saumya Malik and Victoria Graf and Hamish Ivison and Shengyi Huang and Pradeep Dasigi and Nathan Lambert and Hannaneh Hajishirzi},
   year={2025},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025}
}
```

```bibtex
@article{lambert2024tulu3,
  title = {Tülu 3: Pushing Frontiers in Open Language Model Post-Training},
  author = {Nathan Lambert and Jacob Morrison and Valentina Pyatkin and Shengyi Huang and Hamish Ivison and Faeze Brahman and Lester James V. Miranda and Alisa Liu and Nouha Dziri and Shane Lyu and Yuling Gu and Saumya Malik and Victoria Graf and Jena D. Hwang and Jiangjiang Yang and Ronan Le Bras and Oyvind Tafjord and Chris Wilhelm and Luca Soldaini and Noah A. Smith and Yizhong Wang and Pradeep Dasigi and Hannaneh Hajishirzi},
  year = {2024},
  email = {tulu@allenai.org}
}
```

## 📝 License

- **Code**: Apache 2.0 (see [LICENSE](open-instruct/LICENSE))
- **IFBench Data**: ODC-BY-1.0 (see [IFBench LICENSE](IFBench/LICENSE))

## 🤝 Contributing

This is a research project. Feel free to:
- Open issues for bugs or questions
- Submit pull requests for improvements
- Share your training results and insights

---

**Note**: This setup is designed for external users (non-AI2). The Beaker-specific configurations from the original open-instruct scripts have been removed and adapted for direct execution.

