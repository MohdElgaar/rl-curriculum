# IF-RLVR Project Structure

This document provides a comprehensive overview of the project structure and all available scripts.

## 📂 Directory Layout

```
rl_curriculum/
│
├── 📄 README.md                     # Main documentation
├── 📄 QUICKSTART.md                 # Quick start guide
├── 📄 PROJECT_STRUCTURE.md          # This file
│
├── 📁 scripts/                      # All executable scripts
│   ├── train_if_rlvr.sh            # Main multi-GPU training script
│   ├── train_if_rlvr_single_gpu.sh # Single GPU debug/test script
│   ├── setup_docker.sh             # Build Docker image
│   ├── run_docker.sh               # Run training in Docker
│   └── setup_native.sh             # Native environment setup (no Docker)
│
├── 📁 configs/                      # Configuration files
│   └── training_config_template.sh  # Template for custom configs
│
├── 📁 outputs/                      # Training outputs (created automatically)
│   └── {experiment_name}/          # Per-experiment outputs
│       ├── checkpoints/            # Model checkpoints
│       ├── logs/                   # Training logs
│       └── eval/                   # Evaluation results
│
├── 📁 open-instruct/               # Open-instruct repository (clone)
│   ├── open_instruct/
│   │   ├── grpo_fast.py           # Main GRPO training script
│   │   └── IFEvalG/               # IF verification functions
│   ├── scripts/
│   ├── configs/
│   └── ...
│
└── 📁 IFBench/                     # IFBench evaluation suite (clone)
    ├── data/
    │   └── IFBench_test.jsonl     # Test dataset
    ├── run_eval.py                # Evaluation script
    └── ...
```

## 🔧 Script Reference

### Setup Scripts

| Script | Purpose | Usage | Time |
|--------|---------|-------|------|
| `setup_native.sh` | Install dependencies natively | `bash scripts/setup_native.sh` | 5-10 min |
| `setup_docker.sh` | Build Docker image | `bash scripts/setup_docker.sh` | 10-20 min |

### Training Scripts

| Script | Purpose | GPUs | Model Size | Episodes | Duration |
|--------|---------|------|------------|----------|----------|
| `train_if_rlvr_single_gpu.sh` | Quick test/debug | 1 | 0.6B | 200 | 10-30 min |
| `train_if_rlvr.sh` | Full training | 8 | 8B | 2M | 2-3 days |

### Docker Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_docker.sh` | Run training in Docker | `bash scripts/run_docker.sh` |

## 🎯 Workflow Diagrams

### Setup Workflow

```
┌─────────────────┐
│  Start Here     │
└────────┬────────┘
         │
    ┌────▼─────┐
    │ Choose:  │
    │ Docker   │
    │    or    │
    │ Native?  │
    └─┬────┬──┘
      │    │
      │    └─────────────────┐
      │                      │
┌─────▼───────┐      ┌───────▼──────┐
│   Docker    │      │    Native    │
│   Setup     │      │    Setup     │
├─────────────┤      ├──────────────┤
│ 1. Run      │      │ 1. Run       │
│ setup_      │      │ setup_       │
│ docker.sh   │      │ native.sh    │
└─────┬───────┘      └──────┬───────┘
      │                     │
      │                     │
      └──────────┬──────────┘
                 │
         ┌───────▼────────┐
         │  Ready to      │
         │  Train!        │
         └────────────────┘
```

### Training Workflow

```
┌──────────────────┐
│ Setup Complete   │
└────────┬─────────┘
         │
    ┌────▼─────┐
    │ Choose:  │
    │ Test or  │
    │ Full?    │
    └─┬────┬──┘
      │    │
      │    └────────────────────┐
      │                         │
┌─────▼──────────┐     ┌────────▼────────┐
│ Single GPU     │     │  Multi-GPU      │
│ Test Run       │     │  Full Training  │
├────────────────┤     ├─────────────────┤
│ • 1 GPU        │     │ • 8 GPUs        │
│ • 0.6B model   │     │ • 8B model      │
│ • 200 episodes │     │ • 2M episodes   │
│ • 30 minutes   │     │ • 2-3 days      │
└────────┬───────┘     └────────┬────────┘
         │                      │
         │                      │
         └──────────┬───────────┘
                    │
            ┌───────▼────────┐
            │  Training      │
            │  Complete      │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │  Evaluate on   │
            │  IFBench       │
            └────────────────┘
```

## 🚀 Quick Command Reference

### Initial Setup

```bash
# Native setup
bash scripts/setup_native.sh

# OR Docker setup
bash scripts/setup_docker.sh
```

### Quick Test (Recommended First)

```bash
cd /home/mohamed/rl_curriculum/open-instruct
bash ../scripts/train_if_rlvr_single_gpu.sh
```

### Full Training

```bash
# Native
cd /home/mohamed/rl_curriculum/open-instruct
bash ../scripts/train_if_rlvr.sh

# OR Docker
bash scripts/run_docker.sh
```

### Custom Training

```bash
# Using environment variables
export MODEL_NAME="allenai/OLMo-2-1124-7B-DPO"
export TOTAL_EPISODES=100000
export EXP_NAME="my_experiment"
bash scripts/train_if_rlvr.sh

# OR using config file
cp configs/training_config_template.sh configs/my_config.sh
# Edit my_config.sh with your settings
source configs/my_config.sh
bash scripts/train_if_rlvr.sh
```

## 📊 Output Structure

After training, your outputs will be organized as:

```
outputs/
└── {experiment_name}/
    ├── checkpoints/
    │   ├── checkpoint-100/
    │   ├── checkpoint-200/
    │   └── ...
    ├── logs/
    │   ├── training.log
    │   └── events.out.tfevents.*
    └── eval/
        ├── eval_results.json
        └── ...
```

## 🔍 Key Files Explained

### Training Scripts

- **`train_if_rlvr.sh`**: Main production training script with sensible defaults for 8 GPUs
- **`train_if_rlvr_single_gpu.sh`**: Lightweight version for testing with 1 GPU
- Both scripts support extensive customization via environment variables

### Configuration

- **`training_config_template.sh`**: Comprehensive config template with all options documented
- Copy and customize for different experiments
- Source before running training scripts

### Setup

- **`setup_native.sh`**: Installs all dependencies using uv, checks prerequisites
- **`setup_docker.sh`**: Builds Docker image with all dependencies
- **`run_docker.sh`**: Wrapper to run training inside Docker with proper mounts

## 🎓 Learning Path

### For First-Time Users

1. **Read**: `QUICKSTART.md`
2. **Setup**: Run `setup_native.sh` or `setup_docker.sh`
3. **Test**: Run `train_if_rlvr_single_gpu.sh`
4. **Explore**: Check outputs in `outputs/` directory
5. **Customize**: Copy and edit `training_config_template.sh`
6. **Scale**: Run full training with `train_if_rlvr.sh`

### For Advanced Users

1. **Review**: `README.md` for all options
2. **Customize**: `training_config_template.sh` for your needs
3. **Monitor**: Set up Weights & Biases tracking
4. **Evaluate**: Use IFBench for model evaluation
5. **Iterate**: Adjust hyperparameters based on results

## 💡 Tips

- **Always test with single GPU first** before running expensive multi-GPU training
- **Use environment variables** for quick experiments
- **Create config files** for reproducible experiments
- **Monitor with W&B** to track training progress
- **Save frequently** by adjusting `SAVE_FREQ`

## 🔗 External Dependencies

The project relies on two external repositories:

1. **open-instruct** (should be cloned)
   - URL: https://github.com/allenai/open-instruct
   - Contains: GRPO training code, verification functions

2. **IFBench** (should be cloned)
   - URL: https://github.com/allenai/IFBench  
   - Contains: Evaluation suite, test data

## 📝 Notes

- All paths assume base directory: `/home/mohamed/rl_curriculum/`
- Scripts are designed to work from the `open-instruct` directory
- Output paths are relative to avoid hardcoding absolute paths
- Docker scripts use mounted volumes for data persistence

---

**Questions?** Check the main [README.md](README.md) or [QUICKSTART.md](QUICKSTART.md)

