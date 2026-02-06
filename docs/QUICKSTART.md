# Quick Start Guide - IF-RLVR Training

## TL;DR - Get Training in 3 Steps

### Using Docker (Recommended)

```bash
# 1. Build Docker image
cd /home/mohamed/rl_curriculum
bash scripts/setup_docker.sh

# 2. Set your API keys (optional but recommended)
export WANDB_API_KEY="your_wandb_key"
export HF_TOKEN="your_huggingface_token"

# 3. Start training!
bash scripts/run_docker.sh
```

### Using Native Python (uv)

```bash
# 1. Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
cd /home/mohamed/rl_curriculum/open-instruct
uv sync

# 3. Start training!
bash ../scripts/train_if_rlvr.sh
```

## Single GPU Quick Test

```bash
cd /home/mohamed/rl_curriculum/open-instruct
bash ../scripts/train_if_rlvr_single_gpu.sh
```

This runs a lightweight test with:
- Small 0.6B model (Qwen3-0.6B)
- Only 200 episodes
- 1 GPU
- ~10-30 minutes

## What Each Script Does

| Script | Purpose | Time | GPUs |
|--------|---------|------|------|
| `train_if_rlvr_single_gpu.sh` | Quick test/debug | 10-30 min | 1 |
| `train_if_rlvr.sh` | Full IF-RLVR training | Hours/Days | 8 |
| `setup_docker.sh` | Build Docker image | 10-20 min | N/A |
| `run_docker.sh` | Run training in Docker | Variable | 1-8 |

## Common Customizations

### Train Different Model

```bash
export MODEL_NAME="allenai/OLMo-2-1124-7B-DPO"
bash scripts/train_if_rlvr.sh
```

### Shorter Training Run

```bash
export TOTAL_EPISODES=100000
export EXP_NAME="quick_test"
bash scripts/train_if_rlvr.sh
```

### Use Fewer GPUs

```bash
export NUM_GPUS=4
export NUM_LEARNERS_PER_NODE=3
bash scripts/train_if_rlvr.sh
```

## Monitoring

Training logs are automatically saved to:
```
outputs/${EXP_NAME}/
```

If you set `WANDB_API_KEY`, you can monitor training at:
```
https://wandb.ai/your-username/your-project
```

## Output Files

After training completes, you'll find:
- **Model checkpoints**: `outputs/${EXP_NAME}/checkpoints/`
- **Training logs**: `outputs/${EXP_NAME}/logs/`
- **Evaluation results**: `outputs/${EXP_NAME}/eval/`

## Next Steps

1. **Evaluate on IFBench**: See [IFBench/README.md](IFBench/README.md)
2. **Full documentation**: See [README.md](README.md)
3. **Troubleshooting**: Check the main README

## Getting Help

- Check `nvidia-smi` if you see GPU errors
- Look at logs in `outputs/${EXP_NAME}/`
- Try the single GPU script first if full training fails
- See [README.md](README.md) for detailed troubleshooting

---

**Ready to train?** Just run:
```bash
cd /home/mohamed/rl_curriculum/open-instruct
bash ../scripts/train_if_rlvr_single_gpu.sh
```

