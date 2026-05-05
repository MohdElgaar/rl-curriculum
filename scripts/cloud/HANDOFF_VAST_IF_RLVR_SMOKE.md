# Handoff: Vast.ai IF-RLVR 9B smoke run (single node, 6 learners / 2 engines)

## Goal (user request)

Single-node test on **Vast.ai** for **Qwen3.5-9B**, **6 learners**, **2 vLLM engines**, **lr 5e-7**, **no reward shaping**, to validate **W&B**, **checkpointing**, **resume**, and **artifact download** after training or preemption. Budget: **~30 minutes** or **2 training steps**. Prefer **8× A100 or H100 interruptible**.

## Current live instance

| Field | Value |
| --- | --- |
| **Contract / instance id** | `36150233` |
| **Label** | `if_rlvr_9b_smoke` |
| **Status (last check)** | `running`; logs showed `uv sync` still downloading wheels (e.g. vLLM) |
| **SSH (Vast proxy)** | `ssh8.vast.ai` port **30232** (see `vastai show instance 36150233 --raw` for current mapping) |
| **Public IP** | `194.228.55.129` (direct `ssh -p <mapped-22> root@IP` requires an SSH key **attached to the Vast account**, not password auth) |
| **Hardware** | 8× **A100 SXM4** (40 GB): offer id **35646372**, **interruptible** / bid, **Czechia** |
| **Image** | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` |
| **Bid** | `--bid_price 2.35` (above `min_bid` ~2.13 for that offer) |
| **Disk** | 200 GB |

**Note:** At search time there were **no** 8× **H100** (or **A100**) hits under the default verified+rentable **interruptible** query; unfiltered **interruptible** search showed **8× A100 SXM4** with **one rentable** offer (`35646372`), which was used.

## Code pushed to GitHub

Branch: **`cursor/if-rlvr-sbatch-gpus-6-default-13c0`** on `MohdElgaar/rl-curriculum`.

Recent commits (newest first):

1. `3ae042279` — Vast onstart: **`uv python install 3.12`** (replaces deadsnakes PPA; Launchpad had **503** on one host).
2. `a8d0c8c` — **`uv sync --frozen`** with **`--prerelease=allow`** fallback (fresh `uv sync` without lock was unsatisfiable: vLLM vs torchvision bounds).
3. `8d63eee` — Submodule fallback: shallow **`mohdelgaar`** clone of **`open-instruct`** if recorded submodule SHA is missing on GitHub.
4. `f7a4310` — Git **`insteadOf`** when **`GH_TOKEN`** set (private `rl-curriculum` + submodule).
5. Earlier: **`train_if_rlvr.sh`** supports **`EVAL_ON_STEP_0`** (default `True`) and **`EXTRA_GRPO_ARGS`** for extra CLI tokens (smoke sets **`EVAL_ON_STEP_0=False`**, **`EXTRA_GRPO_ARGS='--push_to_hub False'`**).

### Important local (unpushed) drift

On the dev checkout, **`uv.lock`** and **`open-instruct`** submodule may differ from remote. The onstart script is designed to use **`uv sync --frozen`** from the **committed** lock on that branch; if resolution still fails, it falls back to **`uv sync --prerelease=allow`**.

### Submodule inconsistency

The parent repo can point at an **`open-instruct`** commit that is **not** on `origin`; submodule init then fails. Onstart **deletes `open-instruct` and clones** `https://github.com/MohdElgaar/open-instruct.git` branch **`mohdelgaar`** shallow. Long-term fix: **push aligned submodule SHA** or **update `.gitmodules` / gitlink** so `git submodule update` works without fallback.

## Onstart script (source of truth)

Path: **`scripts/cloud/vast_if_rlvr_smoke_onstart.sh`**

Behavior summary:

- Installs **`uv`**, **`uv python install 3.12`**, clones **`rl-curriculum`** at **`RL_CURRICULUM_GIT_REF`** (default branch above).
- Sets training env: **9B**, **6 / 2**, **5e-7**, shaping flags **False**, **`NUM_TRAINING_STEPS=2`**, **`SAVE_FREQ=1`**, **`CHECKPOINT_STATE_FREQ=1`**, **`LOCAL_EVAL_EVERY=1000`**, **`SMOKE_TIMEOUT=30m`** wrapper via **`timeout`**.
- Log on VM: **`/root/vast_if_rlvr_smoke.log`** (same content as **`vastai logs <id>`**).

## Secrets and how they were passed

Instance was created with **`vastai create instance ... --env`** including:

- **`GH_TOKEN`** — GitHub PAT for HTTPS clone (same class of token as `gh` OAuth; **do not log `extra_env` in public artifacts**). Consider **rotating** if instance JSON was exposed.
- **`WANDB_API_KEY`** — from `~/.netrc` **password** field for `api.wandb.ai` (not the word `user`; an earlier attempt incorrectly parsed `login` as the key).
- **`WANDB_PROJECT=rl-curriculum-vast-smoke`**

Optional for gated models: **`HF_TOKEN`** / **`HUGGING_FACE_HUB_TOKEN`** if **`Qwen/Qwen3.5-9B`** download fails.

## How to monitor (recommended order)

1. **`vastai show instance 36150233 --raw`** — `actual_status`, `status_msg`, `dph_total`, SSH ports.
2. **`vastai logs 36150233`** — full container/onstart output; confirm **`uv sync`** then **`Starting training`** / GRPO logs / **`Train finished rc=`**.
3. **W&B** — project **`mohdelgaar/rl-curriculum-vast-smoke`** (entity from API key; verify with `wandb.Api().viewer` or UI). Look for a run named like **`vast_smoke_Qwen3.5-9B_...`** under experiment naming in onstart.
4. **Artifacts** — after run, checkpoints under **`/workspace/rl_outputs/<EXP_NAME>/...`** per `train_if_rlvr.sh` (nested **`run_name`** dirs inside `open_instruct` conventions — confirm on VM with `find /workspace/rl_outputs`).

**SSH from this HPC login node** timed out to **`ssh*.vast.ai`** (possible outbound firewall). Prefer **`vastai logs`** or SSH from a network that allows Vast proxies, or attach a key and use **`vastai attach ssh`** per Vast docs.

## Re-launch command template

After destroying the old contract:

```bash
vastai create instance 35646372 \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  --disk 200 \
  --ssh --direct \
  --bid_price 2.35 \
  --label 'if_rlvr_9b_smoke' \
  --env "-e GH_TOKEN=<pat> -e WANDB_API_KEY=<key> -e WANDB_PROJECT=rl-curriculum-vast-smoke -e RL_CURRICULUM_GIT_REF=cursor/if-rlvr-sbatch-gpus-6-default-13c0" \
  --onstart /path/to/rl-curriculum/scripts/cloud/vast_if_rlvr_smoke_onstart.sh
```

Re-offer if **`35646372`** disappears: search with  
`vastai search offers 'num_gpus=8 gpu_name=A100_SXM4' -i -n -o 'score-'`  
and pick a row with **`rentable=True`**.

## What still needs verification (next agent)

- [ ] **`vastai logs`** shows **two optimizer steps** completed (or clear OOM/HF error).
- [ ] **W&B** run exists with expected **hyperparameters** and **metrics** (not crashed at init).
- [ ] **Checkpoint directories** and **`checkpoint_state`** / DeepSpeed outputs present under **`OUTPUT_DIR`**.
- [ ] **Resume:** re-run with same **`OUTPUT_DIR`** / **`checkpoint_state_dir`** and **`NUM_TRAINING_STEPS`** larger than resumed step (exercise `grpo_fast` resume path).
- [ ] **Download:** from VM (once SSH works) **`rsync`/`scp`** checkpoint tree; document preemption behavior (**interruptible** instance can stop anytime).
- [ ] If **`Qwen3.5-9B`** is gated: add **`HF_TOKEN`** to **`--env`** and redeploy.

## Incidents already handled

| Issue | Mitigation |
| --- | --- |
| Wrong CUDA image tag (`nvidia/cuda:...cudnn9...` manifest missing) | Switched to **`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`**. |
| Private repo clone | **`GH_TOKEN`** + git **`insteadOf`**. |
| Submodule SHA missing on GitHub | Fallback clone **`open-instruct`** branch **`mohdelgaar`**. |
| **`uv sync`** unsatisfiable without lock | **`uv sync --frozen`** + **`--prerelease=allow`** fallback. |
| Deadsnakes PPA **503** / no **python3.12** apt | **`uv python install 3.12`**. |
| **WANDB_API_KEY** parsed as `user` | Use **`.netrc` `password` field**, not `login`. |

## Teardown

```bash
echo y | vastai destroy instance 36150233
```

## Related repo files

- **`scripts/train_if_rlvr.sh`** — main launcher; **`EVAL_ON_STEP_0`**, **`EXTRA_GRPO_ARGS`**.
- **`configs/model_9b_lr5e7.sh`** — reference 9B lr config (not required on Vast; onstart inlines settings).

---

*Generated for continuity: instance may still be installing or training; treat **36150233** as the active smoke unless destroyed or replaced.*
