# KV Cache Concurrency & vLLM Engine Count

Reference for `kv_cache_max_concurrency` per model size (vLLM default under typical GPU memory). Use this to choose `VLLM_NUM_ENGINES` when total rollouts = 768 (e.g. 48 prompts × 16 samples).

| Model | kv_cache_max_concurrency | Rollouts (768) → engines |
|-------|--------------------------|--------------------------|
| 0.6B (Qwen3-0.6B) | 163 | 5 |
| 1.7B (Qwen3-1.7B) | 163 | 5 |
| 4B (Qwen3-4B)     | 126 | 7 |

**Formula:** `ceil(768 / kv_cache_max_concurrency)` → 768/163 ≈ 4.71 → 5; 768/126 ≈ 6.1 → 7.
