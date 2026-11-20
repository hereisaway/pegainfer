# Qwen3.5 snapshot host tier on RTX 5090

> **TL;DR:** In-process scheduler first-token latency for Qwen3.5-4B on one
> RTX 5090, with a 128 MiB GPU snapshot budget (two slots) and a 1024 MiB
> pinned-host budget (twenty slots). Ten distinct cold prompts and ten host
> hits per prefix length, sequential, `max_batch=1`, `max_prefill_tokens=N`.
> Host hits are about 9 ms and flat; cold prefill scales with N.

**Last touched:** 2026-11

## Environment

- GPU: RTX 5090 32 GB, driver `580.178.04`
- CUDA: 13.0 (`/usr/local/cuda-13.0`)
- Model: `Qwen3.5-4B`, BF16, ModelScope snapshot; HF fixture revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Build: Rust `nightly-2026-07-10`; Triton 3.7.1 at build time
- Knobs: `--qwen35-prefix-cache-mib 128` (two GPU slots), the host tier set to
  1024 MiB (twenty slots), `max_batch=1`, `max_prefill_tokens=N`,
  CUDA Graph on.

## Measurement

Each length uses eleven distinct prompts of `N + 1` tokens with
`max_prefill_tokens=N`: the first prefill ends at `N` and publishes a snapshot,
then one token remains. The first prompt is discarded as warm-up; the next ten
are the cold samples. Eleven prefixes leave only the newest two on GPU, so
prompt 0 is host-resident; it is then resent ten times for the host samples.
The sampled token of every host hit must equal its cold run.

This is the scheduler's in-process first-token latency, not HTTP TTFT.

| Prefix N | Cold median | Cold p99 | Host median | Host p99 | Host hits (tokens) |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 54.995 ms | 55.102 ms | 9.021 ms | 9.108 ms | 10,240 |
| 2,048 | 97.244 ms | 97.362 ms | 9.137 ms | 9.180 ms | 20,480 |
| 4,096 | 189.247 ms | 190.002 ms | 9.660 ms | 9.827 ms | 40,960 |

Host hits are flat because a restore is a fixed 49.125 MiB H2D plus one
suffix token; cold prefill grows with N. The p99 over ten samples is the
maximum, so the tail is not statistically meaningful yet.

## Reproduce

```bash
cd pegainfer
source ../env.sh
PEGAINFER_TEST_MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  cargo test --release -p pegainfer-qwen35 --features qwen35 --lib \
  host_snapshot_step4_measurements -- --ignored --nocapture --test-threads=1
```

## TP correctness (not performance)

`tp_host_snapshot_spill_and_hit` passes on two RTX 5090s with NCCL 2.30.7
(`NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`): three distinct 1,025-token prefixes
spill the oldest to host, and resending it restores from host with the same
sampled token and non-zero `host_hits`.

## Not covered yet

- HTTP TTFT (the server binary needs the local bindgen/libclang environment
  fixed; this harness calls the scheduler directly).
- More samples and clock control for meaningful p99.
- TP1/TP2 performance rows and budget tables.
