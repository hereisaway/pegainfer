# Qwen3.5 snapshot host tier on RTX 5090

> **TL;DR:** Host-tier measurements for Qwen3.5-4B on one RTX 5090. Both the
> in-process scheduler first-token latency and streaming HTTP TTFT show a flat
> ~9-12 ms host hit while cold prefill scales with N. The in-process table uses
> a 1024 MiB host budget and 10 cold/10 host samples; the HTTP table uses
> 8192 MiB and 31 cold/30 host samples. Exact budgets and methods are in each
> section.

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

## In-process scheduler first-token latency

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

## HTTP TTFT (streaming, first token)

The same host tier over the real server (`target/release/pegainfer`) on the same
box. Each run starts a fresh server with `--qwen35-prefix-cache-mib 128`
(two GPU slots) and `--qwen35-snapshot-host-mib 8192` (166 host slots),
`--max-batch 1`, `--max-prefill-tokens 1024`. Prompts are token arrays of
`N + 1` ids; each length runs 31 distinct cold prompts (first discarded) and
then 30 resends of prompt 0. TTFT is the time from the POST to the first
streaming text delta.

| Prefix N | Cold median | Cold p99 | Host median | Host p99 | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 57.351 ms | 57.947 ms | 10.968 ms | 17.901 ms | 5.23x |
| 2,048 | 106.282 ms | 106.943 ms | 10.949 ms | 12.273 ms | 9.71x |
| 4,096 | 207.038 ms | 207.773 ms | 11.989 ms | 19.203 ms | 17.27x |

The server exposes the host hits through the shared counter mapping as
`vllm:external_prefix_cache_hits_total` with source `host`, because
`PrefixCacheCounters.host_hits` is forwarded through the connector stats while
`host_queries` stays zero. `vllm:prefix_cache_hits_total` (the local device
family) reads zero for these runs. A dedicated internal-host metric family
still needs a metric-name agreement.

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

## Edge coverage

`host_snapshot_history_and_unaligned_boundaries` covers a `history_tokens`
stop at 3,840 and a 4,160-token prompt whose snapshot lands on 4,096; both
resume from host with the cold-sampled token.

## Not covered yet

- More in-process samples and clock control for a meaningful p99 (HTTP already
  has 31 cold / 30 host samples).
- KV-eviction miss and the `prompt % block_size == 0 && max_tokens == 1`
  admission boundary in the host path.
- TP1/TP2 performance rows and budget tables (the 5090 pair has no NVLink/P2P,
  so TP numbers are not representative).
