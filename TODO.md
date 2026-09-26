- Optimize more for dgx-spark
  - CPU side: the GPU loop is single-threaded and bandwidth-bound, so spare cores can't raise
      raw decode tok/s. What they *can* do is absorb the CPU-side stalls: by default one process
      on one core did HTTP + tokenization (inline, TOKENIZERS_PARALLELISM=false) + streaming +
      parsers, so a 150K+ prompt froze every stream for seconds while it tokenized. Preset now
      sets `tokenizerWorkers = 4`, `detokenizerWorkers = 2`. Verify on the box: host RSS of the
      extra workers, and that stream stalls during a long-prompt submit are gone
      (`journalctl -u sglang`, `top -H`).
  - Remaining GPU-side ideas: `--speculative-num-draft-tokens` sweep (6/8/10) under batch 4-8;
      `--page-size` vs radix hit rate; `--enable-hierarchical-cache` to spill KV to host memory
      (unified memory on GB10, so it is cheap) for the long-prompt reuse case below.
  - pi findings: <https://pi.dev/session/#912dd8b9082787be42ca8e58b00d6c19>
- Poor radix-cache hit rate on repeat long prompts (many 150K+ prefills with `#cached-token < 400`).
  - Check whether `--max-mamba-cache-size 96` / `extra_buffer` is evicting, or clients vary system prompts.
