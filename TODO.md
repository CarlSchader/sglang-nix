- Optimize more for dgx-spark
    - One thing to consider when optimizing is how little of our 20 cpu cores we're actually using. Not sure if those can help with token throughput.
- voice model
- Zombie requests after client disconnect (sglang 0.5.19 regression)
    - Symptom: thousands of `Received output for rid=... but the state was deleted in TokenizerManager`
      in the journal; the scheduler keeps decoding a request whose client left until `max_tokens`
      (seen up to 406s), holding a running slot + KV/mamba state and halving others' tok/s.
    - Cause: TokenizerManager drops `rid_to_state` on disconnect, then the deferred abort
      (`tokenizer_manager.py` `abort_request`) bails because the rid is gone, so `AbortReq`
      never reaches the scheduler.
    - Upstream: https://github.com/sgl-project/sglang/issues/36333 ,
      https://github.com/sgl-project/sglang/issues/36876 , fix PR #36418.
    - Action: bump the pinned sglang once the fix ships, or carry the patch in the uv2nix overlay.
- Poor radix-cache hit rate on repeat long prompts (many 150K+ prefills with `#cached-token < 400`).
    - Check whether `--max-mamba-cache-size 96` / `extra_buffer` is evicting, or clients vary system prompts.
