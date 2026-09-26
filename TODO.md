- Optimize more for dgx-spark
    - One thing to consider when optimizing is how little of our 20 cpu cores we're actually using. Not sure if those can help with token throughput.

- Poor radix-cache hit rate on repeat long prompts (many 150K+ prefills with `#cached-token < 400`).
    - Check whether `--max-mamba-cache-size 96` / `extra_buffer` is evicting, or clients vary system prompts.
