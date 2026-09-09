#!/usr/bin/env bash
# Single-stream streaming decode benchmark: tokens/s net of time-to-first-token,
# greedy, thinking on. Same instrument as hasso5703/dgx-spark-qwen38's bench.sh
# in spirit; numbers are prompt-dependent (speculative decoding accepts
# predictable tokens), so it runs a small mixed battery and prints the median.
#
# Usage: ./bench.sh [base_url] [model]
set -euo pipefail
BASE="${1:-http://127.0.0.1:30000}"
MODEL="${2:-qwen3.8-27b}"
MAX_TOKENS="${MAX_TOKENS:-800}"

PROMPTS=(
  "Write a Python function that parses an ISO-8601 timestamp without external libraries, with type hints and docstring, and a small pytest suite."
  "Prove that the sum of the first n odd numbers is n^2, then compute the sum of the first 250 odd numbers step by step."
  "Explain how CUDA graphs reduce launch overhead and when they cannot be used. Be precise and technical."
  "Write a short story (about 300 words) about a lighthouse keeper who discovers the sea has stopped moving."
  "Refactor this bash into a robust script with set -euo pipefail, functions and argument parsing: for f in *.log; do gzip \$f; mv \$f.gz archive/; done"
)

python3 - "$BASE" "$MODEL" "$MAX_TOKENS" "${PROMPTS[@]}" <<'EOF'
import json, sys, time, urllib.request, statistics
base, model, max_tokens, *prompts = sys.argv[1:]
rates = []
for p in prompts:
    body = json.dumps({
        "model": model, "stream": True, "max_tokens": int(max_tokens),
        "temperature": 0, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": p}],
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            ch = ev.get("choices") or []
            if ch and (ch[0].get("delta") or {}) and t_first is None:
                d = ch[0]["delta"]
                if d.get("content") or d.get("reasoning_content"):
                    t_first = time.perf_counter()
    t_end = time.perf_counter()
    n = usage["completion_tokens"]
    decode_s = t_end - (t_first or t0)
    rate = n / decode_s
    rates.append(rate)
    print(f"{rate:6.1f} tok/s  ({n} tok, TTFT {((t_first or t_end)-t0):.2f}s)  {p[:60]}...")
print(f"\nmedian {statistics.median(rates):.1f} tok/s   min {min(rates):.1f}   max {max(rates):.1f}")
EOF
