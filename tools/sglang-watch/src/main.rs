// sglang-watch — terminal dashboard for a running SGLang server.
//
// Data sources (all read-only GETs against the API port):
//   /v1/loads   always-on scheduler load snapshot (SHM-published, no flags
//               needed): running/queued requests, KV pool, throughput,
//               prefill-busy microseconds, decode step-time moments,
//               spec-decode accept stats, VRAM breakdown, queue detail.
//   /metrics    Prometheus, only if the server was started with
//               --enable-metrics. Adds TTFT / end-to-end latency
//               percentiles.
//   ss(8)       established TCP connections on the API port, to count
//               distinct client hosts ("agents").
//
// Usage:
//   sglang-watch                     # live TUI (1s refresh)
//   sglang-watch --once              # one plain-text snapshot
//   sglang-watch -i 2 --url http://host:30000
//   sglang-watch --window 120        # rates over last 120s
//
// Keys: Ctrl-C or q quits.  Exit codes: 0 ok, 2 server unreachable (--once).
// Pure Rust std (+ serde_json + libc) — no dependencies beyond that, safe to
// run on the serving host while it handles live traffic.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::Value;

const WIDTH: usize = 78;
const SPARK: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

// ---------------------------------------------------------------------------
// colors
// ---------------------------------------------------------------------------

static USE_COLOR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn color_on() {
    let no_color = std::env::var_os("NO_COLOR").is_some();
    USE_COLOR.store(std::io::IsTerminal::is_terminal(&std::io::stdout()) && !no_color,
                    std::sync::atomic::Ordering::Relaxed);
}

fn on() -> bool {
    USE_COLOR.load(std::sync::atomic::Ordering::Relaxed)
}

fn paint(s: &str, code: &str) -> String {
    if !on() {
        return s.to_string();
    }
    format!("\x1b[{code}m{s}\x1b[0m")
}
fn bold(s: &str) -> String { paint(s, "1") }
fn dim(s: &str) -> String { paint(s, "2") }
fn red(s: &str) -> String { paint(s, "1;31") }
fn green(s: &str) -> String { paint(s, "32") }
fn yellow(s: &str) -> String { paint(s, "1;33") }
fn cyan(s: &str) -> String { paint(s, "36") }
fn magenta(s: &str) -> String { paint(s, "35") }

/// Visible length of a string with ANSI escapes removed (for padding).
/// Counts Unicode characters, not bytes, so multi-byte glyphs (▂█─) count
/// as one column each.
fn vis_len(s: &str) -> usize {
    let mut n = 0;
    let mut in_escape = false;
    for c in s.chars() {
        if in_escape {
            if c.is_ascii_alphabetic() {
                in_escape = false;
            }
        } else if c == '\u{1b}' {
            in_escape = true;
        } else {
            n += 1;
        }
    }
    n
}

fn ljust(s: &str, width: usize) -> String {
    let pad = width.saturating_sub(vis_len(s));
    format!("{s}{}", " ".repeat(pad))
}

// ---------------------------------------------------------------------------
// number formatting
// ---------------------------------------------------------------------------

fn human(n: f64) -> String {
    let a = n.abs();
    for (v, suf) in [(1e9, "G"), (1e6, "M"), (1e3, "K")] {
        if a >= v {
            let x = n / v;
            return if x >= 100.0 {
                format!("{x:.0}{suf}")
            } else {
                format!("{x:.1}{suf}")
            };
        }
    }
    format!("{}", n.round() as i64)
}

fn sec(v: Option<f64>) -> String {
    let Some(v) = v.filter(|v| !v.is_nan()) else {
        return "—".into();
    };
    if v < 1.0 {
        format!("{:.0}ms", v * 1000.0)
    } else if v < 60.0 {
        format!("{v:.1}s")
    } else {
        let s = v as i64;
        format!("{}m{:02}s", s / 60, s % 60)
    }
}

fn gb(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.1}G")).unwrap_or_else(|| "—".into())
}

fn pct(v: Option<f64>, digits: u32) -> String {
    match v {
        Some(x) => format!("{:.prec$}%", 100.0 * x, prec = digits as usize),
        None => "—".into(),
    }
}

fn bar(frac: Option<f64>, width: usize) -> String {
    let Some(frac) = frac else {
        return "—".repeat(width);
    };
    let n = (frac.clamp(0.0, 1.0) * width as f64).round() as usize;
    "█".repeat(n) + &"░".repeat(width - n)
}

fn spark(values: &[f64], width: usize) -> String {
    if values.is_empty() {
        return dim(&"▁".repeat(width));
    }
    let tail: Vec<f64> = values.iter().rev().take(width).cloned().collect();
    let peak = tail.iter().fold(0.0f64, |m, v| m.max(*v));
    if peak <= 0.0 {
        return dim(&"▁".repeat(tail.len()));
    }
    let mut out = String::new();
    for v in &tail {
        let scaled = (v.clamp(0.0, f64::MAX) / peak) * (SPARK.len() - 1) as f64 + 0.5;
        let idx = scaled.round() as usize;
        out.push(SPARK[idx.min(SPARK.len() - 1)]);
    }
    cyan(&out)
}

// ---------------------------------------------------------------------------
// HTTP (std only, blocking with timeouts)
// ---------------------------------------------------------------------------

struct Endpoint {
    host: String,
    port: u16,
}

fn parse_url(url: &str) -> Endpoint {
    let u = if url.contains("://") {
        url.to_string()
    } else {
        format!("http://{url}")
    };
    let rest = u.split_once("://").map_or(u.as_str(), |(_, r)| r);
    let hostport = rest.split_once('/').map_or(rest, |(hp, _)| hp);
    let (host, port) = match hostport.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse::<u16>().unwrap_or(30000)),
        None => (hostport.to_string(), 30000),
    };
    Endpoint { host, port }
}

fn http_get(ep: &Endpoint, path: &str) -> Result<(u16, String), String> {
    let addr = format!("{}:{}", ep.host, ep.port);
    let mut addrs = addr
        .to_socket_addrs()
        .map_err(|e| format!("resolve {addr}: {e}"))?;
    let addr = addrs.next().ok_or_else(|| "no addresses".to_string())?;

    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(3))
        .map_err(|e| format!("connect {addr}: {e}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .map_err(|e| e.to_string())?;

    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {}\r\nUser-Agent: sglang-watch/0.1\r\nAccept: */*\r\nConnection: close\r\n\r\n",
        ep.host
    );
    stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;

    use std::io::Read;
    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
    stream
        .read_to_end(&mut buf)
        .map_err(|e| format!("read: {e}"))?;
    let text = String::from_utf8_lossy(&buf).into_owned();
    let (head, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| "malformed response".to_string())?;
    let status: u16 = head
        .lines()
        .next()
        .and_then(|l| l.split(' ').nth(1))
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "bad status line".to_string())?;
    Ok((status, body.to_string()))
}

fn get_json(ep: &Endpoint, path: &str) -> Result<(u16, Value), String> {
    let (status, body) = http_get(ep, path)?;
    let v: Value = serde_json::from_str(&body).map_err(|e| format!("json: {e}"))?;
    Ok((status, v))
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// /v1/loads
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default, PartialEq)]
struct Counters {
    prefill_uncached_tokens: f64,
    prefill_busy_us: f64,
    decode_m0: f64, // decode steps
    decode_m1: f64, // sum batch size
    decode_m2: f64, // sum step_us
    decode_m4: f64, // sum batch*step
    decode_m5: f64, // sum generated tokens
}

impl Counters {
    /// True if any counter moved backwards vs `p` (server restarted).
    fn decreased_from(&self, p: &Counters) -> bool {
        self.prefill_uncached_tokens < p.prefill_uncached_tokens
            || self.prefill_busy_us < p.prefill_busy_us
            || self.decode_m0 < p.decode_m0
            || self.decode_m1 < p.decode_m1
            || self.decode_m2 < p.decode_m2
            || self.decode_m4 < p.decode_m4
            || self.decode_m5 < p.decode_m5
    }
}

#[derive(Default)]
struct Load {
    ok: bool,
    fetched_at: f64,
    version: String,
    running: u32,
    waiting: u32,
    waiting_uncached_tokens: u64,
    used_tokens: u64,
    total_tokens: u64,
    max_total_tokens: u64,
    max_running: u32,
    token_usage: f64,
    cache_hit_rate: f64,
    retracted: u32,
    grammar_q: u32,
    paused: u32,
    counters: Counters,
    spec_accept_length: Option<f64>,
    spec_accept_rate: Option<f64>,
    mem_weight_gb: Option<f64>,
    mem_kv_gb: Option<f64>,
    mem_graph_gb: Option<f64>,
    error: String,
}

fn num(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

fn fetch_loads(ep: &Endpoint) -> Load {
    let mut l = Load::default();
    let (status, doc) = match get_json(ep, "/v1/loads") {
        Ok((s, d)) => (s, d),
        Err(e) => {
            l.error = e;
            return l;
        }
    };
    if status != 200 {
        l.error = format!("HTTP {status}");
        return l;
    }
    let loads = match doc.get("loads").and_then(Value::as_array) {
        Some(a) if !a.is_empty() => a,
        _ => {
            l.error = "empty loads".into();
            return l;
        }
    };

    l.ok = true;
    l.fetched_at = now();
    l.version = doc
        .get("version")
        .and_then(Value::as_str)
        .unwrap_or("")
        .into();

    for r in loads {
        l.running += num(r, "num_running_reqs") as u32;
        l.waiting += num(r, "num_waiting_reqs") as u32;
        l.waiting_uncached_tokens += num(r, "num_waiting_uncached_tokens") as u64;
        l.used_tokens += num(r, "num_used_tokens") as u64;
        l.total_tokens += num(r, "num_total_tokens") as u64;
        l.max_total_tokens = l.max_total_tokens.max(num(r, "max_total_num_tokens") as u64);
        l.max_running = l.max_running.max(num(r, "max_running_requests") as u32);
        l.token_usage = l.token_usage.max(num(r, "token_usage"));
        l.cache_hit_rate = l.cache_hit_rate.max(num(r, "cache_hit_rate"));
        l.counters.prefill_uncached_tokens += num(r, "total_prefill_uncached_tokens");
        l.counters.prefill_busy_us += num(r, "total_prefill_busy_us");
        if let Some(mom) = r.get("decode_moments").and_then(Value::as_array) {
            for (i, slot) in [
                (0, &mut l.counters.decode_m0),
                (1, &mut l.counters.decode_m1),
                (2, &mut l.counters.decode_m2),
                (4, &mut l.counters.decode_m4),
                (5, &mut l.counters.decode_m5),
            ] {
                if let Some(x) = mom.get(i).and_then(Value::as_f64) {
                    *slot += x;
                }
            }
        }
        if let Some(q) = r.get("queues") {
            l.retracted += num(q, "retracted") as u32;
            l.grammar_q += num(q, "grammar") as u32;
            l.paused += num(q, "paused") as u32;
        }
        if let Some(spec) = r.get("speculative") {
            if let Some(al) = spec.get("accept_length").and_then(Value::as_f64) {
                if l.spec_accept_length.map_or(true, |cur| al > cur) {
                    l.spec_accept_length = Some(al);
                }
            }
            if let Some(ar) = spec.get("accept_rate").and_then(Value::as_f64) {
                if l.spec_accept_rate.map_or(true, |cur| ar > cur) {
                    l.spec_accept_rate = Some(ar);
                }
            }
        }
        if let Some(mem) = r.get("memory") {
            if l.mem_weight_gb.is_none() {
                l.mem_weight_gb = mem.get("weight_gb").and_then(Value::as_f64);
                l.mem_kv_gb = mem.get("kv_cache_gb").and_then(Value::as_f64);
                l.mem_graph_gb = mem.get("graph_gb").and_then(Value::as_f64);
            }
        }
    }
    l
}

// ---------------------------------------------------------------------------
// sliding-window rates over cumulative counters
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct Rates {
    gen_tok_s: f64,
    prefill_busy_frac: f64,
    prefill_uncached_rate: f64,
    step_ms: Option<f64>,
    bsz: Option<f64>,
}

struct Window {
    seconds: f64,
    samples: std::collections::VecDeque<(f64, Counters)>,
    prev: Option<Counters>,
    rates: Option<Rates>,
    history: std::collections::VecDeque<f64>,
}

impl Window {
    fn new(seconds: f64) -> Self {
        Self {
            seconds,
            samples: std::collections::VecDeque::new(),
            prev: None,
            rates: None,
            history: std::collections::VecDeque::with_capacity(600),
        }
    }

    /// Rate over the trailing `seconds` (growing from the first sample up to
    /// the full window). Resets cleanly when the server restarts its
    /// counters.
    fn sample(&mut self, t: f64, c: Counters) {
        if self.prev.map_or(false, |p| c.decreased_from(&p)) {
            self.reset();
        }
        self.samples.push_back((t, c));
        while self.samples.front().is_some_and(|(t0, _)| t - t0 > self.seconds) {
            self.samples.pop_front();
        }
        if self.samples.len() >= 2 {
            let (t0, c0) = *self.samples.front().unwrap();
            let (t1, c1) = *self.samples.back().unwrap();
            let dt = t1 - t0;
            if dt >= 1.0 {
                let dgen = c1.decode_m5 - c0.decode_m5;
                let dbusy_us = c1.prefill_busy_us - c0.prefill_busy_us;
                let dunc = c1.prefill_uncached_tokens - c0.prefill_uncached_tokens;
                let dsteps = c1.decode_m0 - c0.decode_m0;
                let rates = Rates {
                    gen_tok_s: dgen / dt,
                    prefill_busy_frac: (dbusy_us / 1e6 / dt).clamp(0.0, 1.0),
                    prefill_uncached_rate: dunc / dt,
                    step_ms: (dsteps > 0.0).then(|| (c1.decode_m2 - c0.decode_m2) / dsteps / 1000.0),
                    bsz: (dsteps > 0.0).then(|| (c1.decode_m1 - c0.decode_m1) / dsteps),
                };
                self.history.push_back(rates.gen_tok_s);
                if self.history.len() > 600 {
                    self.history.pop_front();
                }
                self.rates = Some(rates);
            }
        }
        self.prev = Some(c);
    }

    fn reset(&mut self) {
        self.samples.clear();
        self.prev = None;
        self.rates = None;
        self.history.clear();
    }
}

// ---------------------------------------------------------------------------
// clients via ss
// ---------------------------------------------------------------------------

fn probe_clients(port: u16) -> Option<(usize, usize)> {
    let out = std::process::Command::new("ss")
        .args(["-Ht", "state", "established", &format!("( sport = :{port} )")])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut hosts: BTreeSet<String> = BTreeSet::new();
    let mut conns = 0usize;
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        // columns vary with filters (State is dropped when filtered), but the
        // last column is always Peer:Port
        if parts.len() < 4 {
            continue;
        }
        let peer = parts[parts.len() - 1];
        let before_port = peer.rsplit_once(':').map_or(peer, |(h, _)| h);
        let host = before_port.trim_matches(|c| c == '[' || c == ']');
        if host.is_empty() {
            continue;
        }
        hosts.insert(host.to_string());
        conns += 1;
    }
    (conns > 0).then_some((hosts.len(), conns))
}

// ---------------------------------------------------------------------------
// /metrics (optional: only when server runs with --enable-metrics)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Prom {
    // (name, le in ns; i64::MAX = +Inf) -> cumulative count
    buckets: BTreeMap<(String, i64), f64>,
    sum: BTreeMap<String, f64>,
    count: BTreeMap<String, f64>,
    gauges: BTreeMap<String, f64>,
}

fn parse_prom(text: &str) -> Prom {
    let mut p = Prom::default();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((lhs, val_s)) = line.split_once(' ') else {
            continue;
        };
        let (name, labels) = match lhs.find('{') {
            Some(i) => (&lhs[..i], &lhs[i + 1..lhs.len() - 1]),
            None => (lhs, ""),
        };
        let val: f64 = match val_s {
            "+Inf" | "Inf" => f64::INFINITY,
            "-Inf" => f64::NEG_INFINITY,
            "NaN" => f64::NAN,
            _ => match val_s.parse() {
                Ok(v) => v,
                Err(_) => continue,
            },
        };
        if let Some(base) = name.strip_suffix("_bucket") {
            let le_ns: i64 = labels
                .split(';')
                .find_map(|kv| {
                    let (k, v) = kv.split_once('=')?;
                    (k.trim() == "le").then(|| v.trim().trim_matches('"'))
                })
                .map(|s| {
                    if s == "Inf" || s == "+Inf" {
                        i64::MAX
                    } else {
                        s.parse::<f64>()
                            .map(|f| (f * 1e9).round() as i64)
                            .unwrap_or(i64::MAX)
                    }
                })
                .unwrap_or(i64::MAX);
            *p.buckets.entry((base.to_string(), le_ns)).or_insert(0.0) += val;
        } else if let Some(base) = name.strip_suffix("_sum") {
            *p.sum.entry(base.to_string()).or_insert(0.0) += val;
        } else if let Some(base) = name.strip_suffix("_count") {
            *p.count.entry(base.to_string()).or_insert(0.0) += val;
        } else {
            *p.gauges.entry(name.to_string()).or_insert(0.0) += val;
        }
    }
    p
}

fn hist_quantile(p: &Prom, name: &str, q: f64) -> Option<f64> {
    let total = *p.count.get(name)?;
    if total <= 0.0 {
        return None;
    }
    let mut buckets: Vec<(f64, f64)> = p
        .buckets
        .iter()
        .filter(|((n, le_ns), _)| n == name && *le_ns != i64::MAX)
        .map(|((_, le_ns), c)| (*le_ns as f64 / 1e9, *c))
        .collect();
    buckets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let target = q * total;
    let mut prev_le = 0.0f64;
    let mut prev_c = 0.0f64;
    for (le, c) in &buckets {
        if c >= &target {
            let denom = c - prev_c;
            let frac = if denom > 0.0 {
                (target - prev_c) / denom
            } else {
                0.0
            };
            return Some(prev_le + (le - prev_le) * frac);
        }
        prev_le = *le;
        prev_c = *c;
    }
    Some(prev_le)
}

fn fetch_metrics(ep: &Endpoint) -> Option<Prom> {
    let (status, body) = http_get(ep, "/metrics").ok()?;
    (status == 200).then(|| parse_prom(&body))
}

// ---------------------------------------------------------------------------
// chips / status
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum C {
    Red,
    Green,
    Yellow,
    Dim,
}

fn chip(label: &str, c: C) -> String {
    match c {
        C::Red => red(&format!("[{label}]")),
        C::Green => green(&format!("[{label}]")),
        C::Yellow => yellow(&format!("[{label}]")),
        C::Dim => dim(&format!("[{label}]")),
    }
}

fn compute_chips(l: &Load, rates: Option<&Rates>, offline: bool) -> String {
    if offline {
        return chip("OFFLINE", C::Red);
    }
    let active = l.running > 0 || l.waiting > 0;
    let mut chips = if !active {
        vec![chip("IDLE", C::Dim)]
    } else if l.max_running > 0 && l.running >= l.max_running && l.waiting > 0 {
        vec![chip("SATURATED", C::Red)]
    } else if rates.map_or(false, |r| r.prefill_busy_frac >= 0.6) {
        vec![chip("PREFILL-BUSY", C::Yellow)]
    } else {
        vec![chip("SERVING", C::Green)]
    };
    if l.token_usage >= 0.9 {
        chips.push(chip("KV-PRESSURE", C::Yellow));
    }
    if l.retracted > 0 {
        chips.push(chip(&format!("RETRACTING {}", l.retracted), C::Red));
    }
    if rates.map_or(false, |r| {
        active && r.prefill_uncached_rate > 50.0 && l.cache_hit_rate < 0.25
    }) {
        chips.push(chip("CACHE-MISS", C::Yellow));
    }
    chips.join("  ")
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

fn label(s: &str) -> String {
    format!("{s:<13}")
}

fn clock(now_s: f64) -> String {
    // local time via libc (no time-crate dependency)
    unsafe {
        let mut t: libc::tm = std::mem::zeroed();
        let mut ts = (now_s as i64).abs();
        if libc::localtime_r(&mut ts as *const i64 as *const libc::time_t, &mut t).is_null() {
            return "??:??:??".into();
        }
        format!("{:02}:{:02}:{:02}", t.tm_hour, t.tm_min, t.tm_sec)
    }
}

fn render(
    model: &str,
    url: &str,
    l: &Load,
    rates: Option<&Rates>,
    window: &Window,
    clients: Option<(usize, usize)>,
    metrics: Option<&Prom>,
    now_s: f64,
    interval: f64,
    once: bool,
) -> String {
    let offline = !l.ok;
    let stale = !offline && now_s - l.fetched_at > 3.0 * interval;
    let mut out = String::new();

    let clock = clock(now_s);
    let head = format!("SGLang watch  {}  {}", magenta(model), dim(url));
    out.push_str(&ljust(&head, WIDTH.saturating_sub(vis_len(&clock))));
    out.push_str(&dim(&clock));
    out.push('\n');

    let clients_txt = match clients {
        Some((h, c)) if h > 0 => format!("clients {h} ({c} conns)"),
        _ => "clients —".to_string(),
    };
    let sub = format!(
        "{}   running {}/{}   queued {}   {}",
        compute_chips(l, rates, offline),
        l.running,
        l.max_running,
        l.waiting,
        dim(&clients_txt)
    );
    out.push_str(&ljust(&sub, WIDTH));
    if stale {
        out.push_str(&format!(
            "   {}",
            red(&format!("STALE {}s", (now_s - l.fetched_at) as i64))
        ));
    }
    out.push('\n');
    out.push_str(&dim(&"─".repeat(WIDTH)));
    out.push('\n');

    // throughput -------------------------------------------------------------
    let mut thr = label("Throughput");
    match rates.map(|r| r.gen_tok_s) {
        Some(t) => thr.push_str(&bold(&format!("{t:.1}"))),
        None => thr.push_str(&dim("—")),
    }
    thr.push_str(" tok/s");
    if !once {
        let sp = spark(&window.history.iter().copied().collect::<Vec<_>>(), 24);
        let pad = WIDTH.saturating_sub(vis_len(&sp));
        let mut padded = thr;
        while vis_len(&padded) < pad {
            padded.push(' ');
        }
        thr = padded + &sp;
    }
    out.push_str(&ljust(&thr, WIDTH));
    out.push('\n');

    // prefill ----------------------------------------------------------------
    let busy = rates.map(|r| r.prefill_busy_frac);
    let pre_rate = rates.map(|r| r.prefill_uncached_rate);
    let busy_disp = busy.map(|b| pct(Some(b), 1)).unwrap_or_else(|| "—".into());
    let mut pre = label("Prefill");
    pre.push_str("busy ");
    let busy_colored = if busy.map_or(false, |b| b >= 0.5) {
        yellow(&busy_disp)
    } else {
        busy_disp.clone()
    };
    pre.push_str(&busy_colored);
    pre.push_str("  ");
    pre.push_str(&cyan(&bar(busy, 12)));
    pre.push_str(&format!("  pend {} tok", human(l.waiting_uncached_tokens as f64)));
    if let Some(pr) = pre_rate {
        if pr > 0.0 {
            pre.push_str(&format!("  {}", dim(&format!("{} tok/s uncached", human(pr)))));
        }
    }
    out.push_str(&ljust(&pre, WIDTH));
    out.push('\n');

    // requests ---------------------------------------------------------------
    let run_col = if l.max_running > 0 && l.running >= l.max_running && l.waiting > 0 {
        red(&format!("{}/{}", l.running, l.max_running))
    } else {
        bold(&format!("{}/{}", l.running, l.max_running))
    };
    let mut req = label("Requests");
    req.push_str(&format!("running {run_col}   queued {}", l.waiting));
    if l.grammar_q + l.paused > 0 {
        req.push_str(&format!("   grammar {} paused {}", l.grammar_q, l.paused));
    }
    if l.retracted > 0 {
        req.push_str(&format!("   {}", red(&format!("retracted {}", l.retracted))));
    }
    out.push_str(&ljust(&req, WIDTH));
    out.push('\n');

    // kv cache ---------------------------------------------------------------
    let usage = pct(Some(l.token_usage), 0);
    let usage_col = if l.token_usage >= 0.9 {
        red(&usage)
    } else if l.token_usage >= 0.7 {
        yellow(&usage)
    } else {
        usage
    };
    let mut kv = label("KV cache");
    kv.push_str(&usage_col);
    kv.push_str("  ");
    kv.push_str(&cyan(&bar(Some(l.token_usage), 12)));
    kv.push_str(&format!(
        "  {}/{} tok   hit {:.0}%",
        human(l.used_tokens as f64),
        human(l.max_total_tokens as f64),
        l.cache_hit_rate * 100.0
    ));
    out.push_str(&ljust(&kv, WIDTH));
    out.push('\n');

    // spec decode ------------------------------------------------------------
    match l.spec_accept_length {
        Some(al) => {
            let mut spec = label("Spec decode");
            spec.push_str(&format!("accept len {}", bold(&format!("{al:.2}"))));
            spec.push_str(
                &if l.spec_accept_rate.map_or(false, |ar| ar > 0.0) {
                    format!("   rate {:.0}%", l.spec_accept_rate.unwrap() * 100.0)
                } else {
                    dim("   rate —(idle)")
                },
            );
            out.push_str(&ljust(&spec, WIDTH));
        }
        None => out.push_str(&ljust(
            &format!("{} {}", label("Spec decode"), dim("(none reported)")),
            WIDTH,
        )),
    }
    out.push('\n');

    // latency ----------------------------------------------------------------
    let mut lat = label("Latency");
    if let Some(r) = rates {
        if let Some(ms) = r.step_ms {
            lat.push_str(&format!(
                "step {}  {}",
                bold(&format!("{ms:.0}ms")),
                dim(&format!("(bsz {:.1})", r.bsz.unwrap_or(0.0)))
            ));
        } else {
            lat.push_str(&dim("step —(no decode in window)"));
        }
    } else {
        lat.push_str(&dim("step —(no decode in window)"));
    }
    out.push_str(&ljust(&lat, WIDTH));
    out.push('\n');
    if let Some(m) = metrics {
        let p50 = hist_quantile(m, "sglang:time_to_first_token_seconds", 0.5);
        let p95 = hist_quantile(m, "sglang:time_to_first_token_seconds", 0.95);
        out.push_str(&ljust(
            &format!("{}TTFT p50 {} p95 {}", label(""), sec(p50), sec(p95)),
            WIDTH,
        ));
        out.push('\n');
        let e50 = hist_quantile(m, "sglang:e2e_request_latency_seconds", 0.5);
        let e95 = hist_quantile(m, "sglang:e2e_request_latency_seconds", 0.95);
        out.push_str(&ljust(
            &format!("{}E2E  p50 {} p95 {}", label(""), sec(e50), sec(e95)),
            WIDTH,
        ));
        out.push('\n');
    }

    // memory -----------------------------------------------------------------
    if l.mem_weight_gb.is_some() {
        let mem = format!(
            "{}weights {}  kv {}  graphs {}",
            label("VRAM"),
            gb(l.mem_weight_gb),
            gb(l.mem_kv_gb),
            gb(l.mem_graph_gb)
        );
        out.push_str(&ljust(&mem, WIDTH));
        out.push('\n');
    }

    out.push_str(&dim(&"─".repeat(WIDTH)));
    out.push('\n');

    // footer -----------------------------------------------------------------
    let foot = format!(
        "refresh {:.1}s  window {:.0}s  {}",
        interval, window.seconds, l.version
    );
    let mut hints: Vec<String> = Vec::new();
    if !offline && metrics.is_none() {
        hints.push(dim(
            "/metrics off: add --enable-metrics to the server for TTFT/E2E percentiles",
        ));
    }
    if stale {
        hints.push(red("last /v1/loads fetch failed — showing cached values"));
    }
    if offline {
        let err: String = l.error.chars().take(WIDTH - 40).collect();
        hints.push(red(&format!("unreachable: {err}")));
    }
    out.push_str(&ljust(&foot, WIDTH));
    if !hints.is_empty() {
        out.push_str("  ");
        out.push_str(&hints.join("  "));
    }
    out.push('\n');
    if !once {
        out.push_str(&dim("q / Ctrl-C quit"));
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// termios cbreak + key polling
// ---------------------------------------------------------------------------

struct OldTermios(libc::termios);

impl Drop for OldTermios {
    fn drop(&mut self) {
        unsafe {
            libc::tcsetattr(0, libc::TCSANOW, &self.0);
        }
    }
}

fn cbreak() -> Option<OldTermios> {
    unsafe {
        let mut t: libc::termios = std::mem::zeroed();
        if libc::tcgetattr(0, &mut t) != 0 {
            return None;
        }
        let saved = t;
        t.c_lflag &= !(libc::ICANON | libc::ECHO);
        if libc::tcsetattr(0, libc::TCSANOW, &t) != 0 {
            return None;
        }
        Some(OldTermios(saved))
    }
}

fn read_key(timeout_ms: i32) -> Option<u8> {
    unsafe {
        let mut pfd = libc::pollfd {
            fd: 0,
            events: libc::POLLIN,
            revents: 0,
        };
        let r = libc::poll(&mut pfd, 1, timeout_ms);
        if r <= 0 {
            return None;
        }
        let mut b: u8 = 0;
        let n = libc::read(0, &mut b as *mut u8 as *mut libc::c_void, 1);
        (n == 1).then_some(b)
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut url = "http://127.0.0.1:30000".to_string();
    let mut interval = 1.0f64;
    let mut window_secs = 60.0f64;
    let mut once = false;
    let mut no_clients = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--url" => {
                i += 1;
                if i < args.len() {
                    url = args[i].clone();
                }
            }
            "-i" | "--interval" => {
                i += 1;
                if i < args.len() {
                    interval = args[i].parse().unwrap_or(1.0);
                }
            }
            "--window" => {
                i += 1;
                if i < args.len() {
                    window_secs = args[i].parse().unwrap_or(60.0);
                }
            }
            "--once" => once = true,
            "--no-clients" => no_clients = true,
            _ => {}
        }
        i += 1;
    }

    let ep = parse_url(&url);
    color_on();

    let model: String = get_json(&ep, "/v1/models")
        .ok()
        .and_then(|(_, doc)| {
            doc.get("data")
                .and_then(Value::as_array)
                .and_then(|d| d.first())
                .and_then(|d| d.get("id"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_default();

    let mut window = Window::new(window_secs);
    let mut load = Load {
        ok: false,
        error: "not fetched yet".into(),
        ..Default::default()
    };
    let mut metrics: Option<Prom> = None;
    let mut metrics_probe_due = 0.0f64;

    let termios = if once {
        None
    } else {
        cbreak()
    };
    let interactive = termios.is_some();
    if !once {
        let _ = write!(std::io::stdout(), "\x1b[?25l");
    }

    let mut stop = false;
    loop {
        let now = now();
        let fetched = fetch_loads(&ep);
        let rates;
        if fetched.ok {
            window.sample(now, fetched.counters);
            load = fetched;
            rates = window.rates;
        } else {
            rates = None;
            if !load.ok {
                load = fetched; // carry error text
            }
        }

        if metrics.is_none() || now >= metrics_probe_due {
            metrics = fetch_metrics(&ep);
            if metrics.is_none() {
                metrics_probe_due = now + 10.0;
            }
        }
        let clients = if no_clients {
            None
        } else {
            probe_clients(ep.port)
        };

        let frame = render(
            &model,
            &url,
            &load,
            rates.as_ref(),
            &window,
            clients,
            metrics.as_ref(),
            now,
            interval,
            once,
        );
        if once {
            print!("{frame}");
            break;
        }
        let _ = write!(std::io::stdout(), "\x1b[H{frame}\x1b[J");
        let _ = std::io::stdout().flush();

        // sleep in slices so q/Ctrl-C is responsive
        let deadline = Instant::now() + Duration::from_secs_f64(interval.max(0.05));
        while !stop && Instant::now() < deadline {
            let remain = deadline.saturating_duration_since(Instant::now());
            let ms = remain.as_millis().min(100).max(1) as i32;
            if interactive {
                if let Some(b) = read_key(ms) {
                    if matches!(b, b'q' | b'Q' | 0x1b | 0x03) {
                        stop = true;
                    }
                }
            } else {
                std::thread::sleep(Duration::from_millis(ms as u64));
            }
        }
        if stop {
            break;
        }
    }

    if !once {
        let _ = write!(std::io::stdout(), "\x1b[?25h\x1b[0m\n");
        let _ = std::io::stdout().flush();
    }
    drop(termios);

    if once && !load.ok {
        std::process::exit(2);
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prom_parser_and_quantiles() {
        let text = "sglang:time_to_first_token_seconds_bucket{le=\"0.01\"} 2\n\
                    sglang:time_to_first_token_seconds_bucket{le=\"0.1\"} 5\n\
                    sglang:time_to_first_token_seconds_bucket{le=\"0.5\"} 20\n\
                    sglang:time_to_first_token_seconds_bucket{le=\"1\"} 40\n\
                    sglang:time_to_first_token_seconds_bucket{le=\"2\"} 55\n\
                    sglang:time_to_first_token_seconds_bucket{le=\"5\"} 58\n\
                    sglang:time_to_first_token_seconds_bucket{le=\"10\"} 59\n\
                    sglang:time_to_first_token_seconds_bucket{le=\"30\"} 60\n\
                    sglang:time_to_first_token_seconds_bucket{le=\"+Inf\"} 60\n\
                    sglang:time_to_first_token_seconds_sum 72.5\n\
                    sglang:time_to_first_token_seconds_count 60\n\
                    sglang:num_running_reqs{priority=\"None\"} 3\n\
                    sglang:num_running_reqs{priority=\"1\"} 2\n";
        let p = parse_prom(text);
        // p50: target=30 -> bucket (0.5,1] holds values 21..40 -> 0.5 + 0.5*(10/20)
        let q50 = hist_quantile(&p, "sglang:time_to_first_token_seconds", 0.5).unwrap();
        assert!((q50 - 0.75).abs() < 1e-9, "q50 = {q50}");
        // p95: target=57 -> bucket (2,5] holds 56..58 -> 2 + 3*(2/3) = 4
        let q95 = hist_quantile(&p, "sglang:time_to_first_token_seconds", 0.95).unwrap();
        assert!((q95 - 4.0).abs() < 1e-9, "q95 = {q95}");
        // p99: target=59.4 -> bucket (10,30] holds the 60th -> 10 + 20*0.4 = 18
        let q99 = hist_quantile(&p, "sglang:time_to_first_token_seconds", 0.99).unwrap();
        assert!((q99 - 18.0).abs() < 1e-6, "q99 = {q99}");
        assert_eq!(hist_quantile(&p, "sglang:e2e_request_latency_seconds", 0.5), None);
        assert!((p.sum["sglang:time_to_first_token_seconds"] - 72.5).abs() < 1e-9);
        assert!((p.gauges["sglang:num_running_reqs"] - 5.0).abs() < 1e-9);

        // empty histogram
        let p2 = parse_prom("sglang:x_bucket{le=\"+Inf\"} 0\nsglang:x_count 0\n");
        assert_eq!(hist_quantile(&p2, "sglang:x", 0.5), None);
    }

    #[test]
    fn window_rates_and_reset() {
        let mut w = Window::new(60.0);
        let counters = |t: f64| Counters {
            prefill_uncached_tokens: 0.0,
            prefill_busy_us: 0.0,
            decode_m0: t,
            decode_m1: t,
            decode_m2: 10_000.0 * t,
            decode_m4: t,
            decode_m5: 10.0 * t,
        };
        for t in [100.0, 101.0, 102.0, 103.0] {
            w.sample(t, counters(t));
        }
        let r = w.rates.unwrap();
        assert!((r.gen_tok_s - 10.0).abs() < 1e-9);
        assert!((r.step_ms.unwrap() - 10.0).abs() < 1e-9);
        assert!((r.bsz.unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(w.history.len(), 3);

        // server restart: counters drop -> window resets, no stale rates
        w.sample(
            104.0,
            Counters {
                prefill_uncached_tokens: 0.0,
                prefill_busy_us: 0.0,
                decode_m0: 1.0,
                decode_m1: 1.0,
                decode_m2: 5000.0,
                decode_m4: 1.0,
                decode_m5: 5.0,
            },
        );
        assert!(w.rates.is_none());
        assert!(w.history.is_empty());

        // rates rebuild from the fresh counters
        w.sample(
            105.0,
            Counters {
                prefill_uncached_tokens: 0.0,
                prefill_busy_us: 0.0,
                decode_m0: 2.0,
                decode_m1: 2.0,
                decode_m2: 10_000.0,
                decode_m4: 2.0,
                decode_m5: 15.0,
            },
        );
        assert!((w.rates.unwrap().gen_tok_s - 10.0).abs() < 1e-9);
    }

    #[test]
    fn url_parsing() {
        let ep = parse_url("http://127.0.0.1:30000");
        assert_eq!((ep.host.as_str(), ep.port), ("127.0.0.1", 30000));
        let ep = parse_url("http://host:30000/v1");
        assert_eq!((ep.host.as_str(), ep.port), ("host", 30000));
        let ep = parse_url("10.0.0.5:8080");
        assert_eq!((ep.host.as_str(), ep.port), ("10.0.0.5", 8080));
    }

    #[test]
    fn vis_len_counts_chars_not_bytes() {
        assert_eq!(vis_len("abc"), 3);
        assert_eq!(vis_len("█▁░"), 3);
        assert_eq!(vis_len("\u{1b}[36m███\u{1b}[0m"), 3);
    }
}
