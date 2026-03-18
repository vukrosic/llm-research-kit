#!/usr/bin/env python3
"""Real-time optimization dashboard. Run: python3 dashboard.py
Tunneled via: ssh -L 9091:localhost:5000 ...
"""
from flask import Flask, Response
from collections import defaultdict
import json, os, subprocess, time

app = Flask(__name__)

TIERS = [
    ('T1', '5s', 'Wide Exploration', 20, 8),
    ('T2', '10s', 'Scaling Signal', 12, 5),
    ('T3', '20s', 'Confirmation', 8, 3),
    ('T4', '30s', 'Validation', 5, 3),
    ('T5', '60s', 'Extended', 5, 3),
    ('T6', '90s', 'Extended', 5, 3),
    ('T7', '120s', 'Final', 5, 1),
]

SWEEP_QUEUE_CANDIDATES = [
    'optimization/queue_duration_lr.json',
    'optimization/queue.json',
]
SWEEP_RESULTS_ROOTS = [
    'results/duration_lr_sweep',
    'results',
]
SWEEP_ANALYSIS_DIR = 'optimization/duration_lr_analysis'


def get_gpu_info():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit',
             '--format=csv,noheader,nounits'], text=True, timeout=3
        ).strip()
        parts = [p.strip() for p in out.split(',')]
        return dict(name=parts[0], mem_used=parts[1], mem_total=parts[2],
                    util=parts[3], temp=parts[4], power=parts[5], power_limit=parts[6],
                    mem_pct=float(parts[1])/float(parts[2])*100)
    except:
        return None


def get_status():
    try:
        with open('optimization/status.json') as f:
            return json.load(f)
    except:
        return {}


def load_all_results():
    results = []
    for root_base in SWEEP_RESULTS_ROOTS:
        if not os.path.exists(root_base):
            continue
        for root, dirs, files in os.walk(root_base):
            for f in files:
                if f.endswith('.json'):
                    try:
                        with open(os.path.join(root, f)) as fh:
                            r = json.load(fh)
                        if r.get('status') == 'done' and 'val_loss' in r:
                            results.append(r)
                    except:
                        pass
    return results


def get_queue():
    for path in SWEEP_QUEUE_CANDIDATES:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    queue = json.load(f)
                if queue:
                    return queue
            except:
                pass
    return []


def get_queue_path():
    for path in SWEEP_QUEUE_CANDIDATES:
        if os.path.exists(path):
            return path
    return 'optimization/queue.json'


def get_analysis_summary():
    summary_path = os.path.join(SWEEP_ANALYSIS_DIR, 'summary.txt')
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path) as f:
            return f.read()
    except:
        return None


def classify_duration(r):
    ts = r.get('train_seconds', 0)
    if ts <= 0:
        tt = r.get('training_time', 0)
        if tt <= 7: return '5s'
        elif tt <= 15: return '10s'
        elif tt <= 25: return '20s'
        else: return f'{int(tt)}s'
    if ts <= 5: return '5s'
    elif ts <= 10: return '10s'
    elif ts <= 20: return '20s'
    elif ts <= 30: return '30s'
    else: return f'{ts}s'


def short_config(changes):
    if not changes:
        return 'default'
    parts = []
    for k, v in changes.items():
        if k == 'batch_size': parts.append(f'bs={v}')
        elif k == 'muon_lr': parts.append(f'lr={v}')
        elif k == 'adamw_lr': continue  # skip, derived from muon_lr
        elif k == 'weight_decay': parts.append(f'wd={v}')
        elif k == 'grad_clip': parts.append(f'gc={v}')
        else: parts.append(f'{k}={v}')
    return ', '.join(parts) if parts else 'default'


def get_tier_counts(results):
    counts = defaultdict(int)
    for r in results:
        counts[classify_duration(r)] += 1
    return counts


def render_tier_pipeline(results, queue):
    counts = get_tier_counts(results)
    # Find current tier being worked on
    running = [e for e in queue if e.get('status') == 'running']
    pending = [e for e in queue if e.get('status') == 'pending']
    current_dur = None
    if running:
        current_dur = f"{running[0].get('train_seconds', '?')}s"
    elif pending:
        current_dur = f"{pending[0].get('train_seconds', '?')}s"

    pills = ''
    for tid, dur, desc, target, promote in TIERS:
        n = counts.get(dur, 0)
        is_active = (dur == current_dur)
        is_done = n >= promote  # has enough to promote
        if is_active:
            cls = 'tier-active'
            badge = 'RUNNING'
        elif is_done:
            cls = 'tier-done'
            badge = f'{n} done'
        elif n > 0:
            cls = 'tier-partial'
            badge = f'{n}/{target}'
        else:
            cls = 'tier-pending'
            badge = 'waiting'

        pills += f'''<div class="tier-pill {cls}">
            <div class="tier-id">{tid}</div>
            <div class="tier-dur">{dur}</div>
            <div class="tier-desc">{desc}</div>
            <div class="tier-badge">{badge}</div>
            <div class="tier-promote">Top {promote} advance</div>
        </div>'''

    arrows = '<span class="tier-arrow">&#9654;</span>' * 3
    return f'<div class="tier-pipeline">{pills}</div>'


def render_leaderboard(results, category, limit=8, all_results=None):
    filtered = [r for r in results if classify_duration(r) == category]
    if not filtered:
        return '<p class="dim">No experiments yet</p>'

    filtered.sort(key=lambda r: r['val_loss'])
    seen = set()
    deduped = []
    for r in filtered:
        cs = json.dumps(r.get('changes', {}), sort_keys=True)
        if cs not in seen:
            seen.add(cs)
            deduped.append(r)
    filtered = deduped[:limit]

    # Get rank at other tiers for scaling indicator
    def get_rank_at(changes, dur):
        cs = json.dumps(changes, sort_keys=True)
        tier_results = sorted(
            [r for r in (all_results or []) if classify_duration(r) == dur],
            key=lambda r: r['val_loss']
        )
        seen2 = set()
        for i, r in enumerate(tier_results):
            rcs = json.dumps(r.get('changes', {}), sort_keys=True)
            if rcs not in seen2:
                seen2.add(rcs)
                if rcs == cs:
                    return i + 1
        return None

    best = filtered[0]['val_loss']
    rows = ''
    for i, r in enumerate(filtered):
        delta = r['val_loss'] - best
        cls = 'best' if i == 0 else ''
        cs = short_config(r.get('changes', {}))
        tps = r.get('tokens_per_second', 0)
        tps_str = f'{tps/1000:.0f}K' if tps > 0 else '-'

        # Scaling trend: compare this config's rank HERE vs its rank at 120s (or longest available)
        # Green arrow UP = config ranks HIGHER at longer training (scales well)
        # Red arrow DOWN = config ranks LOWER at longer training (doesn't scale)
        trend = ''
        changes = r.get('changes', {})
        if all_results:
            # Find longest tier where this config has data
            for target_dur in ['120s', '90s', '60s', '30s', '20s']:
                if target_dur == category:
                    break
                rt = get_rank_at(changes, target_dur)
                if rt is not None:
                    if rt < i + 1:
                        trend = f'<span class="trend-up" title="Rank #{i+1} here -> #{rt} at {target_dur}">&#9650;{i+1-rt}</span>'
                    elif rt > i + 1:
                        trend = f'<span class="trend-down" title="Rank #{i+1} here -> #{rt} at {target_dur}">&#9660;{rt-i-1}</span>'
                    else:
                        trend = f'<span class="trend-flat" title="Same rank at {target_dur}">&#9644;</span>'
                    break

        rows += f'''<tr class="{cls}">
            <td>{i+1}</td>
            <td class="config-col">{cs}</td>
            <td><b>{r["val_loss"]:.4f}</b></td>
            <td>{r.get("steps","-")}</td>
            <td>{tps_str}</td>
            <td>{"+" if delta>=0 else ""}{delta:.4f}</td>
            <td>{trend}</td>
        </tr>'''

    return f'''<table>
        <tr><th>#</th><th>Config</th><th>Val Loss</th><th>Steps</th><th>TPS</th><th>vs Best</th><th title="Green UP = this config ranks higher at longer training (scales well). Red DOWN = ranks lower (doesn't scale). Compares to longest available tier.">Scale</th></tr>
        {rows}
    </table>'''


def render_scaling_table(results):
    config_results = defaultdict(dict)
    for r in results:
        cat = classify_duration(r)
        key = json.dumps(r.get('changes', {}), sort_keys=True)
        label = short_config(r.get('changes', {}))
        if cat not in config_results[(key, label)] or r['val_loss'] < config_results[(key, label)][cat]['val_loss']:
            config_results[(key, label)][cat] = r

    multi = {k: v for k, v in config_results.items() if len(v) >= 2}
    if not multi:
        return '<p class="dim">Run configs at multiple durations to see scaling trends</p>'

    # Find all durations present
    all_durs = sorted(set(
        cat for (_, _), durs in config_results.items() for cat in durs.keys()
    ), key=lambda d: float(d.rstrip('s')))

    # Get ranks per tier
    tier_ranks = {}
    for dur in all_durs:
        tier_items = []
        for (key, label), durations in config_results.items():
            if dur in durations:
                tier_items.append((durations[dur]['val_loss'], key))
        tier_items.sort()
        tier_ranks[dur] = {k: i+1 for i, (_, k) in enumerate(tier_items)}

    def sort_key(item):
        _, durations = item
        # Sort by longest duration result first
        for d in reversed(all_durs):
            if d in durations:
                return durations[d]['val_loss']
        return 99

    rows = ''
    for (key, label), durations in sorted(multi.items(), key=sort_key):
        cells = ''
        for dur in all_durs:
            if dur in durations:
                v = durations[dur]['val_loss']
                rank = tier_ranks.get(dur, {}).get(key, '?')
                cells += f'<td>{v:.4f} <span class="rank-badge">#{rank}</span></td>'
            else:
                cells += '<td class="dim">-</td>'

        # Total drop from shortest to longest
        vals = [(float(d.rstrip('s')), durations[d]['val_loss']) for d in all_durs if d in durations]
        if len(vals) >= 2:
            drop = vals[0][1] - vals[-1][1]
            cells += f'<td class="best">{drop:.3f}</td>'
        else:
            cells += '<td class="dim">-</td>'

        # Rank trend
        ranks = []
        for dur in all_durs:
            r = tier_ranks.get(dur, {}).get(key)
            if r is not None:
                ranks.append(f'#{r}')
        trend = ' -> '.join(ranks)
        cells += f'<td>{trend}</td>'

        rows += f'<tr><td class="config-col">{label}</td>{cells}</tr>'

    dur_headers = ''.join(f'<th>{d}</th>' for d in all_durs)
    return f'''<table>
        <tr><th>Config</th>{dur_headers}<th>Total Drop</th><th>Rank Trend</th></tr>
        {rows}
    </table>'''


def render_queue(queue):
    # Only show pending/running, then last 3 done
    active = [e for e in queue if e['status'] in ('pending', 'running')]
    done = [e for e in queue if e['status'] == 'done'][-3:]
    show = active + done
    if not show:
        return '<p class="dim">Queue empty - no experiments scheduled</p>'

    rows = ''
    for exp in show:
        s = exp['status']
        loss = f'{exp["val_loss"]:.4f}' if 'val_loss' in exp else '-'
        dur = f'{exp.get("train_seconds", "?")}s'
        cs = short_config(exp.get('changes', {}))
        status_cls = s
        status_icon = {'pending': '&#9711;', 'running': '&#9654;', 'done': '&#10003;', 'failed': '&#10007;', 'oom': '&#10007;'}.get(s, '?')
        rows += f'''<tr class="{status_cls}">
            <td class="{status_cls}">{status_icon} {s.upper()}</td>
            <td>{dur}</td>
            <td>{cs}</td>
            <td>{loss}</td>
            <td class="dim">{exp.get("hypothesis","")[:60]}</td>
        </tr>'''

    return f'''<table>
        <tr><th>Status</th><th>Dur</th><th>Config</th><th>Val Loss</th><th>Hypothesis</th></tr>
        {rows}
    </table>'''


def render_sweep_status(status, queue, results):
    active = [e for e in queue if e.get('status') == 'running']
    pending = [e for e in queue if e.get('status') == 'pending']
    done = [e for e in queue if e.get('status') == 'done']
    failed = [e for e in queue if e.get('status') in ('failed', 'oom')]
    latest = sorted(done, key=lambda r: (r.get('train_seconds', 0), r.get('seed', 0)))
    latest_done = latest[-1] if latest else None
    analysis = get_analysis_summary()

    rows = [
        f'<div><b>Queue:</b> {get_queue_path()}</div>',
        f'<div><b>Running:</b> {status.get("current_exp", "none")}</div>',
        f'<div><b>Progress:</b> {status.get("progress", "0/0")}</div>',
        f'<div><b>Pending:</b> {len(pending)} | <b>Done:</b> {len(done)} | <b>Failed:</b> {len(failed)}</div>',
    ]

    if latest_done:
        rows.append(
            f'<div><b>Last result:</b> {latest_done["exp_id"]} '
            f'({latest_done.get("train_seconds", "?")}s, lr={latest_done.get("changes", {}).get("muon_lr", "?")}) '
            f'val_loss={latest_done["val_loss"]:.4f}</div>'
        )
    if analysis:
        first_lines = "\n".join(analysis.strip().splitlines()[:4])
        rows.append(f'<pre class="analysis-summary">{first_lines}</pre>')
    else:
        rows.append(f'<div class="dim">Analysis files will appear in {SWEEP_ANALYSIS_DIR} after the batch finishes.</div>')

    return ''.join(f'<div class="sweep-line">{row}</div>' for row in rows)


@app.route('/')
def dashboard():
    os.chdir('/root/llm-research-kit')
    gpu = get_gpu_info()
    status = get_status()
    results = load_all_results()
    queue = get_queue()
    ts = time.strftime('%H:%M:%S')

    # GPU bar
    if gpu:
        gpu_html = f'''<span class="gpu-name">{gpu["name"]}</span>
        <div class="gpu-row">
            <span class="label">VRAM</span>
            <div class="bar-bg"><div class="bar-fill bar-vram" style="width:{gpu["mem_pct"]}%"></div></div>
            <span>{gpu["mem_used"]}/{gpu["mem_total"]}MiB</span>
        </div>
        <div class="gpu-row">
            <span class="label">GPU</span>
            <div class="bar-bg"><div class="bar-fill bar-util" style="width:{gpu["util"]}%"></div></div>
            <span>{gpu["util"]}%</span>
            <span class="dim">| {gpu["temp"]}C | {gpu["power"]}/{gpu["power_limit"]}W</span>
        </div>'''
    else:
        gpu_html = '<span class="dim">GPU unavailable</span>'

    # Live status
    is_running = bool(status.get('current_exp'))
    if is_running:
        live_html = f'''<div class="live-row">
            <div class="running-badge pulse">TRAINING</div>
            <div class="live-info">
                <span class="exp-name">{status["current_exp"]}</span>
                <span class="dim">{status.get("hypothesis","")}</span>
                <span>Progress: {status.get("progress","?")} | Started: {status.get("started","?")}</span>
            </div>
        </div>'''
    else:
        live_html = f'''<div class="live-row">
            <div class="idle-badge">IDLE</div>
            <span class="dim">Last finished: {status.get("finished","never")}</span>
        </div>'''

    # Tier pipeline
    pipeline_html = render_tier_pipeline(results, queue)

    # Queue
    queue_html = render_queue(queue)

    # Leaderboards - build dynamically for all tiers with data
    leaderboards = {}
    for tid, dur, desc, target, promote in TIERS:
        lb = render_leaderboard(results, dur, all_results=results)
        leaderboards[(tid, dur, desc)] = lb

    # Scaling
    scaling_html = render_scaling_table(results)

    # Counts
    counts = get_tier_counts(results)
    total_exps = len(results)
    total_time = sum(r.get('training_time', 0) for r in results)

    page = f'''<!DOCTYPE html>
<html>
<head>
<title>Scaling Research Dashboard</title>
<meta http-equiv="refresh" content="3">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; background: #0a0a1a; color: #c8c8d8; padding: 12px 16px; font-size: 12px; }}

  /* Header */
  .header {{ display: flex; align-items: center; gap: 12px; padding-bottom: 8px; border-bottom: 1px solid #222; margin-bottom: 10px; }}
  .header h1 {{ color: #00d4ff; font-size: 1.3em; white-space: nowrap; }}
  .header .meta {{ color: #555; font-size: 0.85em; }}

  /* Tier Pipeline */
  .tier-pipeline {{ display: flex; gap: 4px; margin-bottom: 10px; }}
  .tier-pill {{ flex: 1; background: #12122a; border: 1px solid #1e1e3a; border-radius: 6px; padding: 8px 10px; text-align: center; }}
  .tier-id {{ font-weight: bold; font-size: 1.1em; margin-bottom: 2px; }}
  .tier-dur {{ font-size: 1.2em; font-weight: bold; }}
  .tier-desc {{ color: #888; font-size: 0.8em; margin: 2px 0; }}
  .tier-badge {{ font-size: 0.85em; padding: 1px 6px; border-radius: 3px; display: inline-block; margin: 2px 0; }}
  .tier-promote {{ color: #555; font-size: 0.75em; }}
  .tier-active {{ border-color: #ffaa00; background: #1a1500; }}
  .tier-active .tier-id {{ color: #ffaa00; }}
  .tier-active .tier-dur {{ color: #ffaa00; }}
  .tier-active .tier-badge {{ background: #332200; color: #ffaa00; }}
  .tier-done {{ border-color: #00ff88; }}
  .tier-done .tier-id {{ color: #00ff88; }}
  .tier-done .tier-dur {{ color: #00ff88; }}
  .tier-done .tier-badge {{ background: #0a2a0a; color: #00ff88; }}
  .tier-partial {{ border-color: #00d4ff; }}
  .tier-partial .tier-id {{ color: #00d4ff; }}
  .tier-partial .tier-dur {{ color: #00d4ff; }}
  .tier-partial .tier-badge {{ background: #0a1a2a; color: #00d4ff; }}
  .tier-pending .tier-id {{ color: #444; }}
  .tier-pending .tier-dur {{ color: #444; }}
  .tier-pending .tier-badge {{ color: #444; }}

  /* Live status + GPU row */
  .top-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }}
  .card {{ background: #12122a; border: 1px solid #1e1e3a; border-radius: 6px; padding: 10px 12px; }}
  .section-label {{ color: #00d4ff; font-size: 0.75em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; font-weight: 600; }}

  .live-row {{ display: flex; align-items: center; gap: 10px; }}
  .live-info {{ display: flex; flex-direction: column; gap: 2px; }}
  .running-badge {{ display: inline-block; background: #332200; color: #ffaa00; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 1em; }}
  .idle-badge {{ display: inline-block; background: #1a2a1a; color: #00ff88; padding: 4px 12px; border-radius: 4px; }}
  .exp-name {{ color: #fff; font-weight: bold; }}
  .pulse {{ animation: pulse 1.5s infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}

  .gpu-name {{ color: #00d4ff; font-weight: bold; }}
  .gpu-row {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; }}
  .bar-bg {{ background: #222; border-radius: 3px; height: 10px; flex: 1; max-width: 180px; }}
  .bar-fill {{ border-radius: 3px; height: 10px; }}
  .bar-vram {{ background: linear-gradient(90deg, #00d4ff, #00ff88); }}
  .bar-util {{ background: linear-gradient(90deg, #ffaa00, #ff6644); }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
  th {{ color: #00d4ff; text-align: left; padding: 3px 5px; border-bottom: 2px solid #222; font-weight: 600; }}
  td {{ padding: 3px 5px; border-bottom: 1px solid #161630; }}
  tr:hover {{ background: #1a1a3e; }}
  .config-col {{ max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

  .best {{ color: #00ff88; font-weight: bold; }}
  .running {{ color: #ffaa00; }}
  .done {{ color: #00ff88; }}
  .failed, .oom {{ color: #ff4444; }}
  .pending {{ color: #555; }}
  .dim {{ color: #555; }}
  .label {{ color: #888; margin-right: 4px; min-width: 35px; }}

  .rank-badge {{ color: #888; font-size: 0.85em; }}
  .trend-up {{ color: #00ff88; font-weight: bold; }}
  .trend-down {{ color: #ff4444; font-weight: bold; }}
  .trend-flat {{ color: #888; }}

  /* Layout */
  .leaderboards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }}
  .full-width {{ margin-bottom: 10px; }}

  h3 {{ color: #00d4ff; font-size: 0.9em; margin: 0 0 4px; }}
  .tier-label {{ color: #00ff88; font-size: 0.8em; }}
  .sweep-line {{ margin: 3px 0; }}
  .analysis-summary {{ margin-top: 6px; padding: 8px 10px; background: #08111d; border: 1px solid #18314f; color: #9ec5ff; white-space: pre-wrap; font-size: 0.82em; }}
  .insight {{ background: #0a1a2a; border-left: 3px solid #00d4ff; padding: 6px 10px; margin: 4px 0; font-size: 0.9em; }}
  .insight b {{ color: #00ff88; }}
  .insight .warn {{ color: #ffaa00; }}
  .legend {{ color: #888; font-size: 0.8em; margin: 4px 0 8px; }}
  .legend span {{ margin-right: 12px; }}
</style>
</head>
<body>

<div class="header">
    <h1>Scaling Research</h1>
    <span class="meta">{ts} | {total_exps} experiments | {total_time:.0f}s training | auto-refresh 3s</span>
</div>

<!-- TIER PIPELINE - shows research progress -->
{pipeline_html}

<!-- LIVE STATUS + GPU -->
<div class="top-row">
  <div class="card">
    <div class="section-label">Live Status</div>
    {live_html}
  </div>
  <div class="card">
    <div class="section-label">GPU</div>
    {gpu_html}
  </div>
</div>

<!-- SWEEP STATUS -->
<div class="card full-width">
  <div class="section-label">Duration → LR Sweep Status</div>
  {render_sweep_status(status, queue, results)}
</div>

<!-- QUEUE -->
<div class="card full-width">
  <div class="section-label">Experiment Queue</div>
  {queue_html}
</div>

<!-- LEADERBOARDS -->
<div class="card full-width">
  <div class="legend">
    <b>Scale column:</b>
    <span class="trend-up">&#9650; Green UP</span> = config ranks HIGHER at longer training (good scaler, trust it)
    <span class="trend-down">&#9660; Red DOWN</span> = config ranks LOWER at longer training (misleading winner, don't trust)
    <span class="trend-flat">&#9644; Flat</span> = same rank at longer duration
  </div>
</div>
<div class="leaderboards">
  {''.join(f"""<div class="card">
    <h3>{tid}: {dur} {desc} <span class="tier-label">({counts.get(dur,0)} exp)</span></h3>
    {lb}
  </div>""" for (tid, dur, desc), lb in leaderboards.items() if counts.get(dur, 0) > 0)}
</div>

<!-- SCALING TABLE -->
<div class="card full-width">
  <div class="section-label">Scaling Analysis - How configs perform across durations (rank trend is key)</div>
  {scaling_html}
</div>

<!-- INSIGHTS -->
<div class="card full-width">
  <div class="section-label">Key Insights from Scaling Research</div>
  <div class="insight"><b>120s Winner:</b> bs=4, lr=0.012 (val_loss=4.850). This config was only #6 at 5s but rose to #1 by 60s and held through 120s.</div>
  <div class="insight"><b>Rankings invert with duration.</b> The 5s winner (gc=0.5) falls to last place by 20s. The 20-30s winner (bs=3) gets overtaken by bs=4 at 60s+. <span class="warn">Never trust short-duration rankings for production.</span></div>
  <div class="insight"><b>Batch size sweet spot is duration-dependent:</b> bs=3 wins at 20-30s (more steps), bs=4 wins at 60-120s (better gradient signal per step). bs=6 and bs=8 are always worse.</div>
  <div class="insight"><b>LR is stable:</b> lr=0.010-0.012 works across all durations. Not worth extensive tuning.</div>
  <div class="insight"><b>Grad clip doesn't scale:</b> gc=0.5 helps at 5s (saves bad early steps) but actively hurts at 20s+ (restricts learning signal).</div>
  <div class="insight"><b>Weight decay is second-order:</b> wd=0.1 gives small consistent benefit but never makes or breaks a config.</div>
  <div class="insight"><span class="warn">Implication:</span> For this 88M model, screen at 10-20s (not 5s), use bs=4 + lr=0.012 for any training run over 60s.</div>
</div>

</body>
</html>'''

    return Response(page, mimetype='text/html')


if __name__ == '__main__':
    os.chdir('/root/llm-research-kit')
    print("Dashboard on http://0.0.0.0:5000 (refresh every 3s)")
    app.run(host='0.0.0.0', port=5000, debug=False)
