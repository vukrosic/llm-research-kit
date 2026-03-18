#!/usr/bin/env python3
"""Real-time optimization dashboard. Run: python3 dashboard.py
Tunneled via: ssh -L 9091:localhost:5000 ...
"""
from flask import Flask, Response
import json, os, subprocess, time, glob

app = Flask(__name__)

def read_file(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except:
        return default


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
    """Load all result JSON files from results/ directory."""
    results = []
    for root, dirs, files in os.walk('results'):
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
    try:
        with open('optimization/queue.json') as f:
            return json.load(f)
    except:
        return []


def classify_duration(r):
    """Classify a result into 5s/10s/20s category based on train_seconds."""
    ts = r.get('train_seconds', 0)
    if ts <= 0:
        # Fallback: infer from training_time
        tt = r.get('training_time', 0)
        if tt <= 7: return '5s'
        elif tt <= 15: return '10s'
        elif tt <= 25: return '20s'
        else: return f'{int(tt)}s'
    if ts <= 5: return '5s'
    elif ts <= 10: return '10s'
    elif ts <= 20: return '20s'
    else: return f'{ts}s'


def changes_str(changes):
    if not changes:
        return 'default'
    parts = []
    for k, v in changes.items():
        if k in ('muon_lr', 'adamw_lr'):
            parts.append(f'{k}={v}')
        elif k == 'batch_size':
            parts.append(f'bs={v}')
        elif k == 'weight_decay':
            parts.append(f'wd={v}')
        elif k == 'grad_clip':
            parts.append(f'gc={v}')
        else:
            parts.append(f'{k}={v}')
    return ', '.join(parts)


def render_leaderboard_section(results, category, limit=8):
    """Render a leaderboard table for one duration category."""
    filtered = [r for r in results if classify_duration(r) == category]
    if not filtered:
        return f'<p class="dim">No {category} experiments yet</p>'

    filtered.sort(key=lambda r: r['val_loss'])
    # Deduplicate by exp_id (keep best)
    seen = set()
    deduped = []
    for r in filtered:
        eid = r['exp_id']
        if eid not in seen:
            seen.add(eid)
            deduped.append(r)
    filtered = deduped[:limit]

    best = filtered[0]['val_loss']
    rows = ''
    for i, r in enumerate(filtered):
        delta = r['val_loss'] - best
        cls = 'best' if i == 0 else ''
        cs = changes_str(r.get('changes', {}))
        tps = r.get('tokens_per_second', 0)
        tps_str = f'{tps/1000:.0f}K' if tps > 0 else '-'
        rows += f'''<tr class="{cls}">
            <td>{i+1}</td>
            <td>{r["exp_id"]}</td>
            <td>{r["val_loss"]:.4f}</td>
            <td>{r.get("steps","-")}</td>
            <td>{tps_str}</td>
            <td>{"+" if delta>=0 else ""}{delta:.4f}</td>
            <td class="dim">{cs}</td>
        </tr>'''

    return f'''<table>
        <tr><th>#</th><th>Experiment</th><th>Val Loss</th><th>Steps</th><th>TPS</th><th>vs Best</th><th>Config</th></tr>
        {rows}
    </table>'''


def render_scaling_table(results):
    """Render the scaling comparison table for configs that appear at multiple durations."""
    # Group by config signature (changes dict)
    from collections import defaultdict
    config_results = defaultdict(dict)
    for r in results:
        cat = classify_duration(r)
        key = json.dumps(r.get('changes', {}), sort_keys=True)
        label = changes_str(r.get('changes', {}))
        if cat not in config_results[(key, label)] or r['val_loss'] < config_results[(key, label)][cat]['val_loss']:
            config_results[(key, label)][cat] = r

    # Only show configs that appear in 2+ duration categories
    multi = {k: v for k, v in config_results.items() if len(v) >= 2}
    if not multi:
        return '<p class="dim">Run configs at multiple durations to see scaling trends</p>'

    rows = ''
    # Sort by 20s val_loss, then 10s, then 5s
    def sort_key(item):
        _, durations = item
        return (durations.get('20s', {}).get('val_loss', 99),
                durations.get('10s', {}).get('val_loss', 99),
                durations.get('5s', {}).get('val_loss', 99))

    for (key, label), durations in sorted(multi.items(), key=sort_key):
        v5 = durations.get('5s', {}).get('val_loss')
        v10 = durations.get('10s', {}).get('val_loss')
        v20 = durations.get('20s', {}).get('val_loss')
        s5 = f'{v5:.4f}' if v5 else '-'
        s10 = f'{v10:.4f}' if v10 else '-'
        s20 = f'{v20:.4f}' if v20 else '-'
        if v5 and v20:
            delta = f'{v20-v5:.3f}'
        else:
            delta = '-'
        rows += f'''<tr>
            <td>{label}</td>
            <td>{s5}</td><td>{s10}</td><td>{s20}</td>
            <td>{delta}</td>
        </tr>'''

    return f'''<table>
        <tr><th>Config</th><th>5s</th><th>10s</th><th>20s</th><th>5s→20s Δ</th></tr>
        {rows}
    </table>'''


@app.route('/')
def dashboard():
    os.chdir('/root/llm-research-kit')
    gpu = get_gpu_info()
    status = get_status()
    results = load_all_results()
    queue = get_queue()
    ts = time.strftime('%H:%M:%S')

    # GPU section
    if gpu:
        gpu_html = f'''
        <span class="stat-value">{gpu["name"]}</span>
        <div class="row">
            <span class="label">VRAM</span>
            <div class="bar-bg"><div class="bar-fill bar-vram" style="width:{gpu["mem_pct"]}%"></div></div>
            <span>{gpu["mem_used"]}/{gpu["mem_total"]}MiB ({gpu["mem_pct"]:.0f}%)</span>
        </div>
        <div class="row">
            <span class="label">Util</span>
            <div class="bar-bg"><div class="bar-fill bar-util" style="width:{gpu["util"]}%"></div></div>
            <span>{gpu["util"]}%</span>
        </div>
        <div class="row"><span class="label">Temp</span> {gpu["temp"]}C | <span class="label">Power</span> {gpu["power"]}/{gpu["power_limit"]}W</div>
        '''
    else:
        gpu_html = '<span class="dim">GPU unavailable</span>'

    # Status section
    is_running = bool(status.get('current_exp'))
    if is_running:
        status_html = f'''
        <div class="running-badge pulse">TRAINING</div>
        <div class="exp-name">{status["current_exp"]}</div>
        <div class="dim">{status.get("hypothesis","")}</div>
        <div>Batch: {status.get("batch","?")} | Progress: {status.get("progress","?")}</div>
        <div class="dim">Started: {status.get("started","?")}</div>
        '''
    else:
        status_html = f'''
        <div class="idle-badge">IDLE</div>
        <div class="dim">Last finished: {status.get("finished","never")}</div>
        '''

    # Queue section
    queue_rows = ''
    for exp in queue:
        s = exp['status']
        loss = f'{exp["val_loss"]:.4f}' if 'val_loss' in exp else '-'
        steps = exp.get('steps', '-')
        dur = f'{exp.get("train_seconds", "?")}s'
        queue_rows += f'''<tr class="{s}">
            <td class="{s}">{s.upper()}</td>
            <td>{exp["exp_id"]}</td>
            <td>{loss}</td><td>{steps}</td><td>{dur}</td>
            <td class="dim">{exp.get("hypothesis","")[:50]}</td>
        </tr>'''
    if queue_rows:
        queue_html = f'''<table>
            <tr><th>Status</th><th>Experiment</th><th>Val Loss</th><th>Steps</th><th>Duration</th><th>Hypothesis</th></tr>
            {queue_rows}
        </table>'''
    else:
        queue_html = '<p class="dim">Queue empty</p>'

    # Leaderboards
    lb_5s = render_leaderboard_section(results, '5s')
    lb_10s = render_leaderboard_section(results, '10s')
    lb_20s = render_leaderboard_section(results, '20s')

    # Scaling table
    scaling_html = render_scaling_table(results)

    # Stats
    total_exps = len(results)
    total_time = sum(r.get('training_time', 0) for r in results)

    page = f'''<!DOCTYPE html>
<html>
<head>
<title>LLM Optimization Dashboard</title>
<meta http-equiv="refresh" content="3">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace; background: #0a0a1a; color: #c8c8d8; padding: 16px; font-size: 13px; }}
  h1 {{ color: #00d4ff; font-size: 1.5em; display: inline; }}
  .header {{ display: flex; align-items: center; gap: 15px; margin-bottom: 12px; border-bottom: 1px solid #222; padding-bottom: 10px; }}
  .header .ts {{ color: #555; font-size: 0.85em; }}
  .header .total {{ color: #888; font-size: 0.85em; }}
  h2 {{ color: #00ff88; font-size: 1.05em; margin: 0 0 8px; }}
  h3 {{ color: #00d4ff; font-size: 0.95em; margin: 12px 0 6px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  .card {{ background: #12122a; border: 1px solid #1e1e3a; border-radius: 6px; padding: 12px; }}
  .card-full {{ grid-column: 1 / -1; }}
  .card-top {{ grid-column: 1 / -1; display: grid; grid-template-columns: 1fr 2fr; gap: 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
  th {{ color: #00d4ff; text-align: left; padding: 4px 6px; border-bottom: 2px solid #222; font-weight: 600; }}
  td {{ padding: 4px 6px; border-bottom: 1px solid #161630; }}
  tr:hover {{ background: #1a1a3e; }}
  .best {{ color: #00ff88; font-weight: bold; }}
  .running {{ color: #ffaa00; }}
  .done {{ color: #00ff88; }}
  .failed, .oom {{ color: #ff4444; }}
  .pending {{ color: #555; }}
  .dim {{ color: #666; }}
  .label {{ color: #888; margin-right: 5px; }}
  .row {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; }}
  .bar-bg {{ background: #222; border-radius: 3px; height: 12px; flex: 1; max-width: 200px; }}
  .bar-fill {{ border-radius: 3px; height: 12px; }}
  .bar-vram {{ background: linear-gradient(90deg, #00d4ff, #00ff88); }}
  .bar-util {{ background: linear-gradient(90deg, #ffaa00, #ff6644); }}
  .stat-value {{ color: #00d4ff; font-size: 1.1em; }}
  .running-badge {{ display: inline-block; background: #332200; color: #ffaa00; padding: 3px 12px; border-radius: 4px; font-weight: bold; font-size: 1.1em; }}
  .idle-badge {{ display: inline-block; background: #1a2a1a; color: #00ff88; padding: 3px 12px; border-radius: 4px; }}
  .exp-name {{ color: #fff; font-size: 1.1em; margin: 4px 0; }}
  .pulse {{ animation: pulse 1.5s infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}
  .tabs {{ display: flex; gap: 0; margin-bottom: 8px; }}
  .tab {{ padding: 6px 16px; background: #0e0e22; border: 1px solid #222; color: #888; cursor: pointer; font-size: 0.9em; }}
  .tab:first-child {{ border-radius: 4px 0 0 4px; }}
  .tab:last-child {{ border-radius: 0 4px 4px 0; }}
  .tab.active {{ background: #1a1a3a; color: #00ff88; border-color: #00ff88; }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}
  .section-label {{ color: #00d4ff; font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }}
</style>
</head>
<body>

<div class="header">
    <h1>LLM Optimization</h1>
    <span class="ts">{ts}</span>
    <span class="total">{total_exps} experiments | {total_time:.0f}s total training</span>
    <span class="ts">auto-refresh 3s</span>
</div>

<!-- ROW 1: Status + GPU -->
<div class="grid">
  <div class="card-top">
    <div class="card">
      <div class="section-label">Live Status</div>
      {status_html}
    </div>
    <div class="card">
      <div class="section-label">GPU</div>
      {gpu_html}
    </div>
  </div>

  <!-- Current Queue -->
  <div class="card card-full">
    <div class="section-label">Current Queue</div>
    {queue_html}
  </div>

  <!-- Leaderboards: 3 categories side by side -->
  <div class="card card-full">
    <div class="section-label">Leaderboards</div>
    <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
      <div>
        <h3>5s Experiments</h3>
        {lb_5s}
      </div>
      <div>
        <h3>10s Experiments</h3>
        {lb_10s}
      </div>
      <div>
        <h3>20s Experiments</h3>
        {lb_20s}
      </div>
    </div>
  </div>

  <!-- Scaling Analysis -->
  <div class="card card-full">
    <div class="section-label">Scaling Analysis (configs tested at multiple durations)</div>
    {scaling_html}
  </div>

</div>
</body>
</html>'''

    return Response(page, mimetype='text/html')


if __name__ == '__main__':
    os.chdir('/root/llm-research-kit')
    print("Dashboard on http://0.0.0.0:5000 (refresh every 3s)")
    app.run(host='0.0.0.0', port=5000, debug=False)
