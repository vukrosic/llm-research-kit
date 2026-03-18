#!/usr/bin/env python3
"""Real-time optimization dashboard. Run: python3 dashboard.py"""
from flask import Flask, render_template_string
import json, os, subprocess, time, glob

app = Flask(__name__)

TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<title>LLM Optimization Dashboard</title>
<meta http-equiv="refresh" content="10">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Courier New', monospace; background: #0a0a1a; color: #c8c8d8; padding: 20px; }
  h1 { color: #00d4ff; margin-bottom: 5px; font-size: 1.8em; }
  h2 { color: #00ff88; margin: 20px 0 10px; font-size: 1.2em; border-bottom: 1px solid #222; padding-bottom: 5px; }
  .subtitle { color: #666; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
  .card { background: #12122a; border: 1px solid #222; border-radius: 8px; padding: 15px; }
  .card-full { grid-column: 1 / -1; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
  th { color: #00d4ff; text-align: left; padding: 6px 8px; border-bottom: 2px solid #333; }
  td { padding: 6px 8px; border-bottom: 1px solid #1a1a2e; }
  tr:hover { background: #1a1a3e; }
  .best { color: #00ff88; font-weight: bold; }
  .running { color: #ffaa00; }
  .done { color: #00ff88; }
  .failed, .oom { color: #ff4444; }
  .pending { color: #666; }
  .stat { display: inline-block; margin: 5px 15px 5px 0; }
  .stat-label { color: #666; font-size: 0.8em; }
  .stat-value { color: #00d4ff; font-size: 1.3em; }
  .bar-bg { background: #222; border-radius: 3px; height: 16px; margin: 3px 0; }
  .bar-fill { border-radius: 3px; height: 16px; transition: width 0.5s; }
  .bar-vram { background: linear-gradient(90deg, #00d4ff, #00ff88); }
  .bar-util { background: linear-gradient(90deg, #ffaa00, #ff4444); }
  pre { white-space: pre-wrap; font-size: 0.85em; line-height: 1.5; }
  .pulse { animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
</style>
</head>
<body>
<h1>LLM Research Kit - Optimization Dashboard</h1>
<p class="subtitle">{{ timestamp }} | Auto-refresh: 10s</p>

<div class="grid">
  <!-- GPU Status -->
  <div class="card">
    <h2>GPU</h2>
    {{ gpu_html | safe }}
  </div>

  <!-- Current Status -->
  <div class="card">
    <h2>Status</h2>
    {{ status_html | safe }}
  </div>

  <!-- Leaderboard -->
  <div class="card card-full">
    <h2>Leaderboard (Top Configs)</h2>
    {{ leaderboard_html | safe }}
  </div>

  <!-- Current Queue -->
  <div class="card card-full">
    <h2>Current Batch</h2>
    {{ queue_html | safe }}
  </div>

  <!-- Insights -->
  <div class="card">
    <h2>Insights</h2>
    <pre>{{ insights }}</pre>
  </div>

  <!-- Plan Summary -->
  <div class="card">
    <h2>Plan</h2>
    <pre>{{ plan }}</pre>
  </div>
</div>
</body>
</html>"""


def read_file(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except:
        return default


def get_gpu_html():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit',
             '--format=csv,noheader,nounits'], text=True
        ).strip()
        parts = [p.strip() for p in out.split(',')]
        name, mem_used, mem_total, util, temp, power, power_limit = parts
        mem_pct = float(mem_used) / float(mem_total) * 100
        return f"""
        <div class="stat"><span class="stat-label">GPU</span><br><span class="stat-value">{name}</span></div>
        <div class="stat"><span class="stat-label">VRAM</span><br><span class="stat-value">{mem_used}/{mem_total} MiB ({mem_pct:.0f}%)</span></div>
        <div class="bar-bg"><div class="bar-fill bar-vram" style="width:{mem_pct}%"></div></div>
        <div class="stat"><span class="stat-label">Utilization</span><br><span class="stat-value">{util}%</span></div>
        <div class="bar-bg"><div class="bar-fill bar-util" style="width:{util}%"></div></div>
        <div class="stat"><span class="stat-label">Temp</span><br><span class="stat-value">{temp}C</span></div>
        <div class="stat"><span class="stat-label">Power</span><br><span class="stat-value">{power}/{power_limit} W</span></div>
        """
    except Exception as e:
        return f"<p>GPU info unavailable: {e}</p>"


def get_status_html():
    try:
        with open('optimization/status.json') as f:
            s = json.load(f)
        if s.get('current_exp'):
            return f"""
            <p class="running pulse">RUNNING: {s['current_exp']}</p>
            <p>{s.get('hypothesis', '')}</p>
            <p>Batch: {s.get('batch','?')} | Progress: {s.get('progress','?')}</p>
            <p>Started: {s.get('started','?')}</p>
            """
        else:
            return f"<p class='done'>Idle</p><p>Last finished: {s.get('finished','never')}</p>"
    except:
        return "<p class='pending'>No experiments running</p>"


def get_leaderboard_html():
    # Aggregate all results to build leaderboard
    results = []
    for root, dirs, files in os.walk('results'):
        for f in files:
            if f.endswith('.json'):
                try:
                    with open(os.path.join(root, f)) as fh:
                        r = json.load(fh)
                    if r.get('status') == 'done':
                        results.append(r)
                except:
                    pass

    if not results:
        return "<p>No completed experiments yet</p>"

    results.sort(key=lambda r: r.get('val_loss', 999))
    top = results[:15]

    rows = ""
    best_loss = top[0]['val_loss'] if top else 999
    for i, r in enumerate(top):
        cls = 'best' if i == 0 else ''
        delta = r['val_loss'] - best_loss
        changes_str = json.dumps(r.get('changes', {})) if r.get('changes') else 'baseline'
        if len(changes_str) > 60:
            changes_str = changes_str[:57] + '...'
        rows += f"""<tr class="{cls}">
            <td>{i+1}</td><td>{r['exp_id']}</td>
            <td>{r['val_loss']:.4f}</td><td>{r['train_loss']:.4f}</td>
            <td>{'+' if delta >= 0 else ''}{delta:.4f}</td>
            <td>{r.get('training_time',0):.1f}s</td>
            <td>{changes_str}</td></tr>"""

    return f"""<table>
        <tr><th>#</th><th>Experiment</th><th>Val Loss</th><th>Train Loss</th>
        <th>vs Best</th><th>Time</th><th>Changes</th></tr>
        {rows}</table>"""


def get_queue_html():
    try:
        with open('optimization/queue.json') as f:
            queue = json.load(f)
    except:
        return "<p>No queue.json found</p>"

    if not queue:
        return "<p>Queue is empty</p>"

    rows = ""
    for exp in queue:
        s = exp['status']
        loss = f"{exp['val_loss']:.4f}" if 'val_loss' in exp else '-'
        t = f"{exp['training_time']:.1f}s" if 'training_time' in exp else '-'
        tokens = f"{exp.get('train_tokens',0):,}"
        rows += f"""<tr class="{s}">
            <td>{exp['exp_id']}</td><td class="{s}">{s}</td>
            <td>{loss}</td><td>{t}</td><td>{tokens}</td>
            <td>{exp.get('hypothesis','')}</td></tr>"""

    return f"""<table>
        <tr><th>ID</th><th>Status</th><th>Val Loss</th><th>Time</th><th>Tokens</th><th>Hypothesis</th></tr>
        {rows}</table>"""


@app.route('/')
def dashboard():
    os.chdir('/root/llm-research-kit')
    return render_template_string(TEMPLATE,
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        gpu_html=get_gpu_html(),
        status_html=get_status_html(),
        leaderboard_html=get_leaderboard_html(),
        queue_html=get_queue_html(),
        insights=read_file('optimization/insights.md', 'No insights yet'),
        plan=read_file('optimization/plan.md', 'No plan yet')[:2000],
    )


if __name__ == '__main__':
    os.chdir('/root/llm-research-kit')
    print("Dashboard starting on http://0.0.0.0:5000")
    print("Use SSH tunnel: ssh -L 5000:localhost:5000 user@server")
    app.run(host='0.0.0.0', port=5000, debug=False)
