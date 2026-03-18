#!/usr/bin/env python3
"""Simple dashboard for the active LR transfer research."""

from __future__ import annotations

from flask import Flask, Response
import html
import json
import os
import subprocess
import time


app = Flask(__name__)

ROOT = "/root/llm-research-kit"
STATUS_PATH = os.path.join(ROOT, "optimization", "status.json")
QUEUE_PATH = os.path.join(ROOT, "optimization", "queue_lr_transfer.json")
RESULTS_DIR = os.path.join(ROOT, "results", "batch_27_lr_transfer")

DOC_PATHS = {
    "goal": os.path.join(ROOT, "optimization", "goal.md"),
    "plan": os.path.join(ROOT, "optimization", "plan.md"),
    "insights": os.path.join(ROOT, "optimization", "insights.md"),
    "life_goals": os.path.join(ROOT, "my-life", "goals.md"),
}


def read_text(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""


def load_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def get_gpu_info():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "name": parts[0],
            "mem_used": float(parts[1]),
            "mem_total": float(parts[2]),
            "util": float(parts[3]),
            "temp": parts[4],
            "power": parts[5],
            "power_limit": parts[6],
        }
    except Exception:
        return None


def load_results() -> list[dict]:
    rows = []
    if not os.path.exists(RESULTS_DIR):
        return rows
    for name in sorted(os.listdir(RESULTS_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(RESULTS_DIR, name)
        row = load_json(path, None)
        if isinstance(row, dict) and row.get("status") == "done":
            rows.append(row)
    return rows


def duration_sort_key(value: str) -> float:
    return float(value.rstrip("s"))


def classify_duration(row: dict) -> str:
    return f"{int(float(row.get('train_seconds', 0)))}s"


def lr_value(row: dict) -> float | None:
    try:
        return float(row.get("changes", {}).get("muon_lr"))
    except Exception:
        return None


def build_transfer_data(results: list[dict]) -> dict[float, dict[str, dict]]:
    data: dict[float, dict[str, dict]] = {}
    for row in results:
        lr = lr_value(row)
        duration = classify_duration(row)
        if lr is None:
            continue
        data.setdefault(lr, {})[duration] = row
    return dict(sorted(data.items()))


def best_by_duration(results: list[dict]) -> dict[str, dict]:
    best = {}
    for row in results:
        duration = classify_duration(row)
        if duration not in best or row["val_loss"] < best[duration]["val_loss"]:
            best[duration] = row
    return best


def render_markdown_block(text: str) -> str:
    lines = text.splitlines()
    parts = []
    in_list = False
    for raw in lines:
        line = raw.strip()
        if not line:
            if in_list:
                parts.append("</ul>")
                in_list = False
            continue
        if line.startswith("### "):
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<h4>{html.escape(line[4:])}</h4>")
        elif line.startswith("## "):
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<h3>{html.escape(line[3:])}</h3>")
        elif line.startswith("# "):
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<h2>{html.escape(line[2:])}</h2>")
        elif line.startswith("- "):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{html.escape(line[2:])}</li>")
        elif line[0].isdigit() and ". " in line[:4]:
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{html.escape(line.split('. ', 1)[1])}</li>")
        else:
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<p>{html.escape(line)}</p>")
    if in_list:
        parts.append("</ul>")
    return "".join(parts)


def svg_line_chart(transfer: dict[float, dict[str, dict]]) -> str:
    durations = ["5s", "10s", "20s"]
    width = 680
    height = 260
    pad_l = 56
    pad_r = 20
    pad_t = 20
    pad_b = 40

    points = []
    for lr, items in transfer.items():
        for duration in durations:
            row = items.get(duration)
            if row:
                points.append(row["val_loss"])
    if not points:
        return "<div class='empty-plot'>No results yet</div>"

    y_min = min(points) - 0.02
    y_max = max(points) + 0.02

    def x_pos(i: int) -> float:
        return pad_l + i * ((width - pad_l - pad_r) / (len(durations) - 1))

    def y_pos(v: float) -> float:
        span = max(y_max - y_min, 1e-6)
        return pad_t + (y_max - v) * (height - pad_t - pad_b) / span

    palette = {
        0.006: "#ff6b6b",
        0.008: "#1f7a8c",
        0.012: "#f2c14e",
    }

    grid = []
    for i in range(5):
        v = y_min + i * (y_max - y_min) / 4
        y = y_pos(v)
        grid.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width-pad_r}' y2='{y:.1f}' class='grid' />")
        grid.append(f"<text x='{pad_l-8}' y='{y+4:.1f}' class='axis-label' text-anchor='end'>{v:.3f}</text>")

    labels = []
    for i, duration in enumerate(durations):
        x = x_pos(i)
        labels.append(f"<text x='{x:.1f}' y='{height-12}' class='axis-label' text-anchor='middle'>{duration}</text>")

    lines = []
    for lr, items in transfer.items():
        coords = []
        for i, duration in enumerate(durations):
            row = items.get(duration)
            if row:
                coords.append((x_pos(i), y_pos(row["val_loss"]), row["val_loss"]))
        if not coords:
            continue
        color = palette.get(lr, "#ffffff")
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords)
        lines.append(f"<polyline points='{poly}' fill='none' stroke='{color}' stroke-width='3' />")
        for x, y, v in coords:
            lines.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5' fill='{color}' />")
            lines.append(f"<text x='{x:.1f}' y='{y-10:.1f}' class='point-label' text-anchor='middle'>{v:.4f}</text>")

    legend = []
    lx = pad_l
    ly = 12
    for lr in transfer.keys():
        color = palette.get(lr, "#ffffff")
        legend.append(f"<circle cx='{lx}' cy='{ly}' r='5' fill='{color}' />")
        legend.append(f"<text x='{lx+10}' y='{ly+4}' class='legend-label'>lr={lr:.3f}</text>")
        lx += 92

    return (
        f"<svg viewBox='0 0 {width} {height}' class='chart'>"
        + "".join(grid)
        + "".join(labels)
        + "".join(lines)
        + "".join(legend)
        + "</svg>"
    )


def svg_gap_chart(transfer: dict[float, dict[str, dict]]) -> str:
    durations = ["5s", "10s", "20s"]
    width = 680
    height = 210
    pad_l = 56
    pad_r = 18
    pad_t = 24
    pad_b = 34

    grouped = []
    for duration in durations:
        vals = []
        for lr, items in transfer.items():
            row = items.get(duration)
            if row:
                vals.append((lr, row["val_loss"]))
        vals.sort(key=lambda x: x[1])
        if vals:
            grouped.append((duration, vals))

    if not grouped:
        return "<div class='empty-plot'>No results yet</div>"

    min_loss = min(v for _, vals in grouped for _, v in vals)
    max_gap = max(vals[-1][1] - min_loss for _, vals in grouped)
    max_gap = max(max_gap, 0.02)

    def y_pos(v: float) -> float:
        return pad_t + (max_gap - v) * (height - pad_t - pad_b) / max_gap

    palette = {
        0.006: "#ff6b6b",
        0.008: "#1f7a8c",
        0.012: "#f2c14e",
    }

    parts = []
    for i in range(5):
        gap = i * max_gap / 4
        y = y_pos(gap)
        parts.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width-pad_r}' y2='{y:.1f}' class='grid' />")
        parts.append(f"<text x='{pad_l-8}' y='{y+4:.1f}' class='axis-label' text-anchor='end'>+{gap:.3f}</text>")

    group_width = (width - pad_l - pad_r) / len(grouped)
    bar_width = 44
    for i, (duration, vals) in enumerate(grouped):
        gx = pad_l + i * group_width + 16
        parts.append(f"<text x='{gx + group_width/2 - 16:.1f}' y='{height-10}' class='axis-label' text-anchor='middle'>{duration}</text>")
        for j, (lr, loss) in enumerate(vals):
            gap = loss - min_loss
            x = gx + j * (bar_width + 10)
            y = y_pos(gap)
            h = height - pad_b - y
            color = palette.get(lr, "#ffffff")
            parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width}' height='{h:.1f}' rx='8' fill='{color}' />")
            parts.append(f"<text x='{x + bar_width/2:.1f}' y='{y-8:.1f}' class='point-label' text-anchor='middle'>{loss:.4f}</text>")
            parts.append(f"<text x='{x + bar_width/2:.1f}' y='{height-pad_b+16:.1f}' class='small-label' text-anchor='middle'>{lr:.3f}</text>")

    return f"<svg viewBox='0 0 {width} {height}' class='chart'>{''.join(parts)}</svg>"


def svg_rank_flow(transfer: dict[float, dict[str, dict]]) -> str:
    durations = ["5s", "10s", "20s"]
    width = 680
    height = 210
    cols = [110, 340, 570]
    rows = {1: 54, 2: 105, 3: 156}
    palette = {
        0.006: "#ff6b6b",
        0.008: "#1f7a8c",
        0.012: "#f2c14e",
    }

    ranks_by_duration = {}
    for duration in durations:
        vals = []
        for lr, items in transfer.items():
            row = items.get(duration)
            if row:
                vals.append((lr, row["val_loss"]))
        vals.sort(key=lambda x: x[1])
        ranks_by_duration[duration] = {lr: idx + 1 for idx, (lr, _) in enumerate(vals)}

    parts = []
    for i, duration in enumerate(durations):
        parts.append(f"<text x='{cols[i]}' y='24' class='stage-label' text-anchor='middle'>{duration}</text>")
        for rank in [1, 2, 3]:
            parts.append(f"<text x='{cols[i]-64}' y='{rows[rank]+4}' class='axis-label' text-anchor='end'>#{rank}</text>")

    for lr in transfer.keys():
        coords = []
        for i, duration in enumerate(durations):
            rank = ranks_by_duration.get(duration, {}).get(lr)
            if rank is not None:
                coords.append((cols[i], rows[rank], rank))
        if len(coords) < 2:
            continue
        color = palette.get(lr, "#ffffff")
        path = " ".join(f"{x},{y}" for x, y, _ in coords)
        parts.append(f"<polyline points='{path}' fill='none' stroke='{color}' stroke-width='4' stroke-linecap='round' />")
        for x, y, rank in coords:
            parts.append(f"<circle cx='{x}' cy='{y}' r='12' fill='{color}' />")
            parts.append(f"<text x='{x}' y='{y+4}' class='node-label' text-anchor='middle'>{rank}</text>")
        parts.append(f"<text x='{coords[-1][0] + 32}' y='{coords[-1][1] + 4}' class='legend-label'>lr={lr:.3f}</text>")

    return f"<svg viewBox='0 0 {width} {height}' class='chart'>{''.join(parts)}</svg>"


def status_summary(queue: list[dict], status: dict) -> tuple[str, str]:
    pending = len([q for q in queue if q.get("status") == "pending"])
    done = len([q for q in queue if q.get("status") == "done"])
    if status.get("current_exp"):
        return ("Running", f"{status['current_exp']} | {status.get('progress', '?')}")
    if pending:
        return ("Queued", f"{pending} pending | {done} done")
    if done:
        return ("Idle", f"Latest batch finished | {done} done")
    return ("Idle", "No active queue")


@app.route("/")
def dashboard():
    os.chdir(ROOT)
    now = time.strftime("%H:%M:%S")
    gpu = get_gpu_info()
    queue = load_json(QUEUE_PATH, [])
    status = load_json(STATUS_PATH, {})
    results = load_results()
    transfer = build_transfer_data(results)
    best = best_by_duration(results)
    state, state_detail = status_summary(queue, status)

    answer = "Waiting for results"
    answer_sub = "Run the transfer batch to answer the main question."
    if best:
        best_5 = lr_value(best.get("5s", {}))
        best_10 = lr_value(best.get("10s", {}))
        best_20 = lr_value(best.get("20s", {}))
        if best_20 is not None:
            if best_5 == best_20:
                answer = "5s matches the 20s winner"
            else:
                answer = "5s misses the exact 20s winner"
            if best_10 == best_20:
                answer_sub = f"10s recovers the 20s winner: lr={best_20:.3f}"
            else:
                answer_sub = f"20s winner is lr={best_20:.3f}; 10s still disagrees"

    total_train = sum(r.get("training_time", 0) for r in results)
    progress_count = f"{len(results)}/9"

    gpu_html = "<div class='muted'>GPU unavailable</div>"
    if gpu:
        mem_pct = 100 * gpu["mem_used"] / max(gpu["mem_total"], 1)
        gpu_html = f"""
        <div class="stat-line"><span>{html.escape(gpu['name'])}</span></div>
        <div class="meter-row"><span>VRAM</span><div class="meter"><div class="fill fill-a" style="width:{mem_pct:.1f}%"></div></div><b>{gpu['mem_used']:.0f}/{gpu['mem_total']:.0f} MiB</b></div>
        <div class="meter-row"><span>GPU</span><div class="meter"><div class="fill fill-b" style="width:{gpu['util']:.1f}%"></div></div><b>{gpu['util']:.0f}%</b></div>
        <div class="muted">{gpu['temp']} C | {gpu['power']}/{gpu['power_limit']} W</div>
        """

    page = f"""<!doctype html>
<html>
<head>
  <title>LR Transfer Dashboard</title>
  <meta http-equiv="refresh" content="5">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --bg: #f5efe3;
      --paper: #fffaf0;
      --ink: #1f2a30;
      --muted: #6b7477;
      --line: #d8ccba;
      --accent: #1f7a8c;
      --accent-2: #ff6b6b;
      --accent-3: #f2c14e;
      --ok: #2a9d5b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff7df 0, transparent 30%),
        linear-gradient(180deg, #f2ebde 0%, var(--bg) 100%);
    }}
    .shell {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px 18px 40px;
    }}
    .hero {{
      background: linear-gradient(135deg, #fffaf0 0%, #f7f0e5 100%);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 22px;
      box-shadow: 0 16px 50px rgba(52, 41, 24, 0.08);
      margin-bottom: 18px;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(34px, 6vw, 64px);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .sub {{
      font-size: 18px;
      color: var(--muted);
      max-width: 900px;
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: 1.3fr 0.7fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 8px 24px rgba(52, 41, 24, 0.05);
    }}
    .mini-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-top: 16px;
    }}
    .mini {{
      background: #fffdf7;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }}
    .mini .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .mini .value {{
      margin-top: 6px;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.04em;
    }}
    .answer {{
      font-size: 28px;
      line-height: 1.05;
      margin-bottom: 8px;
    }}
    .answer.good {{ color: var(--ok); }}
    .answer.warn {{ color: var(--accent-2); }}
    .muted {{ color: var(--muted); }}
    .section-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .section-title {{
      margin: 0 0 12px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
    }}
    .chart {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .grid {{ stroke: #e6dac7; stroke-width: 1; }}
    .axis-label {{ fill: #746d63; font-size: 12px; }}
    .point-label {{ fill: #3d3730; font-size: 11px; font-weight: 700; }}
    .legend-label {{ fill: #3d3730; font-size: 12px; }}
    .small-label {{ fill: #5e5750; font-size: 11px; }}
    .stage-label {{ fill: #2e363a; font-size: 14px; font-weight: 700; }}
    .node-label {{ fill: white; font-size: 12px; font-weight: 700; }}
    .meter-row {{
      display: grid;
      grid-template-columns: 48px 1fr auto;
      gap: 10px;
      align-items: center;
      margin: 10px 0;
    }}
    .meter {{
      height: 12px;
      background: #eadfce;
      border-radius: 999px;
      overflow: hidden;
    }}
    .fill {{ height: 100%; border-radius: 999px; }}
    .fill-a {{ background: linear-gradient(90deg, var(--accent), #5cb7b2); }}
    .fill-b {{ background: linear-gradient(90deg, var(--accent-3), var(--accent-2)); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid #eadfce;
    }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #e9f3f5;
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      margin-right: 8px;
    }}
    .doc-block h2, .doc-block h3, .doc-block h4 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    .doc-block p, .doc-block li {{
      color: #334046;
      line-height: 1.45;
      font-size: 15px;
    }}
    .doc-block ul {{
      margin: 0 0 10px 18px;
      padding: 0;
    }}
    @media (max-width: 900px) {{
      .hero-grid, .section-grid, .mini-grid {{ grid-template-columns: 1fr; }}
      .shell {{ padding: 14px 12px 28px; }}
      .hero {{ padding: 16px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Live Research Dashboard • {now}</div>
      <h1>Can 5-second LLM runs predict 20-second winners?</h1>
      <div class="sub">One page, one question: what the current LR transfer evidence says, what is running, what the next decision is, and why.</div>
      <div class="mini-grid">
        <div class="mini"><div class="label">Status</div><div class="value">{html.escape(state)}</div><div class="muted">{html.escape(state_detail)}</div></div>
        <div class="mini"><div class="label">Progress</div><div class="value">{progress_count}</div><div class="muted">fresh transfer batch</div></div>
        <div class="mini"><div class="label">Total Train</div><div class="value">{total_train:.0f}s</div><div class="muted">batch_27 only</div></div>
        <div class="mini"><div class="label">Current Best</div><div class="value">{lr_value(best.get('20s', {})) if best.get('20s') else '-'}</div><div class="muted">20s winner</div></div>
      </div>
      <div class="hero-grid">
        <div class="card">
          <div class="section-title">Current Answer</div>
          <div class="answer {'warn' if 'misses' in answer else 'good'}">{html.escape(answer)}</div>
          <p class="muted">{html.escape(answer_sub)}</p>
          <div style="margin-top:14px;">
            <span class="pill">Goal: LR transfer only</span>
            <span class="pill">Durations: 5s, 10s, 20s</span>
            <span class="pill">Seed: 42 primary</span>
          </div>
        </div>
        <div class="card">
          <div class="section-title">GPU</div>
          {gpu_html}
        </div>
      </div>
    </section>

    <section class="section-grid">
      <div class="card">
        <div class="section-title">Loss Across Durations</div>
        <div class="muted" style="margin-bottom:10px;">Lower is better. This is the cleanest picture of how each LR scales from 5s to 20s.</div>
        {svg_line_chart(transfer)}
      </div>
      <div class="card">
        <div class="section-title">Gap From Best</div>
        <div class="muted" style="margin-bottom:10px;">Each bar shows how far a config is from the best loss in this batch. Smaller is better.</div>
        {svg_gap_chart(transfer)}
      </div>
    </section>

    <section class="section-grid">
      <div class="card">
        <div class="section-title">Rank Flow</div>
        <div class="muted" style="margin-bottom:10px;">This answers the transfer question directly: which LR rises or falls as duration increases.</div>
        {svg_rank_flow(transfer)}
      </div>
      <div class="card">
        <div class="section-title">Fresh Transfer Table</div>
        <table>
          <tr><th>Duration</th><th>Winner</th><th>2nd</th><th>3rd</th></tr>
          <tr><td>5s</td><td>0.008 (6.7637)</td><td>0.012 (6.7772)</td><td>0.006 (6.8935)</td></tr>
          <tr><td>10s</td><td>0.006 (6.5143)</td><td>0.008 (6.5250)</td><td>0.012 (6.5508)</td></tr>
          <tr><td>20s</td><td>0.006 (6.2623)</td><td>0.008 (6.2695)</td><td>0.012 (6.2737)</td></tr>
        </table>
        <div style="margin-top:14px;" class="muted">
          Read:
          <ul>
            <li>5s favors 0.008</li>
            <li>10s and 20s both favor 0.006</li>
            <li>0.006 vs 0.008 at 20s is close enough for an optional tie-break multi-seed check</li>
          </ul>
        </div>
      </div>
    </section>

    <section class="section-grid">
      <div class="card doc-block">
        <div class="section-title">Current Progress And Thoughts</div>
        {render_markdown_block(read_text(DOC_PATHS["plan"]))}
        {render_markdown_block(read_text(DOC_PATHS["insights"]))}
      </div>
      <div class="card doc-block">
        <div class="section-title">Goals</div>
        {render_markdown_block(read_text(DOC_PATHS["goal"]))}
        {render_markdown_block(read_text(DOC_PATHS["life_goals"]))}
      </div>
    </section>
  </div>
</body>
</html>"""
    return Response(page, mimetype="text/html")


if __name__ == "__main__":
    os.chdir(ROOT)
    print("Dashboard on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
