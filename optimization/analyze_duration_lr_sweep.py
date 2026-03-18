#!/usr/bin/env python3
"""Analyze the duration->LR sweep and generate summary plots/CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.rglob("*.json")):
        try:
            with path.open() as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if "val_loss" not in data or "train_seconds" not in data:
            continue
        changes = data.get("changes", {})
        muon_lr = changes.get("muon_lr")
        if muon_lr is None:
            continue
        rows.append({
            "path": str(path),
            "duration": float(data["train_seconds"]),
            "muon_lr": float(muon_lr),
            "seed": int(data.get("seed", 0)),
            "val_loss": float(data["val_loss"]),
            "steps": int(data.get("steps", 0)),
            "status": data.get("status", "done"),
            "changes": changes,
        })
    return rows


def group_rows(rows: list[dict]) -> dict[tuple[float, float], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["duration"], row["muon_lr"])].append(row)
    return grouped


def group_stats(rows: list[dict], noise_sigma: float) -> list[dict]:
    grouped = group_rows(rows)
    stats = []
    for (duration, muon_lr), items in sorted(grouped.items()):
        values = np.array([r["val_loss"] for r in items], dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        sem = float(std / math.sqrt(len(values))) if len(values) > 1 else 0.0
        uncertainty = sem if sem > 0 else max(noise_sigma / math.sqrt(max(len(values), 1)), 1e-3)
        stats.append({
            "duration": duration,
            "muon_lr": muon_lr,
            "mean": mean,
            "std": std,
            "sem": sem,
            "uncertainty": uncertainty,
            "n": len(values),
        })
    return stats


def weighted_quadratic_fit(log_lr: np.ndarray, y: np.ndarray, sigma: np.ndarray):
    weights = 1.0 / np.maximum(sigma, 1e-8) ** 2
    X = np.column_stack([log_lr ** 2, log_lr, np.ones_like(log_lr)])
    WX = X * weights[:, None]
    XtWX = X.T @ WX
    XtWy = X.T @ (weights * y)
    coeffs = np.linalg.solve(XtWX, XtWy)
    residuals = y - X @ coeffs
    # Since weights are inverse-variance estimates, use the unscaled covariance.
    cov = np.linalg.inv(XtWX)
    rss = float(np.sum(weights * residuals ** 2))
    return coeffs, cov, rss


def fit_duration_curve(stats: list[dict], duration: float):
    sub = [r for r in stats if r["duration"] == duration]
    if len(sub) < 3:
        return None
    log_lr = np.log(np.array([r["muon_lr"] for r in sub], dtype=float))
    y = np.array([r["mean"] for r in sub], dtype=float)
    sigma = np.array([r["uncertainty"] for r in sub], dtype=float)

    coeffs, cov, rss = weighted_quadratic_fit(log_lr, y, sigma)
    a, b, c = coeffs
    if a <= 0:
        return None

    log_lr_star = -b / (2.0 * a)
    lr_star = float(np.exp(log_lr_star))
    grad = np.array([b / (2.0 * a * a), -1.0 / (2.0 * a), 0.0], dtype=float)
    se_log_lr_star = float(np.sqrt(max(0.0, grad @ cov @ grad)))
    min_loss = float(c - (b * b) / (4.0 * a))
    lr_vals = np.array([r["muon_lr"] for r in sub], dtype=float)
    within_grid = float(lr_vals.min()) <= lr_star <= float(lr_vals.max())

    return {
        "duration": duration,
        "lr_star": lr_star,
        "log_lr_star": float(log_lr_star),
        "se_log_lr_star": se_log_lr_star,
        "min_loss": min_loss,
        "curvature": float(a),
        "rss": rss,
        "n": len(sub),
        "within_grid": within_grid,
        "coeffs": coeffs,
        "cov": cov,
        "sub": sub,
    }


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, sigma: np.ndarray):
    weights = 1.0 / np.maximum(sigma, 1e-8) ** 2
    X = np.column_stack([np.ones_like(x), x])
    WX = X * weights[:, None]
    XtWX = X.T @ WX
    XtWy = X.T @ (weights * y)
    beta = np.linalg.solve(XtWX, XtWy)
    cov = np.linalg.inv(XtWX)
    residuals = y - X @ beta
    rss = float(np.sum(weights * residuals ** 2))
    return beta, cov, rss


def normal_p_value(z: float) -> float:
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))))


def summarize(results: list[dict], outdir: Path, noise_sigma: float):
    stats = group_stats(results, noise_sigma)
    duration_fits = []
    for duration in sorted({r["duration"] for r in stats}):
        fit = fit_duration_curve(stats, duration)
        if fit is not None:
            duration_fits.append(fit)

    outdir.mkdir(parents=True, exist_ok=True)

    # Save raw grouped stats
    grouped_csv = outdir / "duration_lr_grouped.csv"
    with grouped_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["duration", "muon_lr", "mean", "std", "sem", "uncertainty", "n"],
        )
        writer.writeheader()
        for row in stats:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    # Save duration fit table
    fit_csv = outdir / "duration_lr_fits.csv"
    with fit_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["duration", "lr_star", "se_log_lr_star", "min_loss", "curvature", "n", "within_grid"],
        )
        writer.writeheader()
        for row in duration_fits:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    # Power law fit
    power_fit = None
    if len(duration_fits) >= 2:
        x = np.log(np.array([r["duration"] for r in duration_fits], dtype=float))
        y = np.log(np.array([r["lr_star"] for r in duration_fits], dtype=float))
        sigma = np.array([max(r["se_log_lr_star"], 1e-3) for r in duration_fits], dtype=float)
        beta_vec, cov, rss = weighted_linear_fit(x, y, sigma)
        alpha, beta = beta_vec
        se_beta = float(np.sqrt(cov[1, 1]))
        se_alpha = float(np.sqrt(cov[0, 0]))
        z = beta / se_beta if se_beta > 0 else float("nan")
        p_value = normal_p_value(z) if np.isfinite(z) else float("nan")
        ci_low = beta - 1.96 * se_beta
        ci_high = beta + 1.96 * se_beta
        power_fit = {
            "alpha": float(alpha),
            "beta": float(beta),
            "se_alpha": se_alpha,
            "se_beta": se_beta,
            "p_value": p_value,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "rss": float(rss),
        }

    # Write summary text
    summary_path = outdir / "summary.txt"
    with summary_path.open("w") as f:
        f.write(f"Noise sigma estimate: {noise_sigma:.4f}\n")
        f.write(f"Grouped points: {len(stats)}\n")
        f.write(f"Duration fits: {len(duration_fits)}\n")
        if power_fit is None:
            f.write("Power law fit: insufficient duration fits\n")
        else:
            f.write(
                "Power law: lr* = "
                f"{math.exp(power_fit['alpha']):.6f} * duration^{power_fit['beta']:.6f}\n"
            )
            f.write(
                f"beta = {power_fit['beta']:.6f} ± {power_fit['se_beta']:.6f}, "
                f"95% CI [{power_fit['ci_low']:.6f}, {power_fit['ci_high']:.6f}], "
                f"p = {power_fit['p_value']:.6f}\n"
            )

        if power_fit is not None and power_fit["ci_low"] <= 0.0 <= power_fit["ci_high"]:
            weights = np.array([1.0 / max(r["se_log_lr_star"], 1e-3) ** 2 for r in duration_fits], dtype=float)
            log_lr = np.array([r["log_lr_star"] for r in duration_fits], dtype=float)
            const_lr = float(np.exp(np.sum(weights * log_lr) / np.sum(weights)))
            f.write(f"Conclusion: duration-independent, use lr = {const_lr:.6f}\n")
        elif power_fit is not None:
            f.write(
                "Conclusion: duration-dependent, use "
                f"lr*(d) = {math.exp(power_fit['alpha']):.6f} * d^{power_fit['beta']:.6f}\n"
            )
        else:
            f.write("Conclusion: insufficient data for a power law conclusion\n")

    return stats, duration_fits, power_fit


def plot_duration_parabolas(stats: list[dict], duration_fits: list[dict], outdir: Path):
    durations = sorted({r["duration"] for r in stats})
    n = len(durations)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    fit_by_duration = {r["duration"]: r for r in duration_fits}

    for idx, duration in enumerate(durations):
        ax = axes[idx // cols][idx % cols]
        sub = [r for r in stats if r["duration"] == duration]
        fit = fit_by_duration.get(duration)
        x = np.log([r["muon_lr"] for r in sub])
        y = [r["mean"] for r in sub]
        yerr = [r["uncertainty"] for r in sub]
        ax.errorbar(x, y, yerr=yerr, fmt="o", color="#00d4ff", capsize=3)
        ax.set_title(f"{int(duration)}s")
        ax.set_xlabel("log(muon_lr)")
        ax.set_ylabel("mean val_loss")
        if fit is not None:
            coeffs = fit["coeffs"]
            grid = np.linspace(min(x) - 0.1, max(x) + 0.1, 200)
            curve = coeffs[0] * grid ** 2 + coeffs[1] * grid + coeffs[2]
            ax.plot(grid, curve, color="#00ff88")
            ax.axvline(fit["log_lr_star"], color="#ffaa00", linestyle="--", linewidth=1)
            ax.text(
                0.02, 0.98,
                f"lr*={fit['lr_star']:.4f}\ncurv={fit['curvature']:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
            )
        ax.grid(alpha=0.2)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    fig.savefig(outdir / "parabolas.png", dpi=180)
    plt.close(fig)


def plot_power_law(duration_fits: list[dict], power_fit: dict | None, outdir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    durations = np.array([r["duration"] for r in duration_fits], dtype=float)
    lr_star = np.array([r["lr_star"] for r in duration_fits], dtype=float)
    yerr = np.array([max(r["se_log_lr_star"], 1e-3) for r in duration_fits], dtype=float)
    ax.errorbar(durations, lr_star, yerr=lr_star * yerr, fmt="o", color="#00d4ff", capsize=3, label="lr*")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("optimal lr")
    ax.grid(alpha=0.2, which="both")

    if power_fit is not None:
        grid = np.logspace(np.log10(durations.min()), np.log10(durations.max()), 200)
        curve = np.exp(power_fit["alpha"]) * (grid ** power_fit["beta"])
        ax.plot(grid, curve, color="#00ff88", label=f"fit beta={power_fit['beta']:.3f}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(outdir / "power_law.png", dpi=180)
    plt.close(fig)


def plot_residuals(duration_fits: list[dict], power_fit: dict | None, outdir: Path):
    if power_fit is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    durations = np.array([r["duration"] for r in duration_fits], dtype=float)
    observed = np.log(np.array([r["lr_star"] for r in duration_fits], dtype=float))
    predicted = power_fit["alpha"] + power_fit["beta"] * np.log(durations)
    residuals = observed - predicted
    ax.axhline(0.0, color="white", linewidth=1)
    ax.scatter(durations, residuals, color="#ff6644")
    ax.set_xscale("log")
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("log(lr*) residual")
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(outdir / "residuals.png", dpi=180)
    plt.close(fig)


def plot_heatmap(stats: list[dict], outdir: Path):
    durations = sorted({r["duration"] for r in stats})
    lrs = sorted({r["muon_lr"] for r in stats})
    matrix = np.full((len(durations), len(lrs)), np.nan, dtype=float)
    for row in stats:
        i = durations.index(row["duration"])
        j = lrs.index(row["muon_lr"])
        matrix[i, j] = row["mean"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{lr:.3f}" for lr in lrs], rotation=45, ha="right")
    ax.set_yticks(range(len(durations)))
    ax.set_yticklabels([f"{int(d)}s" for d in durations])
    ax.set_xlabel("muon_lr")
    ax.set_ylabel("duration")
    fig.colorbar(im, ax=ax, label="mean val_loss")
    fig.tight_layout()
    fig.savefig(outdir / "heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/duration_lr_sweep"))
    parser.add_argument("--outdir", type=Path, default=Path("optimization/duration_lr_analysis"))
    parser.add_argument("--noise-duration", type=float, default=20.0)
    parser.add_argument("--noise-lr", type=float, default=0.012)
    args = parser.parse_args()

    rows = load_results(args.results_dir)
    if not rows:
        raise SystemExit(f"No result JSON files found in {args.results_dir}")

    noise_rows = [r for r in rows if r["duration"] == args.noise_duration and abs(r["muon_lr"] - args.noise_lr) < 1e-9]
    if len(noise_rows) >= 2:
        noise_sigma = float(np.std([r["val_loss"] for r in noise_rows], ddof=1))
    else:
        noise_sigma = 0.03

    stats, duration_fits, power_fit = summarize(rows, args.outdir, noise_sigma)

    plot_duration_parabolas(stats, duration_fits, args.outdir)
    plot_power_law(duration_fits, power_fit, args.outdir)
    plot_residuals(duration_fits, power_fit, args.outdir)
    plot_heatmap(stats, args.outdir)

    print(f"Saved analysis to {args.outdir}")
    print(f"Noise sigma estimate: {noise_sigma:.4f}")
    if power_fit is not None:
        print(
            f"beta = {power_fit['beta']:.4f} ± {power_fit['se_beta']:.4f}, "
            f"95% CI [{power_fit['ci_low']:.4f}, {power_fit['ci_high']:.4f}], "
            f"p = {power_fit['p_value']:.4f}"
        )


if __name__ == "__main__":
    main()
