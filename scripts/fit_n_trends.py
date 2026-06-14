from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


DEFAULT_METRICS = [
    "mean_chosen_flatness",
    "mean_selection_gain",
    "distinct_1",
    "distinct_2",
    "distinct_3",
    "repetition_4",
    "mean_runtime_seconds",
    "mean_tokens_per_second",
]

N_TRIAL_METHODS = {"random", "minflat", "maxflat"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, help="CSV from scripts/build_results_table.py")
    parser.add_argument("--output", default="results/summary/n_trend_models.csv")
    parser.add_argument("--figure-dir", default="results/figures/n_trends")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    return parser.parse_args()


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def aic(y: np.ndarray, yhat: np.ndarray, k: int) -> float:
    n = len(y)
    if n == 0:
        return math.inf
    rss = float(np.sum((y - yhat) ** 2))
    rss = max(rss, 1e-12)
    return n * math.log(rss / n) + 2 * k


def fit_linear_model(x: np.ndarray, y: np.ndarray, transform: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    xt = transform(x)
    coeff = np.polyfit(xt, y, deg=1)
    yhat = coeff[0] * xt + coeff[1]
    return coeff, yhat


def power_model(n: np.ndarray, asymptote: float, scale: float, alpha: float) -> np.ndarray:
    return asymptote + scale * np.power(n, -alpha)


def fit_models(x: np.ndarray, y: np.ndarray) -> list[dict]:
    rows: list[dict] = []

    model_specs = [
        ("linear_N", lambda n: n, "slope", 2),
        ("linear_logN", np.log, "slope_logN", 2),
        ("linear_invN", lambda n: 1.0 / n, "slope_invN", 2),
    ]

    for name, transform, slope_name, k in model_specs:
        try:
            coeff, yhat = fit_linear_model(x, y, transform)
            rows.append(
                {
                    "trend_model": name,
                    slope_name: float(coeff[0]),
                    "intercept": float(coeff[1]),
                    "r2": r2_score(y, yhat),
                    "aic": aic(y, yhat, k),
                    "num_points": len(y),
                }
            )
        except Exception as exc:
            rows.append({"trend_model": name, "error": str(exc), "num_points": len(y)})

    if len(x) >= 4:
        try:
            bounds = ([-np.inf, -np.inf, 0.0], [np.inf, np.inf, 10.0])
            params, _ = curve_fit(power_model, x, y, p0=[float(y[-1]), float(y[0] - y[-1]), 1.0], bounds=bounds, maxfev=20000)
            yhat = power_model(x, *params)
            rows.append(
                {
                    "trend_model": "power_asymptote_plus_scale_N_minus_alpha",
                    "asymptote": float(params[0]),
                    "scale": float(params[1]),
                    "alpha": float(params[2]),
                    "r2": r2_score(y, yhat),
                    "aic": aic(y, yhat, 3),
                    "num_points": len(y),
                }
            )
        except Exception as exc:
            rows.append({"trend_model": "power_asymptote_plus_scale_N_minus_alpha", "error": str(exc), "num_points": len(y)})

    return rows


def best_model(rows: list[dict]) -> dict | None:
    valid = [row for row in rows if "aic" in row and not pd.isna(row.get("aic"))]
    if not valid:
        return None
    return min(valid, key=lambda row: row["aic"])


def plot_metric(part: pd.DataFrame, method: str, metric: str, figure_dir: Path) -> None:
    clean = part[["n_trials", metric]].dropna().sort_values("n_trials")
    if len(clean) < 2:
        return
    x = clean["n_trials"].to_numpy(dtype=float)
    y = clean[metric].to_numpy(dtype=float)

    plt.plot(x, y, marker="o", label="observed")

    model_rows = fit_models(x, y)
    best = best_model(model_rows)
    if best is not None:
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        name = best["trend_model"]
        if name == "linear_N":
            ys = best["slope"] * xs + best["intercept"]
        elif name == "linear_logN":
            ys = best["slope_logN"] * np.log(xs) + best["intercept"]
        elif name == "linear_invN":
            ys = best["slope_invN"] * (1.0 / xs) + best["intercept"]
        elif name == "power_asymptote_plus_scale_N_minus_alpha":
            ys = power_model(xs, best["asymptote"], best["scale"], best["alpha"])
        else:
            ys = None
        if ys is not None:
            plt.plot(xs, ys, linestyle="--", label=f"best: {name}")

    plt.xlabel("N trials")
    plt.ylabel(metric)
    plt.title(f"{method}: {metric} vs N")
    plt.legend()
    plt.tight_layout()
    safe_metric = metric.replace("/", "_")
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_dir / f"trend_{method}_{safe_metric}.png")
    plt.close()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.table)
    figure_dir = Path(args.figure_dir)
    rows: list[dict] = []

    if "method" not in df.columns or "n_trials" not in df.columns:
        raise SystemExit("table must contain method and n_trials columns")

    for method, part in df[df["method"].isin(N_TRIAL_METHODS)].groupby("method"):
        part = part[part["n_trials"].notna()].copy()
        if part.empty:
            continue
        part["n_trials"] = pd.to_numeric(part["n_trials"], errors="coerce")
        for metric in args.metrics:
            if metric not in part.columns:
                continue
            clean = part[["n_trials", metric]].dropna().sort_values("n_trials")
            if len(clean) < 2:
                continue
            x = clean["n_trials"].to_numpy(dtype=float)
            y = clean[metric].to_numpy(dtype=float)
            model_rows = fit_models(x, y)
            for row in model_rows:
                row.update({"method": method, "metric": metric})
                rows.append(row)
            plot_metric(part, method, metric, figure_dir)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {output} with {len(rows)} model rows")
    print(f"wrote trend figures to {figure_dir}")


if __name__ == "__main__":
    main()
