"""
pipeline.py — Unified pair-trading pipeline
Runs all 6 notebook stages in order and saves outputs to a timestamped run folder.

Usage
-----
    python pipeline.py [options]

All parameters have sensible defaults matching the original notebooks.
Run with --help to see every option.
"""

import argparse
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# CLI — all notebook parameters exposed here
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pair-trading pipeline (dimension-reduction → pagerank → "
                    "cointegration → GMM regimes → minhash → backtest)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── paths ──────────────────────────────────────────────────────────────
    p.add_argument("--data-dir", default="data",
                   help="Root data directory (must contain raw/equity_returns.csv)")
    p.add_argument("--runs-dir", default="data/runs",
                   help="Parent directory for run folders")
    p.add_argument("--run-name", default=None,
                   help="Name for this run folder (default: timestamp)")

    # ── stage 1: dimension reduction ───────────────────────────────────────
    p.add_argument("--pca-components", type=int, default=10,
                   help="Number of PCA components to retain")

    # ── stage 2: pagerank ──────────────────────────────────────────────────
    p.add_argument("--corr-threshold", type=float, default=0.8,
                   help="Correlation threshold for the PageRank adjacency matrix")
    p.add_argument("--num-assets", type=int, default=25,
                   help="Number of top-ranked assets to keep")
    p.add_argument("--pagerank-beta", type=float, default=0.85,
                   help="PageRank damping factor")
    p.add_argument("--pagerank-tol", type=float, default=1e-6,
                   help="PageRank convergence tolerance")
    p.add_argument("--pagerank-max-iter", type=int, default=100,
                   help="PageRank maximum iterations")

    # ── stage 3: cointegration ─────────────────────────────────────────────
    p.add_argument("--coint-pvalue", type=float, default=0.05,
                   help="p-value threshold for cointegration test")

    # ── stage 4: GMM regimes ───────────────────────────────────────────────
    p.add_argument("--gmm-k-range", nargs="+", type=int, default=[2, 3, 4, 5],
                   help="Candidate numbers of GMM components to evaluate (BIC selects best)")
    p.add_argument("--gmm-entry-z", type=float, default=1.0,
                   help="Global z-score magnitude to enter a trade")
    p.add_argument("--gmm-exit-z", type=float, default=0.2,
                   help="Global z-score magnitude below which the position is exited")

    # ── stage 5: minhash ───────────────────────────────────────────────────
    p.add_argument("--minhash-window", type=int, default=20,
                   help="Look-back window size (days) for MinHash pattern matching")
    p.add_argument("--minhash-forward-days", type=int, default=10,
                   help="Forward look-ahead (days) used to compute historical returns")
    p.add_argument("--minhash-num-perm", type=int, default=128,
                   help="Number of MinHash permutations")
    p.add_argument("--minhash-threshold", type=float, default=0.5,
                   help="Jaccard similarity threshold for LSH queries")
    p.add_argument("--min-signals", type=int, default=200,
                   help="Minimum non-zero GMM signals required to keep a pair in stage 5")
    p.add_argument("--shingle-k", type=int, default=3,
                   help="Shingle length (k-gram size) for discretised spread symbols")

    # ── stage 6: backtest ──────────────────────────────────────────────────
    p.add_argument("--train-cutoff", default="2024-01-01",
                   help="Date (YYYY-MM-DD) from which the test period starts")
    p.add_argument("--baseline-entry-z", type=float, default=1.0,
                   help="Baseline strategy: z-score entry threshold")
    p.add_argument("--baseline-exit-z", type=float, default=0.3,
                   help="Baseline strategy: z-score exit threshold")

    return p.parse_args()


# ─────────────────────────────────────────────
# Helper: logging
# ─────────────────────────────────────────────

def log(stage, msg):
    print(f"[{stage}] {msg}", flush=True)


# ─────────────────────────────────────────────
# Stage 1: Dimension Reduction (PCA)
# ─────────────────────────────────────────────

def run_dimension_reduction(args, paths):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    log("01-dim-reduction", "Loading equity returns …")
    equity_returns = pd.read_csv(
        paths["raw"] / "equity_returns.csv", index_col=0, parse_dates=True
    )

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(equity_returns),
        index=equity_returns.index,
        columns=equity_returns.columns,
    )

    log("01-dim-reduction", f"Fitting PCA with {args.pca_components} components …")
    pca = PCA(n_components=args.pca_components)
    X_pca = pca.fit_transform(scaled)
    X_reconstructed = pca.inverse_transform(X_pca)

    var_explained = np.cumsum(pca.explained_variance_ratio_)[-1]
    log("01-dim-reduction",
        f"Variance explained by {args.pca_components} components: {var_explained:.3%}")

    residuals = equity_returns.values - X_reconstructed
    residuals_df = pd.DataFrame(
        residuals, index=equity_returns.index, columns=equity_returns.columns
    )

    out_path = paths["run_manip"] / "residuals.csv"
    residuals_df.to_csv(out_path)
    log("01-dim-reduction", f"Saved residuals → {out_path}")
    return residuals_df


# ─────────────────────────────────────────────
# Stage 2: PageRank
# ─────────────────────────────────────────────

def pagerank(mtx, beta=0.85, tol=1e-6, max_iter=100):
    import numpy as np

    A = mtx.to_numpy() if hasattr(mtx, "to_numpy") else mtx
    n = A.shape[0]
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = A / row_sums
    r = np.ones(n) / n
    for _ in range(max_iter):
        r_new = beta * P.T @ r + (1 - beta) / n
        if np.linalg.norm(r_new - r, 1) < tol:
            break
        r = r_new
    return r


def run_pagerank(args, paths, residuals_df):
    import pandas as pd
    import numpy as np

    log("02-pagerank", "Computing correlation matrix …")
    equity_corr = residuals_df.corr()

    log("02-pagerank",
        f"Building adjacency matrix (threshold={args.corr_threshold}) …")
    weight_matrix = equity_corr.copy()
    weight_matrix[equity_corr <= args.corr_threshold] = 0
    weight_matrix = weight_matrix ** 2
    weight_matrix = weight_matrix.mask(np.eye(len(weight_matrix), dtype=bool), 0)

    r = pagerank(
        weight_matrix,
        beta=args.pagerank_beta,
        tol=args.pagerank_tol,
        max_iter=args.pagerank_max_iter,
    )
    pagerank_df = pd.DataFrame(
        {"Asset": weight_matrix.index, "PageRank": r}
    ).sort_values("PageRank", ascending=False)

    top_assets = pagerank_df.head(args.num_assets)
    log("02-pagerank", f"Top {args.num_assets} assets selected")

    filtered_residuals = residuals_df[top_assets["Asset"]]
    out_path = paths["run_manip"] / "pr_filtered_residuals.csv"
    filtered_residuals.to_csv(out_path, index=False)
    log("02-pagerank", f"Saved filtered residuals → {out_path}")
    return filtered_residuals


# ─────────────────────────────────────────────
# Stage 3: Cointegration
# ─────────────────────────────────────────────

def run_cointegration(args, paths, filtered_residuals):
    import pandas as pd
    from statsmodels.tsa.stattools import coint
    from sklearn.linear_model import LinearRegression

    assets = filtered_residuals.columns
    log("03-cointegration",
        f"Testing cointegration for {len(assets)} assets "
        f"(p-value threshold={args.coint_pvalue}) …")

    pairs = []
    tickers = list(assets)
    total = len(tickers) * (len(tickers) - 1) // 2
    tested = 0
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            _, pvalue, _ = coint(filtered_residuals[a], filtered_residuals[b])
            if pvalue < args.coint_pvalue:
                model = LinearRegression().fit(
                    filtered_residuals[[b]], filtered_residuals[a]
                )
                beta = model.coef_[0]
                pairs.append((a, b, pvalue, beta))
            tested += 1
            if tested % 50 == 0:
                log("03-cointegration", f"  {tested}/{total} pairs tested …")

    pairs_df = pd.DataFrame(
        pairs, columns=["asset_a", "asset_b", "pvalue", "beta"]
    ).sort_values("pvalue")

    out_path = paths["run_manip"] / "cointegrated_pairs.csv"
    pairs_df.to_csv(out_path, index=False)
    log("03-cointegration",
        f"Found {len(pairs_df)} cointegrated pairs → {out_path}")
    return pairs_df


# ─────────────────────────────────────────────
# Stage 4: GMM Regimes
# ─────────────────────────────────────────────

def select_best_k(spread, k_range):
    from sklearn.mixture import GaussianMixture
    import numpy as np

    X = spread.values.reshape(-1, 1)
    bic_scores = []
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            random_state=42, n_init=5
        )
        gmm.fit(X)
        bic_scores.append((k, gmm.bic(X)))
    return min(bic_scores, key=lambda x: x[1])[0]


def fit_gmm(spread, n_components):
    from sklearn.mixture import GaussianMixture

    X = spread.values.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full",
        random_state=42, n_init=5
    )
    gmm.fit(X)
    return gmm


def run_gmm_regimes(args, paths, filtered_residuals, cointegrated_pairs):
    import pandas as pd
    import numpy as np

    spreads_dir = paths["run_spreads"]
    spreads_dir.mkdir(parents=True, exist_ok=True)

    log("04-gmm-regimes",
        f"Computing GMM regimes for {len(cointegrated_pairs)} pairs "
        f"(entry_z={args.gmm_entry_z}, exit_z={args.gmm_exit_z}) …")

    for _, row in cointegrated_pairs.iterrows():
        a, b, beta = row["asset_a"], row["asset_b"], row["beta"]
        spread = filtered_residuals[a] - beta * filtered_residuals[b]

        best_k = select_best_k(spread, args.gmm_k_range)
        gmm = fit_gmm(spread, best_k)

        labels = gmm.predict(spread.values.reshape(-1, 1))
        sorted_idx = np.argsort(gmm.means_.flatten())
        remap = {old: new for new, old in enumerate(sorted_idx)}
        sorted_labels = np.array([remap[l] for l in labels])

        z_scores = (spread - spread.mean()) / spread.std()
        sorted_stds = np.sqrt(gmm.covariances_.flatten())[sorted_idx]
        calm_regime = int(np.argmin(sorted_stds))

        signal = pd.Series(0, index=spread.index)
        signal[(z_scores < -args.gmm_entry_z) & (sorted_labels == calm_regime)] = 1
        signal[(z_scores > args.gmm_entry_z) & (sorted_labels == calm_regime)] = -1
        signal[z_scores.abs() < args.gmm_exit_z] = 0

        df = pd.DataFrame(
            {"spread": spread.values, "regime": sorted_labels,
             "z_score": z_scores.values, "signal": signal.values},
            index=spread.index,
        )
        df.to_csv(spreads_dir / f"{a}_{b}.csv")

    log("04-gmm-regimes",
        f"Spread CSVs saved to {spreads_dir}")


# ─────────────────────────────────────────────
# Stage 5: MinHash signal filtering
# ─────────────────────────────────────────────

def discretise(spread_window):
    mean = spread_window.mean()
    std = spread_window.std()
    z = (spread_window - mean) / std
    symbols = []
    for val in z:
        if val < -1.5:   symbols.append("--")
        elif val < -0.3: symbols.append("-")
        elif val < 0.3:  symbols.append("0")
        elif val < 1.5:  symbols.append("+")
        else:            symbols.append("++")
    return symbols


def shingle(symbols, k=3):
    return set("".join(symbols[i: i + k]) for i in range(len(symbols) - k + 1))


def make_minhash(shingles, num_perm=128):
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def build_lsh_index(spread, window_size, num_perm, threshold, shingle_k):
    from datasketch import MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    signatures = {}
    for i in range(len(spread) - window_size):
        window = spread.iloc[i: i + window_size]
        syms = discretise(window)
        shingles_set = shingle(syms, k=shingle_k)
        m = make_minhash(shingles_set, num_perm=num_perm)
        key = f"w_{i}"
        lsh.insert(key, m)
        signatures[key] = (m, i)
    return lsh, signatures


def compute_minhash_signals(spread, gmm_signals, lsh, signatures,
                            window_size, forward_days, num_perm, shingle_k):
    import numpy as np
    import pandas as pd

    minhash_signals = pd.Series(0.0, index=spread.index)
    signal_days = gmm_signals[gmm_signals != 0].index

    for date in signal_days:
        i = spread.index.get_loc(date)
        if i < window_size:
            continue
        window = spread.iloc[i - window_size: i]
        syms = discretise(window)
        shingles_set = shingle(syms, k=shingle_k)
        query_m = make_minhash(shingles_set, num_perm=num_perm)
        similar_keys = lsh.query(query_m)
        if not similar_keys:
            continue
        forward_returns = []
        for key in similar_keys:
            _, start_idx = signatures[key]
            end_idx = start_idx + window_size + forward_days
            if end_idx < len(spread) and start_idx + window_size <= i:
                fwd = (spread.iloc[start_idx + window_size + forward_days]
                       - spread.iloc[start_idx + window_size])
                forward_returns.append(fwd)
        if forward_returns:
            minhash_signals.loc[date] = np.mean(forward_returns)

    return minhash_signals


def combine_signals(gmm_signal, minhash_signal):
    if gmm_signal == 0:
        return 0
    if gmm_signal == 1 and minhash_signal > 0:
        return 1
    if gmm_signal == -1 and minhash_signal < 0:
        return -1
    return 0


def run_minhash(args, paths, cointegrated_pairs):
    import pandas as pd

    spreads_dir = paths["run_spreads"]

    log("05-minhash",
        f"Filtering pairs by min_signals={args.min_signals} …")
    active_pairs = []
    for _, row in cointegrated_pairs.iterrows():
        a, b = row["asset_a"], row["asset_b"]
        fp = spreads_dir / f"{a}_{b}.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        n_signals = (df["signal"] != 0).sum()
        if n_signals >= args.min_signals:
            active_pairs.append(row)
            log("05-minhash", f"  {a}/{b} — {n_signals} signals, keeping")
        else:
            log("05-minhash", f"  {a}/{b} — {n_signals} signals, dropping")

    active_pairs_df = pd.DataFrame(active_pairs)
    log("05-minhash",
        f"{len(active_pairs_df)} pairs kept from {len(cointegrated_pairs)}")

    log("05-minhash",
        f"Computing MinHash signals "
        f"(window={args.minhash_window}, forward_days={args.minhash_forward_days}, "
        f"threshold={args.minhash_threshold}) …")

    for _, row in active_pairs_df.iterrows():
        a, b = row["asset_a"], row["asset_b"]
        fp = spreads_dir / f"{a}_{b}.csv"
        df = pd.read_csv(fp, index_col=0, parse_dates=True, date_format="%Y-%m-%d")

        spread = df["spread"]
        lsh, signatures = build_lsh_index(
            spread, args.minhash_window,
            args.minhash_num_perm, args.minhash_threshold, args.shingle_k
        )
        minhash_sigs = compute_minhash_signals(
            spread, df["signal"], lsh, signatures,
            args.minhash_window, args.minhash_forward_days,
            args.minhash_num_perm, args.shingle_k
        )
        df["minhash_signal"] = minhash_sigs
        df["final_signal"] = df.apply(
            lambda r: combine_signals(r["signal"], r["minhash_signal"]), axis=1
        )
        df.to_csv(fp)
        log("05-minhash", f"  done: {a}/{b}")

    return active_pairs_df


# ─────────────────────────────────────────────
# Stage 6: Backtest
# ─────────────────────────────────────────────

def backtest_pair(df, cutoff, date_index):
    df = df.copy()
    df.index = date_index[: len(df)]
    df = df[df.index >= cutoff]

    trades = []
    position = 0
    entry_price = entry_date = None

    for date, row in df.iterrows():
        signal = row["final_signal"]
        spread = row["spread"]

        if position == 0:
            if signal != 0:
                position, entry_price, entry_date = signal, spread, date
        else:
            if signal == 0 or signal != position:
                trades.append({
                    "entry_date":   entry_date,
                    "exit_date":    date,
                    "direction":    position,
                    "entry_price":  entry_price,
                    "exit_price":   spread,
                    "pnl":          position * (spread - entry_price),
                    "holding_days": (date - entry_date).days,
                })
                if signal != 0:
                    position, entry_price, entry_date = signal, spread, date
                else:
                    position = 0
                    entry_price = entry_date = None

    import pandas as pd
    return pd.DataFrame(trades)


def backtest_baseline(df, cutoff, date_index, entry_z, exit_z):
    import pandas as pd

    df = df.copy()
    df.index = date_index[: len(df)]
    df = df[df.index >= cutoff].copy()
    df["final_signal"] = 0
    df.loc[df["z_score"] < -entry_z, "final_signal"] = 1
    df.loc[df["z_score"] > entry_z, "final_signal"] = -1
    df.loc[df["z_score"].abs() < exit_z, "final_signal"] = 0
    return backtest_pair(df, cutoff, date_index[: len(df)])


def compute_metrics(trades_df):
    import numpy as np

    if trades_df.empty:
        return {}
    pnl = trades_df["pnl"]
    cumulative  = pnl.cumsum()
    rolling_max = cumulative.cummax()
    return {
        "total_pnl":        round(pnl.sum(), 4),
        "n_trades":         len(pnl),
        "win_rate":         round((pnl > 0).mean(), 4),
        "avg_pnl":          round(pnl.mean(), 4),
        "avg_win":          round(pnl[pnl > 0].mean() if (pnl > 0).any() else 0, 4),
        "avg_loss":         round(pnl[pnl < 0].mean() if (pnl < 0).any() else 0, 4),
        "sharpe":           round((pnl.mean() / pnl.std() * (252 ** 0.5)) if pnl.std() > 0 else 0, 4),
        "max_drawdown":     round((cumulative - rolling_max).min(), 4),
        "avg_holding_days": round(trades_df["holding_days"].mean(), 1),
    }


def run_backtest(args, paths, cointegrated_pairs, active_pairs_df):
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    spreads_dir = paths["run_spreads"]
    results_dir = paths["run_results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # date index from residuals (to restore integer-indexed CSVs)
    residuals = pd.read_csv(
        paths["run_manip"] / "residuals.csv", index_col=0, parse_dates=True
    )
    date_index = residuals.index

    cutoff = args.train_cutoff
    log("06-backtest",
        f"Backtesting from {cutoff} onwards …")

    # ── strategy (GMM + MinHash) ───────────────────────────────────────────
    all_trades   = []
    summary_rows = []

    for _, row in cointegrated_pairs.iterrows():
        a, b     = row["asset_a"], row["asset_b"]
        fp       = spreads_dir / f"{a}_{b}.csv"
        if not fp.exists():
            log("06-backtest", f"  missing: {fp}")
            continue
        df = pd.read_csv(fp, index_col=0)
        if "final_signal" not in df.columns:
            log("06-backtest", f"  no final_signal: {a}/{b} — skipping")
            continue

        trades_df = backtest_pair(df, cutoff, date_index)
        if trades_df.empty:
            continue

        trades_df["asset_a"] = a
        trades_df["asset_b"] = b
        all_trades.append(trades_df)

        metrics            = compute_metrics(trades_df)
        metrics["asset_a"] = a
        metrics["asset_b"] = b
        summary_rows.append(metrics)

    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary_df    = pd.DataFrame(summary_rows).sort_values("total_pnl", ascending=False) \
        if summary_rows else pd.DataFrame()
    log("06-backtest",
        f"{len(summary_df)} pairs traded, {len(all_trades_df)} total trades")

    # ── baseline ──────────────────────────────────────────────────────────
    baseline_trades = []
    for _, row in cointegrated_pairs.iterrows():
        a, b = row["asset_a"], row["asset_b"]
        fp   = spreads_dir / f"{a}_{b}.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, index_col=0)
        trades_df = backtest_baseline(
            df, cutoff, date_index,
            args.baseline_entry_z, args.baseline_exit_z
        )
        if not trades_df.empty:
            trades_df["asset_a"] = a
            trades_df["asset_b"] = b
            baseline_trades.append(trades_df)

    baseline_df      = pd.concat(baseline_trades, ignore_index=True) if baseline_trades else pd.DataFrame()
    baseline_metrics = compute_metrics(baseline_df)

    # ── metrics comparison ────────────────────────────────────────────────
    portfolio_metrics = compute_metrics(all_trades_df)
    comparison = pd.DataFrame(
        {"strategy": portfolio_metrics, "baseline": baseline_metrics}
    )
    log("06-backtest", "\nMetrics comparison:\n" + comparison.to_string())

    # ── save CSVs ─────────────────────────────────────────────────────────
    if not all_trades_df.empty:
        all_trades_df.to_csv(results_dir / "strategy_trades.csv", index=False)
    if not baseline_df.empty:
        baseline_df.to_csv(results_dir / "baseline_trades.csv", index=False)
    if not summary_df.empty:
        summary_df.to_csv(results_dir / "pair_summary.csv", index=False)
    comparison.to_csv(results_dir / "metrics_comparison.csv")
    log("06-backtest", f"CSVs saved to {results_dir}")

    # ── final plot: strategy vs baseline ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    if not all_trades_df.empty:
        strategy_cum = all_trades_df.groupby("exit_date")["pnl"].sum().cumsum()
        ax.plot(strategy_cum.index, strategy_cum.values,
                label="Strategy (GMM + MinHash)")

    if not baseline_df.empty:
        baseline_cum = baseline_df.groupby("exit_date")["pnl"].sum().cumsum()
        ax.plot(baseline_cum.index, baseline_cum.values,
                label="Baseline (fixed threshold)", linestyle="--")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("Strategy vs Baseline — Cumulative PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL (spread units)")
    ax.legend()
    plt.tight_layout()

    plot_path = results_dir / "backtest_cumulative_pnl.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log("06-backtest", f"Plot saved → {plot_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def build_paths(args):
    data_dir  = Path(args.data_dir)
    runs_dir  = Path(args.runs_dir)
    run_name  = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = runs_dir / run_name

    paths = {
        "data":        data_dir,
        "raw":         data_dir / "raw",
        "runs":        runs_dir,
        "run":         run_dir,
        "run_manip":   run_dir / "manipulated",
        "run_spreads": run_dir / "manipulated" / "spreads",
        "run_results": run_dir / "results",
    }

    for p in (paths["run_manip"], paths["run_spreads"], paths["run_results"]):
        p.mkdir(parents=True, exist_ok=True)

    return paths, run_name


def save_config(args, paths):
    import json

    cfg = vars(args)
    cfg_path = paths["run"] / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    log("pipeline", f"Config saved → {cfg_path}")


def main():
    args = parse_args()
    paths, run_name = build_paths(args)

    print("=" * 60)
    print(f"  Pipeline run: {run_name}")
    print(f"  Run folder  : {paths['run']}")
    print("=" * 60)

    save_config(args, paths)

    # Stage 1
    residuals_df = run_dimension_reduction(args, paths)

    # Stage 2
    filtered_residuals = run_pagerank(args, paths, residuals_df)

    # Stage 3
    cointegrated_pairs = run_cointegration(args, paths, filtered_residuals)

    if cointegrated_pairs.empty:
        log("pipeline", "No cointegrated pairs found — exiting.")
        sys.exit(1)

    # Stage 4
    run_gmm_regimes(args, paths, filtered_residuals, cointegrated_pairs)

    # Stage 5
    active_pairs_df = run_minhash(args, paths, cointegrated_pairs)

    # Stage 6
    run_backtest(args, paths, cointegrated_pairs, active_pairs_df)

    print("=" * 60)
    print(f"  Pipeline complete.  Results in: {paths['run']}")
    print("=" * 60)


if __name__ == "__main__":
    main()