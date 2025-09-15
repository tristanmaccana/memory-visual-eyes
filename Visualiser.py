import os
import sys
import math
import datetime as dt

import pandas as pd
import numpy as np

# --- Try seaborn, fallback to matplotlib-only if not installed ---
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False
import matplotlib.pyplot as plt

# ========= SETTINGS =========
csv_path = r"C:\Users\jmaccat\OneDrive - Johnson Controls\Translation\TMX Scoring Script\de-DE-techcomms\scored_csv_all-de.csv"
score_col = "cometkiwi_score"

# Optional: also export filtered subsets by single cutoffs (existing behavior)
export_filtered = True
cutoffs = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# Optional: bins (set to None to use an automatic rule)
bins = 40  # or None for Freedman–Diaconis rule

# ========= Range capture/export =========
# You can specify either percentages (e.g., 70–80) or decimals (0.70–0.80)
export_ranges = True
ranges = [(60, 85), (85, 100)]  # e.g., 70–80, 80–85, 85–100
shade_ranges_on_hist = True  # set False if you don't want shading on the plot
# ============================================

# --- Load ---
df = pd.read_csv(csv_path, encoding="utf-8-sig")
if score_col not in df.columns:
    raise KeyError(f"Column '{score_col}' not found in CSV. Available: {df.columns.tolist()}")

scores = pd.to_numeric(df[score_col], errors="coerce").dropna()
n = len(scores)
if n == 0:
    raise ValueError("No valid numeric scores found.")

# --- Compute stats & quantiles ---
quantiles = scores.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).to_dict()
stats = scores.describe()
print("\n===== COMETKiwi Score Summary =====")
print(f"Rows with scores: {n}")
print(stats.to_string())
print("\nQuantiles:")
for q, v in quantiles.items():
    print(f"  P{int(q*100):>2}: {v:.6f}")

# --- Choose bins if None using Freedman–Diaconis rule ---
def fd_bins(x):
    x = np.asarray(x)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return 40  # fallback
    h = 2 * iqr * (len(x) ** (-1/3))
    if h <= 0:
        return 40
    k = int(np.ceil((x.max() - x.min()) / h))
    return max(10, min(120, k))

if bins is None:
    bins = fd_bins(scores)

# ========= NEW: Helpers for ranges =========
def _normalize_ranges(ranges_list):
    """
    Accepts ranges as [(a,b), ...]. If any bound > 1.5, treat input as percentages and divide by 100.
    Ensures ranges sorted and non-overlapping (warns if overlapping).
    Returns (normalized_ranges_in_decimals, were_percent_bool).
    """
    if not ranges_list:
        return [], False

    # If any bound looks like percentage (>= 1.5), convert all to decimals
    were_percent = any(
        (b is not None and b > 1.5) or (a is not None and a > 1.5)
        for a, b in ranges_list
    )
    if were_percent:
        norm = [(
            (a / 100.0 if a is not None else None),
            (b / 100.0 if b is not None else None)
        ) for a, b in ranges_list]
    else:
        norm = ranges_list[:]

    # Replace None with -inf / +inf for sorting and masking logic
    norm_inf = [(
        (-np.inf if a is None else float(a)),
        ( np.inf if b is None else float(b))
    ) for a, b in norm]

    # Sort by lower bound
    norm_inf.sort(key=lambda t: t[0])

    # Basic overlap check (soft-warn)
    for i in range(1, len(norm_inf)):
        prev_hi = norm_inf[i-1][1]
        curr_lo = norm_inf[i][0]
        if curr_lo < prev_hi:
            print("⚠️  Warning: Ranges overlap; rows may appear in more than one range.")

    return norm_inf, were_percent

def _label_for_range(lo, hi, as_percent=True):
    def fmt(x):
        if x in (-np.inf, np.inf):
            return "−∞" if x == -np.inf else "+∞"
        return f"{x*100:.0f}%" if as_percent else f"{x:.2f}"
    return f"{fmt(lo)}–{fmt(hi)}"

# Pre-normalize ranges once (for plotting + exports)
norm_ranges, input_was_percent = _normalize_ranges(ranges)
# ============================================

# --- Plot ---
plt.figure(figsize=(9, 4.8), dpi=120)

if _HAS_SNS:
    sns.histplot(scores, bins=bins, kde=True, color="#4C78A8", edgecolor="white")
else:
    plt.hist(scores, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.9)

plt.title("COMETKiwi Score Distribution")
plt.xlabel("COMETKiwi score")
plt.ylabel("Count")

# Vertical lines for quantiles
vline_style_q = dict(color="#333333", linestyle="--", linewidth=1.0, alpha=0.9)
plt.axvline(quantiles[0.50], **vline_style_q, label=f"Median = {quantiles[0.50]:.3f}")
plt.axvline(quantiles[0.25], **vline_style_q)
plt.axvline(quantiles[0.75], **vline_style_q)
plt.axvline(quantiles[0.10], **vline_style_q)
plt.axvline(quantiles[0.90], **vline_style_q)
# Annotations for key quantiles
plt.text(quantiles[0.50], plt.ylim()[1]*0.9, "Median", rotation=90, va="top", ha="right", fontsize=8)
plt.text(quantiles[0.90], plt.ylim()[1]*0.9, "P90", rotation=90, va="top", ha="right", fontsize=8)
plt.text(quantiles[0.10], plt.ylim()[1]*0.9, "P10", rotation=90, va="top", ha="right", fontsize=8)

# Vertical lines for common cutoffs
vline_style_c = dict(color="#E45756", linestyle="-.", linewidth=1.2, alpha=0.9)
for c in cutoffs:
    if scores.min() <= c <= scores.max():
        plt.axvline(c, **vline_style_c)
        plt.text(c, plt.ylim()[1]*0.75, f"Cutoff {c:.2f}", rotation=90, va="top", ha="right", color="#E45756", fontsize=8)

# ========= NEW: Shade requested ranges on histogram =========
if shade_ranges_on_hist and norm_ranges:
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    y_top = plt.ylim()[1]
    for i, (lo, hi) in enumerate(norm_ranges):
        col = colors[i % len(colors)]
        plt.axvspan(lo, hi, color=col, alpha=0.08)
        label = _label_for_range(lo, hi, as_percent=True)  # annotate as %
        x_mid = lo if np.isinf(hi) else (hi if np.isinf(lo) else (lo + hi) / 2)
        x_mid = max(min(x_mid, scores.max()), scores.min())
        plt.text(x_mid, y_top * 0.98, label, ha="center", va="top", fontsize=8, color=col)
# ============================================================

plt.tight_layout()

# --- Save PNG next to CSV ---
base, _ = os.path.splitext(csv_path)
ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
png_path = f"{base}_score_hist_{ts}.png"
plt.savefig(png_path, bbox_inches="tight")
print("\n📈 Saved histogram to:", png_path)

# --- Optional: export filtered subsets by single cutoffs (existing behavior) ---
if export_filtered:
    for c in cutoffs:
        out_path = f"{base}_ge_{str(c).replace('.', '')}.csv"
        keep = df[pd.to_numeric(df[score_col], errors="coerce") >= c]
        keep.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"💾 ≥ {c:.2f}: kept {len(keep):4d}/{len(df):4d} rows  -> {out_path}")

# ========= NEW: export by explicit ranges & add 'score_range' column =========
if export_ranges and norm_ranges:
    vals = pd.to_numeric(df[score_col], errors="coerce").to_numpy()
    labels = []
    masks = []

    # Build masks for each range (left-inclusive, right-exclusive; last right-inclusive)
    for i, (lo, hi) in enumerate(norm_ranges):
        if i < len(norm_ranges) - 1:
            mask = (vals >= lo) & (vals < hi)
        else:
            mask = (vals >= lo) & (vals <= hi + 1e-12)  # include last upper bound
        masks.append(mask)
        labels.append(_label_for_range(lo, hi, as_percent=True))

    # Assign a 'score_range' label per row (rows outside ranges get NaN)
    score_range = np.array([np.nan] * len(vals), dtype=object)
    for lab, m in zip(labels, masks):
        score_range[m] = lab
    df["score_range"] = score_range
    # Export one CSV per range
    for lab, m in zip(labels, masks):
        sub = df[m]
        # Clean label for filename (e.g., "70%-80%" -> "70-80")
        fname_label = lab.replace("%", "").replace("−∞", "min").replace("+∞", "max").replace("–", "-")
        out_path = f"{base}_range_{fname_label}.csv"
        sub.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"💾 {lab}: kept {len(sub):4d}/{len(df):4d} rows -> {out_path}")

    # Also print a tidy summary
    print("\n===== Range counts =====")
    for lab, m in zip(labels, masks):
        print(f"{lab:>12}: {int(m.sum()):6d}")

    # Save the main CSV with the new 'score_range' annotation
    annotated_path = f"{base}_with_score_range_{ts}.csv"
    df.to_csv(annotated_path, index=False, encoding="utf-8-sig")
    print("💾 Added 'score_range' column and saved to:", annotated_path)
# =============================================================================
