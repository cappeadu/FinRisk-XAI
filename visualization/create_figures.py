# visualizations/create_figures.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "sans-serif"


with Path("results/evaluation/quantitative_results.json").open("r") as f:
    results = json.load(f)


# ========================================
# Figure 1: Faithfulness Comparison
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

methods = ["LogReg\n+ LIME", "LogReg\n+ SHAP", "XGBoost\n+ LIME", "XGBoost\n+ SHAP"]
colors = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#95E1D3"]

# Sufficiency
sufficiency = [
    results["logistic_lime"]["faithfulness"]["sufficiency_mean"],
    results["logistic_shap"]["faithfulness"]["sufficiency_mean"],
    results["xgboost_lime"]["faithfulness"]["sufficiency_mean"],
    results["xgboost_shap"]["faithfulness"]["sufficiency_mean"],
]
sufficiency_std = [
    results["logistic_lime"]["faithfulness"]["sufficiency_std"],
    results["logistic_shap"]["faithfulness"]["sufficiency_std"],
    results["xgboost_lime"]["faithfulness"]["sufficiency_std"],
    results["xgboost_shap"]["faithfulness"]["sufficiency_std"],
]

axes[0].bar(
    methods, sufficiency, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2
)
# axes[0].errorbar(
#     range(len(methods)),
#     sufficiency,
#     yerr=sufficiency_std,
#     fmt="none",
#     ecolor="black",
#     capsize=5,
#     capthick=2,
# )
axes[0].set_title(
    "Sufficiency Score\n(Can top-10 features alone predict accurately?)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
axes[0].set_ylabel("Sufficiency Score", fontsize=12, fontweight="bold")
axes[0].set_ylim([0, 1.0])
axes[0].axhline(
    y=0.7,
    color="red",
    linestyle="--",
    alpha=0.5,
    linewidth=2,
    label="Good threshold (0.7)",
)
axes[0].legend(fontsize=10)
axes[0].grid(axis="y", alpha=0.3)

# Add value labels on bars
# for i, (val, std) in enumerate(zip(sufficiency, sufficiency_std)):
#     axes[0].text(
#         i, val + std + 0.03, f"{val:.2f}", ha="center", fontsize=10, fontweight="bold"
#     )

# Comprehensiveness
comprehensiveness = [
    results["logistic_lime"]["faithfulness"]["comprehensiveness_mean"],
    results["logistic_shap"]["faithfulness"]["comprehensiveness_mean"],
    results["xgboost_lime"]["faithfulness"]["comprehensiveness_mean"],
    results["xgboost_shap"]["faithfulness"]["comprehensiveness_mean"],
]
comprehensiveness_std = [
    results["logistic_lime"]["faithfulness"]["comprehensiveness_std"],
    results["logistic_shap"]["faithfulness"]["comprehensiveness_std"],
    results["xgboost_lime"]["faithfulness"]["comprehensiveness_std"],
    results["xgboost_shap"]["faithfulness"]["comprehensiveness_std"],
]

axes[1].bar(
    methods,
    comprehensiveness,
    color=colors,
    alpha=0.8,
    edgecolor="black",
    linewidth=1.2,
)
# axes[1].errorbar(
#     range(len(methods)),
#     comprehensiveness,
#     yerr=comprehensiveness_std,
#     fmt="none",
#     ecolor="black",
#     capsize=5,
#     capthick=2,
# )
axes[1].set_title(
    "Comprehensiveness Score\n(Does removing top-10 features change prediction?)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
axes[1].set_ylabel("Comprehensiveness Score", fontsize=12, fontweight="bold")
axes[1].set_ylim([0, 0.20])
axes[1].axhline(
    y=0.1,
    color="red",
    linestyle="--",
    alpha=0.5,
    linewidth=2,
    label="Good threshold (0.1)",
)
axes[1].legend(fontsize=10)
axes[1].grid(axis="y", alpha=0.3)

# Add value labels
# for i, (val, std) in enumerate(zip(comprehensiveness, comprehensiveness_std)):
#     axes[1].text(
#         i, val + std + 0.005, f"{val:.3f}", ha="center", fontsize=10, fontweight="bold"
#     )

plt.tight_layout()
plt.savefig(
    Path("results") / "figures/faithfulness_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
print("Saved: faithfulness_comparison.png")
plt.close()

# ========================================
# Figure 2: Efficiency Comparison (Log Scale)
# ========================================
fig, ax = plt.subplots(figsize=(12, 7))

all_methods = ["LogReg + LIME", "LogReg + SHAP", "XGBoost + LIME", "XGBoost + SHAP"]

times = [
    results["logistic_lime"]["efficiency"]["mean_time"],
    results["logistic_shap"]["efficiency"]["mean_time"],
    results["xgboost_lime"]["efficiency"]["mean_time"],
    results["xgboost_shap"]["efficiency"]["mean_time"],
]

colors_eff = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#95E1D3"]

bars = ax.barh(
    all_methods, times, color=colors_eff, alpha=0.8, edgecolor="black", linewidth=1.2
)
ax.set_xlabel("Mean Time (seconds, log scale)", fontsize=13, fontweight="bold")
ax.set_title(
    "Explanation Generation Time\n(Lower is better for real-time deployment)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xscale("log")
ax.axvline(
    x=0.1, color="green", linestyle="--", alpha=0.6, linewidth=2, label="Fast (<0.1s)"
)
ax.axvline(
    x=1.0, color="orange", linestyle="--", alpha=0.6, linewidth=2, label="Moderate (1s)"
)
ax.legend(fontsize=11, loc="lower right")
ax.grid(axis="x", alpha=0.3)

# Add value labels
for i, (bar, time) in enumerate(zip(bars, times)):
    ax.text(
        time + time * 0.1,
        bar.get_y() + bar.get_height() / 2,
        f"{time:.3f}s",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(
    Path("results") / "figures/efficiency_comparison.png", dpi=300, bbox_inches="tight"
)
print("Saved: efficiency_comparison.png")
plt.close()

# ========================================
# Figure 3: Sparsity Comparison
# ========================================
fig, ax = plt.subplots(figsize=(12, 7))

sparsity_methods = [
    "LogReg + LIME",
    "LogReg + SHAP",
    "XGBoost + LIME",
    "XGBoost + SHAP",
]

sparsity_values = [
    results["logistic_lime"]["sparsity"]["mean_significant_features"],
    results["logistic_shap"]["sparsity"]["mean_significant_features"],
    results["xgboost_lime"]["sparsity"]["mean_significant_features"],
    results["xgboost_shap"]["sparsity"]["mean_significant_features"],
]

colors_sparse = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#95E1D3"]

bars = ax.bar(
    range(len(sparsity_methods)),
    sparsity_values,
    color=colors_sparse,
    alpha=0.8,
    edgecolor="black",
    linewidth=1.2,
)
ax.set_xticks(range(len(sparsity_methods)))
ax.set_xticklabels(sparsity_methods, rotation=15, ha="right")
ax.set_ylabel("Mean Number of Significant Features", fontsize=13, fontweight="bold")
ax.set_title(
    "Explanation Sparsity\n(Fewer features = easier for users to process)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.axhline(
    y=10,
    color="orange",
    linestyle="--",
    alpha=0.6,
    linewidth=2,
    label="Ideal range (~10 features)",
)
ax.axhline(
    y=5,
    color="green",
    linestyle="--",
    alpha=0.6,
    linewidth=2,
    label="Very concise (<5 features)",
)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, sparsity_values)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 1.5,
        f"{val:.1f}",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(
    Path("results") / "figures/sparsity_comparison.png", dpi=300, bbox_inches="tight"
)
print("Saved: sparsity_comparison.png")
plt.close()


# ========================================
# Figure 4: Trade-off Analysis
# ========================================
fig, ax = plt.subplots(figsize=(12, 8))

# X-axis: Efficiency (inverse time - higher is better)
# Y-axis: Faithfulness (sufficiency)
# Size: Sparsity (inverse features - larger bubble = more concise)

x_values = [1 / t for t in times]
y_values = sufficiency
sizes = [1000 / s for s in sparsity_values]  # Inverse for size
labels = all_methods

scatter = ax.scatter(
    x_values,
    y_values,
    s=sizes,
    c=colors_eff,
    alpha=0.6,
    edgecolors="black",
    linewidth=2,
)

# Add labels
for i, label in enumerate(labels):
    ax.annotate(
        label,
        (x_values[i], y_values[i]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=colors_eff[i], alpha=0.3),
    )

ax.set_xlabel("Efficiency (1/time) - Higher is Faster", fontsize=13, fontweight="bold")
ax.set_ylabel(
    "Faithfulness (Sufficiency) - Higher is Better", fontsize=13, fontweight="bold"
)
ax.set_title(
    "XAI Trade-off Analysis\n(Bubble size = Sparsity, larger = more concise)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.grid(True, alpha=0.3)

# Add ideal region
ax.axhline(y=0.7, color="green", linestyle="--", alpha=0.3, linewidth=2)
ax.axvline(x=1, color="green", linestyle="--", alpha=0.3, linewidth=2)
ax.fill_between(
    [1, ax.get_xlim()[1]],
    0.7,
    1.0,
    alpha=0.1,
    color="green",
    label="Ideal region (fast + faithful)",
)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(
    Path("results") / "figures/tradeoff_analysis.png", dpi=300, bbox_inches="tight"
)
print("Saved: tradeoff_analysis.png")
plt.close()

# ========================================
# Figure 5: Summary Table
# ========================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("tight")
ax.axis("off")

summary_data = [
    [
        "Method",
        "Sufficiency ↑",
        "Comprehensiveness ↑",
        "Time (s) ↓",
        "Features ↓",
        "Overall Rank",
    ],
    [
        "LogReg + LIME",
        f"{sufficiency[0]:.3f}",
        f"{comprehensiveness[0]:.3f}",
        f"{times[0]:.3f}",
        f"{sparsity_values[0]:.1f}",
        "2nd",
    ],
    [
        "LogReg + SHAP",
        f"{sufficiency[1]:.3f}",
        f"{comprehensiveness[1]:.3f}",
        f"{times[1]:.3f}",
        f"{sparsity_values[1]:.1f}",
        "1st (fastest)",
    ],
    [
        "XGBoost + LIME",
        f"{sufficiency[2]:.3f}",
        f"{comprehensiveness[2]:.3f}",
        f"{times[2]:.3f}",
        f"{sparsity_values[2]:.1f}",
        "4th",
    ],
    [
        "XGBoost + SHAP",
        f"{sufficiency[3]:.3f}",
        f"{comprehensiveness[3]:.3f}",
        f"{times[3]:.3f}",
        f"{sparsity_values[3]:.1f}",
        "3rd",
    ],
]

table = ax.table(
    cellText=summary_data,
    cellLoc="center",
    loc="center",
    colWidths=[0.18, 0.14, 0.18, 0.12, 0.12, 0.15],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor("#4ECDC4")
    cell.set_text_props(weight="bold", color="white", fontsize=12)

# Style data rows with alternating colors
for i in range(1, 5):
    color = "#F0F0F0" if i % 2 == 0 else "white"
    for j in range(6):
        table[(i, j)].set_facecolor(color)

plt.title(
    "Summary of XAI Method Performance\n(↑ higher is better, ↓ lower is better)",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
plt.savefig(Path("results") / "figures/summary_table.png", dpi=300, bbox_inches="tight")
print("Saved: summary_table.png")
plt.close()

print("\nAll visualizations created successfully!")
print("Saved to: results/figures/")
