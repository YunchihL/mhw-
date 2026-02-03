import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ======================================================
# paths
# ======================================================
IN_PATH = "results/step5q/step5q_gating_month_lat.csv"
OUT_DIR = "results/step5q/figs"
os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# load
# ======================================================
df = pd.read_csv(IN_PATH)

# ------------------------------------------------------
# helper: pivot for heatmap
# ------------------------------------------------------
def pivot_for_heatmap(df, value_col):
    piv = df.pivot_table(
        index="lat_band",
        columns="month",
        values=value_col
    )
    piv = piv.reindex(sorted(piv.columns), axis=1)
    return piv

# ======================================================
# 1) Facilitation ratio heatmap
#    sharp contrast at 0.5, continuous on both sides
# ======================================================
fac_piv = pivot_for_heatmap(df, "fac_ratio")

vmin = np.nanmin(fac_piv.values)
vmax = np.nanmax(fac_piv.values)

# --- build two continuous colormaps ---
cmap_low  = plt.cm.Blues(np.linspace(0.35, 1.0, 128))  # < 0.5
cmap_high = plt.cm.Reds(np.linspace(0.35, 1.0, 128))   # > 0.5

cmap_combined = mcolors.ListedColormap(
    np.vstack((cmap_low, cmap_high))
)

# --- force a hard color break at 0.5 ---
N = cmap_combined.N
bounds = np.linspace(vmin, vmax, N + 1)

# index where 0.5 should sit
mid_idx = int((0.5 - vmin) / (vmax - vmin) * N)
mid_idx = np.clip(mid_idx, 1, N - 1)

bounds[mid_idx] = 0.5
norm = mcolors.BoundaryNorm(bounds, N)

plt.figure(figsize=(10, 5))
im = plt.imshow(
    fac_piv,
    aspect="auto",
    cmap=cmap_combined,
    norm=norm
)

cbar = plt.colorbar(im)
cbar.set_label("Facilitation ratio (ΔGPP < 0)")
cbar.ax.axhline(
    (0.5 - vmin) / (vmax - vmin),
    color="black",
    linewidth=1.5
)

plt.xticks(range(len(fac_piv.columns)), fac_piv.columns)
plt.yticks(range(len(fac_piv.index)), fac_piv.index)
plt.xlabel("Month")
plt.ylabel("Latitude band")
plt.title(
    "Facilitation dominance of MHW-induced GPP responses\n"
    "(sharp contrast at 0.5)"
)

plt.tight_layout()
plt.savefig(
    f"{OUT_DIR}/heatmap_fac_ratio_sharp_center0p5.png",
    dpi=300
)
plt.close()

# ======================================================
# 2) NDVI gating heatmap (fac - sup, median)
# ======================================================
ndvi_piv = pivot_for_heatmap(df, "NDVI_avg_delta_med")
lim = np.nanmax(np.abs(ndvi_piv.values))

plt.figure(figsize=(10, 5))
im = plt.imshow(
    ndvi_piv,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-lim,
    vmax=lim
)
plt.colorbar(im, label="Δ NDVI (fac − sup, median)")
plt.xticks(range(len(ndvi_piv.columns)), ndvi_piv.columns)
plt.yticks(range(len(ndvi_piv.index)), ndvi_piv.index)
plt.xlabel("Month")
plt.ylabel("Latitude band")
plt.title("Background canopy activity gating (NDVI)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/heatmap_ndvi_gating.png", dpi=300)
plt.close()

# ======================================================
# 3) VPD gating heatmap (fac - sup, median)
# ======================================================
vpd_piv = pivot_for_heatmap(df, "vpd_pa_delta_med")
lim = np.nanmax(np.abs(vpd_piv.values))

plt.figure(figsize=(10, 5))
im = plt.imshow(
    vpd_piv,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-lim,
    vmax=lim
)
plt.colorbar(im, label="Δ VPD (fac − sup, median)")
plt.xticks(range(len(vpd_piv.columns)), vpd_piv.columns)
plt.yticks(range(len(vpd_piv.index)), vpd_piv.index)
plt.xlabel("Month")
plt.ylabel("Latitude band")
plt.title("Atmospheric drought gating (VPD)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/heatmap_vpd_gating.png", dpi=300)
plt.close()

print("=" * 80)
print("Heatmaps generated:")
print(f" - {OUT_DIR}/heatmap_fac_ratio_sharp_center0p5.png")
print(f" - {OUT_DIR}/heatmap_ndVI_gating.png")
print(f" - {OUT_DIR}/heatmap_vpd_gating.png")
print("=" * 80)
