#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    "analysis/results/grid_mean_impact_cf_minus_factual.csv"
)

# 只看负贡献
df = df[df["mean_impact_year"] < 0].copy()
df["abs_loss"] = -df["mean_impact_year"]

df = df.sort_values("abs_loss", ascending=False)
df["cum_share"] = df["abs_loss"].cumsum() / df["abs_loss"].sum()

plt.figure(figsize=(6,4))
plt.plot(np.arange(1, len(df)+1), df["cum_share"], lw=2)
plt.xlabel("Grid rank (by mean loss)")
plt.ylabel("Cumulative share of total loss")
plt.title("Spatial concentration of MHW-associated GPP loss")
plt.ylim(0,1)

plt.tight_layout()
plt.savefig("analysis/results/fig_step2_spatial_concentration.png", dpi=200)
plt.close()

print("[OK] fig_step2_spatial_concentration.png")
