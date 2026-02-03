#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "analysis/results/annual_global_mhw_impact_cf_minus_factual.csv"
)

plt.figure(figsize=(8,4))
plt.plot(df["year"], df["impact_year"], marker="o")
plt.axhline(0, color="k", linestyle="--", lw=1)

plt.xlabel("Year")
plt.ylabel("ΔGPP (cf − factual)")
plt.title("Global annual MHW-associated GPP impact")

plt.tight_layout()
plt.savefig("analysis/results/fig_step1_global_annual_impact.png", dpi=200)
plt.close()

print("[OK] fig_step1_global_annual_impact.png")
