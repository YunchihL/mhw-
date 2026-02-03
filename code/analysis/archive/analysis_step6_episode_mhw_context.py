#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis_step6_episode_mhw_context.py

Step 6｜Episode-level MHW temporal context

Goal
----
Quantify the temporal isolation / clustering of MHW episodes by checking
whether months before and after each episode are non-MHW months.

Definitions
-----------
- Episode:
    Consecutive months with isMHW == 1 on the same grid,
    with length >= 2 months.
- pre_1:
    The month immediately before the episode.
- post_k:
    The k-th month after the episode end (k = 1..max_post).

This script DOES NOT use any model output.
It only uses the original isMHW labels.

Outputs
-------
1) episode_mhw_context_table.csv
   Episode-level table with isMHW flags before and after episodes
2) episode_mhw_context_summary.csv
   Summary statistics:
     - fraction of episodes with non-MHW pre/post months
     - fraction with continuous non-MHW windows after episodes
"""

import os
import argparse
import pandas as pd


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def make_time_index(df):
    """Create continuous time index within each grid."""
    df = df.sort_values(["grid_id", "year", "month"]).copy()
    df["time_idx"] = df.groupby("grid_id").cumcount()
    return df


def find_episodes(df, min_len=2):
    """
    Identify MHW episodes.

    Returns a list of dicts:
    {
        grid_id,
        start_idx,
        end_idx,
        length
    }
    """
    episodes = []

    for gid, g in df.groupby("grid_id"):
        g = g.sort_values("time_idx")

        in_run = False
        start = None
        prev_idx = None

        for _, r in g.iterrows():
            if r["isMHW"] == 1 and not in_run:
                in_run = True
                start = r["time_idx"]

            if in_run and (r["isMHW"] == 0):
                end = prev_idx
                length = end - start + 1
                if length >= min_len:
                    episodes.append(
                        dict(
                            grid_id=gid,
                            start_idx=start,
                            end_idx=end,
                            length=length,
                        )
                    )
                in_run = False
                start = None

            prev_idx = r["time_idx"]

        # handle run reaching the end
        if in_run:
            end = prev_idx
            length = end - start + 1
            if length >= min_len:
                episodes.append(
                    dict(
                        grid_id=gid,
                        start_idx=start,
                        end_idx=end,
                        length=length,
                    )
                )

    return episodes


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="data.csv with isMHW")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max_post", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    df = pd.read_csv(args.data)

    required_cols = ["grid_id", "year", "month", "isMHW"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"data.csv missing required column: {c}")

    # create time index
    df = make_time_index(df)

    # identify episodes
    episodes = find_episodes(df, min_len=2)

    print("============================================================")
    print("[STEP 6] Episode-level MHW temporal context")
    print(f"Total episodes found: {len(episodes)}")
    print("------------------------------------------------------------")

    # --------------------------------------------------------
    # Episode-level context table
    # --------------------------------------------------------
    rows = []

    for i, ep in enumerate(episodes):
        g = df[df["grid_id"] == ep["grid_id"]].set_index("time_idx")

        row = {
            "episode_id": i,
            "grid_id": ep["grid_id"],
            "length": ep["length"],
        }

        # pre_1
        pre_idx = ep["start_idx"] - 1
        row["pre_1_isMHW"] = g.loc[pre_idx, "isMHW"] if pre_idx in g.index else None

        # post_k
        for k in range(1, args.max_post + 1):
            idx = ep["end_idx"] + k
            row[f"post_{k}_isMHW"] = g.loc[idx, "isMHW"] if idx in g.index else None

        rows.append(row)

    ep_ctx = pd.DataFrame(rows)

    out_table = os.path.join(args.outdir, "episode_mhw_context_table.csv")
    ep_ctx.to_csv(out_table, index=False)

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------
    summary_rows = []

    # pre
    valid = ep_ctx["pre_1_isMHW"].notna()
    summary_rows.append(
        dict(
            phase="pre_1",
            n=int(valid.sum()),
            frac_nonMHW=float((ep_ctx.loc[valid, "pre_1_isMHW"] == 0).mean()),
        )
    )

    # post k
    for k in range(1, args.max_post + 1):
        col = f"post_{k}_isMHW"
        valid = ep_ctx[col].notna()
        summary_rows.append(
            dict(
                phase=f"post_{k}",
                n=int(valid.sum()),
                frac_nonMHW=float((ep_ctx.loc[valid, col] == 0).mean()),
            )
        )

    # continuous non-MHW windows
    for window in [(1, 3), (1, 6)]:
        k1, k2 = window
        cols = [f"post_{k}_isMHW" for k in range(k1, k2 + 1)]
        valid = ep_ctx[cols].notna().all(axis=1)
        cond = (ep_ctx.loc[valid, cols] == 0).all(axis=1)

        summary_rows.append(
            dict(
                phase=f"post_{k1}_{k2}_all_nonMHW",
                n=int(valid.sum()),
                frac_nonMHW=float(cond.mean()) if valid.sum() > 0 else float("nan"),
            )
        )

    summary = pd.DataFrame(summary_rows)

    out_summary = os.path.join(args.outdir, "episode_mhw_context_summary.csv")
    summary.to_csv(out_summary, index=False)

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------
    print(summary.to_string(index=False))
    print("------------------------------------------------------------")
    print("[OUT]")
    print(f" - {out_table}")
    print(f" - {out_summary}")
    print("============================================================")


if __name__ == "__main__":
    main()
