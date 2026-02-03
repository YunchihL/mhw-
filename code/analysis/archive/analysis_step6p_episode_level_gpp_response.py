# ============================================================
# STEP 6′ (Step 6p): Episode-level GPP response to MHW
#   - Episode defined as consecutive isMHW==1 with length>=2
#   - Episode-level aggregation (mean delta within episode)
#   - Post analysis:
#       A) post_k (k=1..6) with n_episodes reported
#       B) combined post window: post_1_3 (mean)
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd

# ----------------------------
# helpers
# ----------------------------
def make_time_index(df):
    df = df.sort_values(["grid_id", "year", "month"]).copy()
    df["time_idx"] = df.groupby("grid_id").cumcount()
    return df

def find_episodes(df, min_len=2):
    """
    Return list of episodes:
    each episode = dict(grid_id, start_idx, end_idx, length)
    """
    episodes = []
    for gid, g in df.groupby("grid_id"):
        g = g.sort_values("time_idx")
        in_run = False
        start = None
        for _, r in g.iterrows():
            if r["isMHW"] == 1 and not in_run:
                in_run = True
                start = r["time_idx"]
            if (r["isMHW"] == 0 or _ == g.index[-1]) and in_run:
                # end previous run
                end = r["time_idx"] if r["isMHW"] == 1 and _ == g.index[-1] else prev_idx
                length = end - start + 1
                if length >= min_len:
                    episodes.append(
                        dict(grid_id=gid, start_idx=start, end_idx=end, length=length)
                    )
                in_run = False
                start = None
            prev_idx = r["time_idx"]
    return episodes

def summarize(series):
    return dict(
        n=len(series),
        mean=float(np.nanmean(series)),
        median=float(np.nanmedian(series)),
        p25=float(np.nanpercentile(series, 25)),
        p75=float(np.nanpercentile(series, 75)),
    )

# ----------------------------
# main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factual", required=True)
    parser.add_argument("--counterfactual", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max_post", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load
    fac = pd.read_csv(args.factual)
    cf  = pd.read_csv(args.counterfactual)
    dat = pd.read_csv(args.data)

    # merge (month-level)
    key = ["grid_id", "year", "month"]
    df = fac.merge(cf[key + ["gpp_pred"]], on=key, suffixes=("_factual", "_cf"))
    df = df.merge(dat[key + ["isMHW"]], on=key)

    # delta
    df["delta_gpp"] = df["gpp_pred_cf"] - df["gpp_pred_factual"]

    # time index
    df = make_time_index(df)

    # episodes
    episodes = find_episodes(df, min_len=2)
    print("============================================================")
    print("[STEP 6′] Episode-based GPP response (episode-level)")
    print(f"Total episodes found: {len(episodes)}")
    print("------------------------------------------------------------")

    rows = []
    for i, ep in enumerate(episodes):
        g = df[df["grid_id"] == ep["grid_id"]].set_index("time_idx")

        during = g.loc[ep["start_idx"]:ep["end_idx"], "delta_gpp"]
        row = {
            "episode_id": i,
            "grid_id": ep["grid_id"],
            "length": ep["length"],
            "during_mean": during.mean(),
        }

        # pre
        if ep["start_idx"] - 1 in g.index:
            row["pre_1"] = g.loc[ep["start_idx"] - 1, "delta_gpp"]
        else:
            row["pre_1"] = np.nan

        # post k
        post_vals = []
        for k in range(1, args.max_post + 1):
            idx = ep["end_idx"] + k
            if idx in g.index:
                v = g.loc[idx, "delta_gpp"]
                row[f"post_{k}"] = v
                post_vals.append(v)
            else:
                row[f"post_{k}"] = np.nan

        # combined post window (1-3)
        post_1_3 = [row.get("post_1"), row.get("post_2"), row.get("post_3")]
        post_1_3 = [v for v in post_1_3 if not pd.isna(v)]
        row["post_mean_1_3"] = np.mean(post_1_3) if len(post_1_3) > 0 else np.nan

        rows.append(row)

    ep_df = pd.DataFrame(rows)
    ep_df.to_csv(os.path.join(args.outdir, "step6p_episode_level_table.csv"), index=False)

    # summaries
    summary = {}

    summary["during"] = summarize(ep_df["during_mean"])

    for k in range(1, args.max_post + 1):
        col = f"post_{k}"
        summary[col] = summarize(ep_df[col].dropna())

    summary["post_mean_1_3"] = summarize(ep_df["post_mean_1_3"].dropna())

    summary_df = (
        pd.DataFrame(summary)
        .T.reset_index()
        .rename(columns={"index": "phase"})
    )

    summary_df.to_csv(
        os.path.join(args.outdir, "step6p_episode_phase_summary.csv"),
        index=False,
    )

    print(summary_df)
    print("------------------------------------------------------------")
    print("[OUT]")
    print(f" - {args.outdir}/step6p_episode_level_table.csv")
    print(f" - {args.outdir}/step6p_episode_phase_summary.csv")
    print("============================================================")


if __name__ == "__main__":
    main()
