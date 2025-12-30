"""
run_all_evaluations.py
======================

Unified runner for evaluation steps A1–A4.

Usage logic:
- If no -aX flags are given → run all A1–A4
- If any -aX flags are given → run only specified steps
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys


STEP_MODULES = {
    "a1": "code.evaluate.a1_factual_metrics",
    "a2": "code.evaluate.a2_latband_timeseries",
    "a3": "code.evaluate.a3_inertness_check",
    "a4": "code.evaluate.a4_dose_response_check",
}


def run_step(step: str, argv: list[str]) -> None:
    module = importlib.import_module(STEP_MODULES[step])
    if not hasattr(module, "main"):
        raise RuntimeError(f"{STEP_MODULES[step]} does not define main()")

    print(f"\n===== Running {step.upper()} =====")
    original_argv = sys.argv
    try:
        sys.argv = [STEP_MODULES[step]] + argv
        module.main()
    finally:
        sys.argv = original_argv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evaluation steps A1–A4 (default: all)"
    )

    # common inputs
    parser.add_argument(
        "--pred-csv",
        default="results/predictions/factual_rolling_predictions.csv",
        help="Rolling factual prediction CSV for A2–A4",
    )
    parser.add_argument(
        "--data-csv",
        default="data/data.csv",
        help="Original data.csv for A3–A4",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Base output directory for A3–A4 (A2 uses out-dir/figures)",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint path for A1 (default: latest in checkpoints/)",
    )
    parser.add_argument(
        "--a1-interpret",
        action="store_true",
        help="Enable feature importance output for A1",
    )

    # short flags for steps
    parser.add_argument("-a1", action="store_true", help="Run A1 only")
    parser.add_argument("-a2", action="store_true", help="Run A2 only")
    parser.add_argument("-a3", action="store_true", help="Run A3 only")
    parser.add_argument("-a4", action="store_true", help="Run A4 only")

    # step-specific optional args
    parser.add_argument(
        "--a2-seed",
        type=int,
        default=1,
        help="Random seed for A2 grid selection",
    )
    parser.add_argument(
        "--a2-window",
        type=int,
        default=24,
        help="Number of months to display in A2",
    )
    parser.add_argument(
        "--a3-seed",
        type=int,
        default=42,
        help="Seed for A3 bootstrap CI",
    )
    parser.add_argument(
        "--a3-eps",
        type=float,
        default=1e-6,
        help="Small epsilon to avoid division by zero in A3",
    )
    parser.add_argument(
        "--a3-clip-pct",
        type=float,
        default=200.0,
        help="Clip abs_delta_pct for plotting in A3",
    )
    parser.add_argument(
        "--a4-frac",
        type=float,
        default=0.3,
        help="LOWESS smoothing fraction for A4",
    )

    args = parser.parse_args()

    # determine which steps to run
    selected_steps = [step for step in ["a1", "a2", "a3", "a4"] if getattr(args, step)]
    if not selected_steps:
        selected_steps = ["a1", "a2", "a3", "a4"]

    fig_dir = os.path.join(args.out_dir, "figures")

    for step in selected_steps:
        if step == "a1":
            argv = []
            if args.ckpt is not None:
                argv += ["--ckpt", args.ckpt]
            if args.a1_interpret:
                argv.append("--interpret")
            run_step(step, argv)

        elif step == "a2":
            run_step(
                step,
                [
                    "--csv", args.pred_csv,
                    "--out-dir", fig_dir,
                    "--seed", str(args.a2_seed),
                    "--window", str(args.a2_window),
                ],
            )

        elif step == "a3":
            run_step(
                step,
                [
                    "--pred", args.pred_csv,
                    "--data", args.data_csv,
                    "--out-dir", args.out_dir,
                    "--eps", str(args.a3_eps),
                    "--clip-pct", str(args.a3_clip_pct),
                    "--seed", str(args.a3_seed),
                ],
            )

        elif step == "a4":
            run_step(
                step,
                [
                    "--pred-csv", args.pred_csv,
                    "--data-csv", args.data_csv,
                    "--out-dir", args.out_dir,
                    "--frac", str(args.a4_frac),
                ],
            )

    print("\n===== Selected evaluation steps completed =====")


if __name__ == "__main__":
    main()
