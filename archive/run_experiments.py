#!/usr/bin/env python
"""
Run multiple TFT training experiments with different hyperparameters.

Each experiment will be saved in its own directory under experiments/
"""

import os
import sys
import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from code.train.train_tft_experiment import train_experiment


def create_experiment_dir(base_dir="experiments", experiment_name=None):
    """Create a directory for an experiment with timestamp."""
    os.makedirs(base_dir, exist_ok=True)

    if experiment_name:
        # Use provided name with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        # Auto-numbered experiment
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Find next available number
        existing = [d for d in os.listdir(base_dir) if d.startswith("exp_")]
        if existing:
            numbers = []
            for d in existing:
                try:
                    num = int(d.split("_")[1])
                    numbers.append(num)
                except (IndexError, ValueError):
                    continue
            next_num = max(numbers) + 1 if numbers else 1
        else:
            next_num = 1
        dir_name = f"exp_{next_num:03d}_{timestamp}"

    experiment_dir = os.path.join(base_dir, dir_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def main():
    """Run multiple experiments with different hyperparameters."""

    # Base experiments directory
    experiments_base = "experiments"

    # Define experiment configurations
    # Each dict contains parameter overrides for one experiment
    experiments = [
        # Experiment 1: Baseline (current config)
        {
            "name": "baseline",
            "overrides": {}  # No changes
        },

        # Experiment 2: Larger model
        {
            "name": "large_model",
            "overrides": {
                "model.hidden_size": 128,
                "model.hidden_continuous_size": 32,
            }
        },

        # Experiment 3: Smaller model (faster training)
        {
            "name": "small_model",
            "overrides": {
                "model.hidden_size": 32,
                "model.hidden_continuous_size": 8,
                "model.lstm_layers": 1,
            }
        },

        # Experiment 4: Higher dropout for regularization
        {
            "name": "high_dropout",
            "overrides": {
                "model.dropout": 0.2,
                "training.batch_size": 16,  # Smaller batch with higher dropout
            }
        },

        # Experiment 5: Different learning rate
        {
            "name": "higher_lr",
            "overrides": {
                "model.learning_rate": 0.002,
                "training.gradient_clip_val": 0.2,  # Higher clip for higher LR
            }
        },

        # Experiment 6: Lower learning rate
        {
            "name": "lower_lr",
            "overrides": {
                "model.learning_rate": 0.0005,
                "training.max_epochs": 40,  # May need more epochs with lower LR
            }
        },

        # Experiment 7: Different attention heads
        {
            "name": "more_heads",
            "overrides": {
                "model.attention_head_size": 8,
            }
        },

        # Experiment 8: Larger batch size
        {
            "name": "large_batch",
            "overrides": {
                "training.batch_size": 64,
                "model.learning_rate": 0.002,  # Often use higher LR with larger batch
            }
        },

        # Experiment 9: More LSTM layers
        {
            "name": "more_layers",
            "overrides": {
                "model.lstm_layers": 3,
                "model.dropout": 0.15,  # Slightly higher dropout for deeper network
            }
        },

        # Experiment 10: Combined tuning
        {
            "name": "tuned_combined",
            "overrides": {
                "model.hidden_size": 96,
                "model.learning_rate": 0.0015,
                "model.dropout": 0.15,
                "training.batch_size": 48,
            }
        },
    ]

    print("=" * 70)
    print(f"Starting {len(experiments)} TFT training experiments")
    print(f"Results will be saved to: {experiments_base}/")
    print("=" * 70)

    results = []

    for i, exp_config in enumerate(experiments, 1):
        exp_name = exp_config["name"]
        overrides = exp_config.get("overrides", {})

        print(f"\n{'='*70}")
        print(f"Experiment {i}/{len(experiments)}: {exp_name}")
        print(f"{'='*70}")

        # Create experiment directory
        experiment_dir = create_experiment_dir(experiments_base, f"{exp_name}")

        # Print overrides
        if overrides:
            print("Parameter overrides:")
            for key, value in overrides.items():
                print(f"  {key}: {value}")
        else:
            print("Using baseline configuration (no overrides)")

        try:
            # Run training
            print(f"\nStarting training...")
            print(f"Experiment directory: {experiment_dir}")

            model, scaler = train_experiment(
                config_path=None,  # Use default config
                experiment_dir=experiment_dir,
                overrides=overrides
            )

            results.append({
                "experiment": exp_name,
                "directory": experiment_dir,
                "status": "success",
                "error": None
            })

            print(f"✓ Experiment '{exp_name}' completed successfully")

        except Exception as e:
            print(f"✗ Experiment '{exp_name}' failed: {e}")
            results.append({
                "experiment": exp_name,
                "directory": experiment_dir,
                "status": "failed",
                "error": str(e)
            })
            # Continue with next experiment
            continue

    # Print summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")

    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")

    if failed_count > 0:
        print("\nFailed experiments:")
        for r in results:
            if r["status"] == "failed":
                print(f"  - {r['experiment']}: {r['error']}")

    print(f"\nAll experiment directories are under: {experiments_base}/")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()