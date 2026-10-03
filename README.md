# Mamba-T

Pedestrian trajectory prediction using Mamba-based temporal modeling.

Stage 1: Matched Transformer (ETT) vs Mamba (EMT) backbone evaluation on ETH/UCY, using only target pedestrian history.

Datasets: ETH, HOTEL, UNIV, ZARA1, ZARA2.

Protocol: 8 observed steps, 12 predicted steps, 2.5 Hz, world coordinates in meters, deterministic K=1, ADE / FDE.

Five leave-one-scene-out folds. Source tracks are assigned to train/validation by a fixed SHA1 hash (85%/15%). Seeds 42, 123, 2024 affect training only. Held-out scenes are reserved and never evaluated in this stage.

Use the existing `ped_intent` conda environment. No Torch/CUDA/Mamba installation is required.

```bash
conda activate ped_intent
python scripts/acquire_eth_ucy.py
python scripts/build_eth_ucy_sequences.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python scripts/run_eth_ucy_baselines.py
python scripts/summarize_eth_ucy_baselines.py
```

The runner first executes a five-epoch ETH/seed-42 smoke for both models and checks loss reduction, finite metrics, and matching decoder hashes. It then runs all 30 experiments, each up to 100 epochs, with early stopping after 12 epochs without validation ADE improvement. Completed runs are reused. Interrupted runs restart; smoke outputs are separate from full runs.

Only source-scene validation metrics are reported. This stage cannot establish held-out benchmark superiority. Internal GO/STOP is computed from paired source validation runs. Condition B uses the same metric that satisfies condition A; paired ADE, FDE and joint wins are all reported. No second stage or hyperparameter search starts automatically.

Raw trajectories and checkpoints are ignored. Results, provenance copies, audits, and fixed split manifests are tracked. UNIV combines students001 and students003 with recording-qualified pedestrian IDs. Window stride is one, with no interpolation; incomplete or discontinuous 20-step windows are excluded. Frame differences are 10 in all recordings, representing distributed 2.5 Hz samples. Input displacements are meters per sampling step, not m/s.
