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

The matrix uses six independent training processes on one GPU, followed by exclusive inference benchmarking. Training wall times include GPU contention and should not be treated as standalone model speed comparisons. Peak GPU memory reports per-process PyTorch allocated bytes, excluding other processes and driver allocations.

Stage 2 runs on `feat/social_residual`: frozen ETT/EMT checkpoints with EMT-ZR (capacity control), EMT-SR (social correction), EMT-GSR (scalar gate), and ETT-SR. First-stage results and split manifests are frozen.

```bash
python scripts/build_eth_ucy_social_sequences.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python scripts/run_social_residual.py
python scripts/summarize_social_residual.py
```

Each sample uses at most eight nearest neighbors at the last observed frame, present throughout the eight observed frames in the same recording. No radius or future-based selection is used. Neighbor motion features use each neighbor's own last observed position; relation features are target-relative [dx, dy, distance]. Neighbor candidates may include source tracks assigned to the other target split, using their current observations only. Held-out recordings cannot be opened by the train/validation loader.

Temporal encoder and base decoder weights stay frozen and in eval mode. Frozen contexts are computed once per fold, backbone and seed, reused for training new modules, and checked against the original validation metrics. Full inference latency is measured without caches after all training exits. Social and gate GO/STOP thresholds require the percentage improvement and paired wins in the same metric. Neither decision starts held-out evaluation or further tuning.

Stage 4 (`feat/final_eth_ucy_benchmark`) performs the formal K=1 ETH/UCY benchmark with ETT, EMT, ETT-SR and EMT-SR. All eligible targets in the four source scenes are used; there is no internal validation split. Target-only models are retrained from random initialization, then SR uses the corresponding final fold/seed backbone frozen in eval mode. Train and test neighbors use the same eight-frame observation-only nearest-eight policy. Historical results and datasets remain frozen.

Final epochs are derived automatically from the median of the three validation best epochs: stage one for ETT/EMT, clean validation for ETT-SR/EMT-SR. Full-source training uses the original AdamW, learning rate, weight decay, loss, batch size and clipping. The learning rate stays at 1e-3 because the validation-driven plateau scheduler has no validation metric in this phase. There is no early stopping or metric-based checkpoint selection.

```bash
python scripts/build_eth_ucy_final_data.py
python scripts/freeze_final_eth_ucy_protocol.py --epochs-only
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
# Commit code/config/tests before freezing the protocol.
python scripts/freeze_final_eth_ucy_protocol.py
python scripts/run_final_eth_ucy_training.py --smoke-only
python scripts/run_final_eth_ucy_training.py
# Review FINAL_TRAINING_COMPLETE.json and commit training evidence before opening test.
python scripts/evaluate_final_eth_ucy.py --enable-heldout-evaluation
python scripts/summarize_final_eth_ucy.py
```

The held-out evaluator refuses to run until all 60 final checkpoints, fixed epochs, training logs and protocol hashes match. Each fold/model/seed is predicted once and logged before inference. Saved NPZ predictions include IDs, frames, ground truth and per-sample metrics; SR also includes neighbors, diagnostic attention, residual and base predictions. Attention diagnostics do not replace the original prediction path. Tables use fold-wise three-seed mean ± sample SD, then equally average the five folds. Validation EMT-ZR is kept separate from the formal test table. Visualization candidate indices are saved; no plots or test-based tuning follow automatically.

Stage 5 (`diag/source_validation_training_regime`) is a **source validation diagnostic only**. Formal test has already been accessed historically and is never reopened in this phase. It restores the frozen source pedestrian splits and clean neighbors, retrains ETT/EMT with the unchanged final epoch file and fixed learning rate, then freezes each corresponding new backbone for SR. Validation metrics are passive logs: they cannot change learning rate, duration or checkpoint choice. CPU/CUDA RNG and model state are checked around validation, and only the last epoch is saved.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python scripts/run_training_regime_diagnostic.py --smoke-only
python scripts/run_training_regime_diagnostic.py
python scripts/summarize_training_regime_diagnostic.py
```

The source-only entry points reject `heldout_test`. Legacy held-out loader routing tests use synthetic data in memory. Per-epoch JSON/CSV curves, final/best diagnostic gaps, density/distance/correction strata and correction norms are saved without figures. A diagnostic best epoch is computed after training and never used to select a checkpoint or rerun formal test. Old clean comparisons hold the data and architecture fixed while changing the entire backbone/SR training chain; they are not a scheduler-only ablation. Historical formal aggregate numbers are quoted from the stage request, without loading or recomputing formal test artifacts.
