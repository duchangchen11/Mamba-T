# Mamba-T

Pedestrian trajectory prediction using Mamba-based temporal modeling.

Stage 1: Matched Transformer (ETT) vs Mamba (EMT) backbone evaluation on ETH/UCY, using only target pedestrian history.

Datasets: ETH, HOTEL, UNIV, ZARA1, ZARA2.

Protocol: 8 observed steps, 12 predicted steps, 2.5 Hz, world coordinates in meters, deterministic K=1, ADE / FDE.

Five leave-one-scene-out folds. Source tracks are assigned to train/validation by a fixed SHA1 hash (85%/15%). Seeds 42, 123, 2024 affect training only. Held-out scenes are reserved and never evaluated in this stage.
