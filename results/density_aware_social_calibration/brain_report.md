SOURCE VALIDATION DIAGNOSTIC ONLY

1. Branch: feat/density_aware_social_calibration.
2. Latest result commit SHA: recorded in the final handoff; a commit cannot contain its own SHA.
3. Push status: final result tree and branch head will be verified against GitHub before handoff.
4. heldout_test_accessed=false.
5. historical_test_already_accessed=true.
6. Full pytest: 139 passed, 0 failed.
7. New source-validation runs: 15/15; old EMT-SR baseline was reused, not retrained.
8. Per fold/seed scene-equal ADE/FDE and paired deltas: summary.md, “Fold / Seed”.
9. Five-scene equal ADE/FDE and pooled averages: comparison.json, `overall`.
10. ΔADE/ΔFDE=-0.005466/-0.012230; relative=-1.019%/-1.123%; scene-equal paired ADE/FDE wins=10/15, 11/15.
11. Per-scene baseline/density-aware social ADE gains appear in summary.md.
12. Baseline scene gain population SD=0.009066.
13. Density-aware scene gain population SD=0.004171.
14. Baseline scene gain range=0.023254.
15. Density-aware scene gain range=0.012417.
16. Baseline worst scene gain=-0.001199.
17. Density-aware worst scene gain=0.012596.
18. Neighbor groups 0, 1–2, 3–4, 5–6, 7–8 show base/baseline/new ADE/FDE and gains in summary.md.
19. Baseline neighbor-count gain population SD=0.018770.
20. Density-aware neighbor-count gain population SD=0.016572.
21. Mean scale at n=0…8 across 15 models appears in summary.md.
22. Scale vs count scene-equal Pearson/Spearman=0.289137/0.408125.
23. Scale vs residual norm scene-equal Pearson/Spearman=-0.130760/-0.125322.
24. Scale mean/std/min/max=0.713003/0.095422/0.507527/1.020501; finite and within strict bounds=True/True.
25. Density response MEANINGFUL; Var_n(E[s|n])=0.00295584, scale range over n=0.168667; higher neighbor counts increase the social scale.
26. Maximum pre-clip gradient norm=0.561264; NaN/Inf=False.
27. DENSITY CALIBRATION: GO.
28. DENSITY RESPONSE: MEANINGFUL.
29. CROSS-SCENE STABILITY: IMPROVED.
30. Summary: results/density_aware_social_calibration/summary.md.
31. Brain report: results/density_aware_social_calibration/brain_report.md.
32. Source-only diagnostic complete; no formal test or SDD was run.

The density module sees only observed neighbor count/8 and scales the attention context. This is density-conditioned calibration, not the prior scalar gate. The final layer starts at zero; shared social/residual initialization matches the frozen EMT-SR runs.
All values are averaged equally across validation folds/seeds; within-fold source scenes are weighted equally for the primary view. Pooled sample results are retained because scene counts differ.
