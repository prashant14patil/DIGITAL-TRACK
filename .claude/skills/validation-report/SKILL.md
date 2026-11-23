---
name: validation-report
description: Produce an accuracy and repeatability report from scan measurements versus reference values (tape, known objects, reference scanner). Use after a validation stage run.
---

Inputs: measured values per scan/participant, reference values, repeat-scan grouping, validation stage (1 camera + scale, 2 turntable + reconstruction, 3 mannequin, 4 human pilot).

For each measurement compute and tabulate:
- MAE (cm), mean signed error (bias), relative % error
- Bland-Altman bias and 95% limits of agreement
- Within-session and between-session repeatability (SD / CV); ICC where appropriate
- Smallest detectable change
For geometry (if meshes available): point-to-surface / mesh-to-mesh distance, RMS deviation, completeness, registration residual.

Rules: report bias and agreement, never correlation alone; state sample size and conditions (clothing, distance, protocol, calibration_id); compare against provisional targets (~2 cm or 3-5% circumference error) as evaluation targets, not achieved claims; don't cite other sensors' papers as iPad accuracy; list scans excluded by quality checks and why.

Output a markdown report with a summary table, key caveats, and a go/no-go recommendation for the next phase.
