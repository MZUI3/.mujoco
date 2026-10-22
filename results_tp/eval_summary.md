# Trajectory planner evaluation (300 problems, seed 2026)

| method | success | reach<2cm | collision | median err (mm) | median clearance (cm) | EE path (m) | median jerk | track RMSE (deg) | latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| learned | 100.0% | 100.0% | 0.0% | 0.13 | 7.81 | 1.269 | 122.7 | 0.09 | 66.7 |
| learned+refine | 100.0% | 100.0% | 0.0% | 0.88 | 5.03 | 1.248 | 93.9 | 0.09 | 469.8 |
| trajopt | 100.0% | 100.0% | 0.0% | 0.81 | 5.41 | 1.241 | 97.4 | 0.08 | 5685.2 |
| linear | 70.3% | 89.0% | 29.7% | 0.12 | 4.66 | 1.184 | 2093.2 | 3.54 | 82.8 |
