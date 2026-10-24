# Trajectory planner evaluation (300 problems, seed 2026)

poles per problem uniform in [1, 1], fixed 4 cm x 30 cm

| method | success | reach<2cm | collision | median err (mm) | median clearance (cm) | EE path (m) | median jerk | track RMSE (deg) | latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| learned | 99.7% | 100.0% | 0.3% | 0.12 | 8.76 | 1.286 | 162.9 | 0.10 | 79.2 |
| linear | 71.7% | 87.0% | 28.3% | 0.12 | 5.58 | 1.191 | 2487.2 | 3.51 | 71.2 |

Success rate by number of obstacles

| method | 1 pole (n=300) |
|---|---|
| learned | 99.7% |
| linear | 71.7% |
