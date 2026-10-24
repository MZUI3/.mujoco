# Trajectory planner evaluation (300 problems, seed 2026)

poles per problem uniform in [1, 3], radius 3-7 cm, height 15-40 cm

| method | success | reach<2cm | collision | median err (mm) | median clearance (cm) | EE path (m) | median jerk | track RMSE (deg) | latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| learned | 99.7% | 100.0% | 0.3% | 0.13 | 8.48 | 1.324 | 164.4 | 0.10 | 101.3 |
| learned+refine | 100.0% | 100.0% | 0.0% | 0.85 | 4.95 | 1.308 | 115.4 | 0.09 | 475.4 |
| trajopt | 100.0% | 100.0% | 0.0% | 0.81 | 5.08 | 1.303 | 114.3 | 0.08 | 7253.1 |
| linear | 70.0% | 86.3% | 30.0% | 0.12 | 4.69 | 1.209 | 2576.9 | 4.91 | 74.3 |

Success rate by number of obstacles

| method | 1 pole (n=119) | 2 poles (n=102) | 3 poles (n=79) |
|---|---|---|---|
| learned | 100.0% | 99.0% | 100.0% |
| learned+refine | 100.0% | 100.0% | 100.0% |
| trajopt | 100.0% | 100.0% | 100.0% |
| linear | 78.2% | 67.6% | 60.8% |
