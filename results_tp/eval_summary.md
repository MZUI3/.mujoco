# Trajectory planner evaluation (300 problems, seed 2026)

poles per problem uniform in [1, 3], radius 3-7 cm, height 15-40 cm

| method | success | reach<2cm | collision | median err (mm) | median clearance (cm) | EE path (m) | median jerk | track RMSE (deg) | latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| learned | 99.7% | 100.0% | 0.3% | 0.13 | 8.48 | 1.324 | 164.4 | 0.10 | 38.0 |
| learned+refine | 100.0% | 100.0% | 0.0% | 0.87 | 5.17 | 1.308 | 116.0 | 0.09 | 223.6 |
| trajopt | 100.0% | 100.0% | 0.0% | 0.81 | 5.56 | 1.301 | 113.9 | 0.08 | 4146.7 |
| linear | 70.0% | 86.3% | 30.0% | 0.12 | 4.76 | 1.211 | 2576.9 | 4.97 | 35.1 |

Success rate by number of obstacles

| method | 1 pole (n=119) | 2 poles (n=102) | 3 poles (n=79) |
|---|---|---|---|
| learned | 100.0% | 99.0% | 100.0% |
| learned+refine | 100.0% | 100.0% | 100.0% |
| trajopt | 100.0% | 100.0% | 100.0% |
| linear | 78.2% | 67.6% | 60.8% |
