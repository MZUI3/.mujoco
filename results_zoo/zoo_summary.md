# RL model zoo evaluation

결정적 정책, 환경당 20 에피소드(200 step), seed 10000. success는 마지막 step의 과제 성공 조건.

## 환경

| env id | 과제 | obs | act |
|---|---|---|---|
| `Reach-v0` | 무작위 목표 위치에 EE 도달 (dense 보상, 성공 3 cm) | 17 | 4 |
| `ReachSparse-v0` | Reach와 같은 과제, 희소 보상 (3 cm 안에서만 +1) | 17 | 4 |
| `ReachObstacle-v0` | 시작-목표 사이 기둥(4 cm × 30 cm)을 피해 도달, 접촉 시 패널티·실패 | 21 | 4 |
| `ReachDR-v0` | Reach + 질량·댐핑·마찰 ±20~30 %, 행동 스케일 ±15 %, 관측 노이즈 | 17 | 4 |
| `ReachMoving-v0` | 반지름 5~10 cm 원을 1~2 rad/s로 도는 목표 추적 (성공 4 cm) | 20 | 4 |
| `Push-v0` | 바닥의 상자(6 cm)를 10~18 cm 떨어진 목표로 밀기 (성공 5 cm) | 26 | 4 |
| `Lift-v0` | 상자를 잡아 15 cm 이상 들기, 그리퍼 열기/닫기 포함 (행동 5) | 23 | 5 |
| `TrackTrajectory-v0` | 무작위 2초 관절 궤적 추종 (성공: 평균 오차 < 0.1 rad) | 21 | 4 |
| `HoldPose-v0` | 0.5초마다 손목에 5~15 N 외력, 목표 자세 유지 (성공: 마지막 0.5초 평균 오차 < 0.1 rad) | 19 | 4 |
| `ApproachObjects-v0` | 기존 DomainRandomizedEnv: 무작위 물체 2개에 접근, 토크 6, 동역학 랜덤화 | 38 | 6 |

## 모델

| # | env | model | steps | success | mean return ± std | final dist (m) | collision | train (min) | 파일 |
|---|---|---|---|---|---|---|---|---|---|
|  | `Reach-v0` | random (기준선) | - | 0% | -249.2 ± 63.2 | 0.549 | 90% | - | - |
| 1 | `Reach-v0` | A2C | 300k | **0%** | -174.8 ± 34.5 | 0.400 | 0% | 8.9 | `models_zoo/Reach-v0__A2C/model.zip` |
| 2 | `Reach-v0` | PPO | 300k | **0%** | -133.4 ± 47.9 | 0.335 | 0% | 12.4 | `models_zoo/Reach-v0__PPO/model.zip` |
| 3 | `Reach-v0` | SAC | 100k | **55%** | 275.0 ± 250.8 | 0.077 | 0% | 45.8 | `models_zoo/Reach-v0__SAC/model.zip` |
| 4 | `Reach-v0` | TD3 | 100k | **30%** | 174.5 ± 218.5 | 0.085 | 5% | 29.3 | `models_zoo/Reach-v0__TD3/model.zip` |
|  | `ReachSparse-v0` | random (기준선) | - | 0% | 0.1 ± 0.5 | 0.651 | 90% | - | - |
| 5 | `ReachSparse-v0` | PPO | 300k | **0%** | 0.0 ± 0.0 | 0.524 | 60% | 12.5 | `models_zoo/ReachSparse-v0__PPO/model.zip` |
| 6 | `ReachSparse-v0` | SAC | 100k | **0%** | 0.0 ± 0.0 | 0.392 | 20% | 45.9 | `models_zoo/ReachSparse-v0__SAC/model.zip` |
|  | `ReachObstacle-v0` | random (기준선) | - | 0% | -309.1 ± 58.2 | 0.704 | 90% | - | - |
| 7 | `ReachObstacle-v0` | PPO | 300k | **0%** | -149.2 ± 88.0 | 0.323 | 25% | 12.5 | `models_zoo/ReachObstacle-v0__PPO/model.zip` |
| 8 | `ReachObstacle-v0` | SAC | 100k | **0%** | -33.3 ± 184.0 | 0.156 | 25% | 45.5 | `models_zoo/ReachObstacle-v0__SAC/model.zip` |
|  | `ReachDR-v0` | random (기준선) | - | 0% | -260.1 ± 82.2 | 0.657 | 95% | - | - |
| 9 | `ReachDR-v0` | PPO | 300k | **10%** | 4.6 ± 143.4 | 0.161 | 0% | 12.3 | `models_zoo/ReachDR-v0__PPO/model.zip` |
| 10 | `ReachDR-v0` | SAC | 100k | **75%** | 362.8 ± 197.2 | 0.039 | 0% | 46.1 | `models_zoo/ReachDR-v0__SAC/model.zip` |
|  | `ReachMoving-v0` | random (기준선) | - | 0% | -245.1 ± 65.0 | 0.659 | 95% | - | - |
| 11 | `ReachMoving-v0` | SAC | 100k | **80%** | 389.4 ± 178.1 | 0.044 | 5% | 44.8 | `models_zoo/ReachMoving-v0__SAC/model.zip` |
|  | `Push-v0` | random (기준선) | - | 0% | -197.9 ± 38.1 | 0.189 | 0% | - | - |
| 12 | `Push-v0` | SAC | 100k | **70%** | 213.0 ± 192.7 | 0.045 | 0% | 45.2 | `models_zoo/Push-v0__SAC/model.zip` |
|  | `Lift-v0` | random (기준선) | - | 0% | -112.9 ± 21.9 | 0.150 | 0% | - | - |
| 13 | `Lift-v0` | SAC | 100k | **0%** | 91.0 ± 133.0 | 0.150 | 0% | 45.1 | `models_zoo/Lift-v0__SAC/model.zip` |
|  | `TrackTrajectory-v0` | random (기준선) | - | 0% | -740.3 ± 32.0 | 2.540 | 80% | - | - |
| 14 | `TrackTrajectory-v0` | PPO | 300k | **80%** | 83.4 ± 165.3 | 0.167 | 0% | 12.4 | `models_zoo/TrackTrajectory-v0__PPO/model.zip` |
| 15 | `TrackTrajectory-v0` | SAC | 100k | **50%** | 53.2 ± 162.5 | 0.196 | 0% | 42.5 | `models_zoo/TrackTrajectory-v0__SAC/model.zip` |
|  | `HoldPose-v0` | random (기준선) | - | 0% | -694.7 ± 145.8 | 2.319 | 85% | - | - |
| 16 | `HoldPose-v0` | PPO | 300k | **90%** | 160.9 ± 38.7 | 0.037 | 0% | 11.1 | `models_zoo/HoldPose-v0__PPO/model.zip` |
|  | `ApproachObjects-v0` | random (기준선) | - | 5% | 126.4 ± 64.8 | 0.152 | 0% | - | - |
| 17 | `ApproachObjects-v0` | SAC | 100k | **25%** | 273.4 ± 54.3 | 0.100 | 0% | 39.4 | `models_zoo/ApproachObjects-v0__SAC/model.zip` |
| 18 | ApproachObjects-v0 | PPO (PyTorch 자체 구현) | - | - | 157.2 ± 62.6 | - | - | - | `models/ppo_policy.pt` (results/compare_results.csv 기록 (200 step 에피소드)) |
| 19 | ApproachObjects-v0 | Diffusion Policy (TemporalUNet+DDPM) | - | - | 37.0 ± 95.9 | - | - | - | `models/diffusion_policy.pt` (results/compare_results.csv 기록 (200 step 에피소드)) |
| 20 | ApproachObjects-v0 | Diffusion Policy + Domain Randomization | - | - | - | - | - | - | `models_dr/diffusion_policy_dr.pt` (평가 기록 없음) |
| 21 | TrajectoryPlanning (기둥 1~3개) | Learned Trajectory Planner (self-supervised) | - | 99.7% | - | 0.0001 | 0.3% | - | `models_tp/planner.pt` (results_tp/eval_summary.json, 300문제 MuJoCo 롤아웃) |

총 21개 모델 (SB3 zoo 17개 + 기존 4개).

![success](zoo_success.png)

![returns](zoo_returns.png)
