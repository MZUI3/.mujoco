# RL model zoo evaluation

결정적 정책, 환경당 20 에피소드(200 step), 평가 seed 10000. success는 마지막 step의 과제 성공 조건.
학습 seed가 여러 개인 모델은 seed별 성공률의 평균 ± 표준편차(n = seed 수)로 표시합니다.

## 환경

| env id | 과제 | obs | act |
|---|---|---|---|
| `Reach-v0` | 무작위 목표 위치에 EE 도달 (dense 보상, 성공 3 cm) | 17 | 4 |
| `ReachSparse-v0` | Reach와 같은 과제, 희소 보상 (3 cm 안에서만 +1) | 17 | 4 |
| `ReachObstacle-v0` | 시작-목표 사이 기둥(4 cm × 30 cm)을 피해 도달, 접촉 시 패널티·실패 | 21 | 4 |
| `ReachMultiObstacle-v0` | 기둥 1~3개(반지름 3~7 cm, 높이 15~40 cm)를 피해 도달, 궤적 플래너와 같은 문제 분포, 접촉 시 패널티·실패 | 29 | 4 |
| `ReachDR-v0` | Reach + 질량·댐핑·마찰 ±20~30 %, 행동 스케일 ±15 %, 관측 노이즈 | 17 | 4 |
| `ReachMoving-v0` | 반지름 5~10 cm 원을 1~2 rad/s로 도는 목표 추적 (성공 4 cm) | 20 | 4 |
| `Push-v0` | 바닥의 상자(6 cm)를 10~18 cm 떨어진 목표로 밀기 (성공 5 cm) | 26 | 4 |
| `Lift-v0` | 상자를 잡아 15 cm 이상 들기, 그리퍼 열기/닫기 포함 (행동 5) | 23 | 5 |
| `TrackTrajectory-v0` | 무작위 2초 관절 궤적 추종 (성공: 평균 오차 < 0.1 rad) | 21 | 4 |
| `HoldPose-v0` | 0.5초마다 손목에 5~15 N 외력, 목표 자세 유지 (성공: 마지막 0.5초 평균 오차 < 0.1 rad) | 19 | 4 |
| `ApproachObjects-v0` | 기존 DomainRandomizedEnv: 무작위 물체 2개에 접근, 토크 6, 동역학 랜덤화 | 38 | 6 |

## 모델

| # | env | model | steps | seeds | success | mean return ± std | final dist (m) | collision | train (min) | 파일 |
|---|---|---|---|---|---|---|---|---|---|---|
|  | `Reach-v0` | random (기준선) | - | - | 0% | -259.3 ± 62.3 | 0.591 | 90% | - | - |
| 1 | `Reach-v0` | A2C | 300k | 1 | **0%** | -175.0 ± 34.7 | 0.400 | 0% | 8.9 | `models_zoo/Reach-v0__A2C/model.zip` |
| 2 | `Reach-v0` | PPO | 300k | 1 | **0%** | -133.4 ± 47.9 | 0.335 | 0% | 12.4 | `models_zoo/Reach-v0__PPO/model.zip` |
| 5 | `Reach-v0` | SAC | 100k | 3 | **70 ± 11%** | 349.5 ± 52.9 | 0.058 | 2% | 37.5 | `models_zoo/Reach-v0__SAC/model.zip` |
| 6 | `Reach-v0` | TD3 | 100k | 1 | **30%** | 174.5 ± 218.5 | 0.085 | 5% | 29.3 | `models_zoo/Reach-v0__TD3/model.zip` |
|  | `ReachSparse-v0` | random (기준선) | - | - | 0% | 0.1 ± 0.4 | 0.611 | 95% | - | - |
| 7 | `ReachSparse-v0` | PPO | 300k | 1 | **0%** | 0.0 ± 0.0 | 0.518 | 60% | 12.5 | `models_zoo/ReachSparse-v0__PPO/model.zip` |
| 8 | `ReachSparse-v0` | SAC | 300k | 1 | **15%** | 28.9 ± 60.7 | 0.294 | 40% | 110.3 | `models_zoo/ReachSparse-v0__SAC/model.zip` |
| 9 | `ReachSparse-v0` | SAC_HER | 300k | 1 | **65%** | 116.8 ± 84.5 | 0.054 | 5% | 118.1 | `models_zoo/ReachSparse-v0__SAC_HER/model.zip` |
|  | `ReachObstacle-v0` | random (기준선) | - | - | 0% | -306.3 ± 76.1 | 0.637 | 95% | - | - |
| 10 | `ReachObstacle-v0` | PPO | 300k | 1 | **0%** | -146.2 ± 80.5 | 0.323 | 25% | 12.5 | `models_zoo/ReachObstacle-v0__PPO/model.zip` |
| 11 | `ReachObstacle-v0` | SAC | 300k | 1 | **30%** | 160.4 ± 330.1 | 0.090 | 30% | 110.7 | `models_zoo/ReachObstacle-v0__SAC/model.zip` |
|  | `ReachMultiObstacle-v0` | random (기준선) | - | - | 0% | -317.5 ± 81.7 | 0.631 | 95% | - | - |
| 12 | `ReachMultiObstacle-v0` | BC | 1019k demo | 1 | **25%** | -159.3 ± 51.2 | 0.047 | 10% | 13.8 | `models_zoo/ReachMultiObstacle-v0__BC/model.pt` |
| 13 | `ReachMultiObstacle-v0` | BC_naive | 379k demo | 1 | **0%** | -211.5 ± 73.8 | 0.338 | 40% | 1.3 | `models_zoo/ReachMultiObstacle-v0__BC_naive/model.pt` |
| 14 | `ReachMultiObstacle-v0` | SAC | 100k | 1 | **0%** | -7.4 ± 47.7 | 0.110 | 15% | 20.8 | `models_zoo/ReachMultiObstacle-v0__SAC/model.zip` |
|  | `ReachDR-v0` | random (기준선) | - | - | 0% | -277.9 ± 79.7 | 0.628 | 95% | - | - |
| 15 | `ReachDR-v0` | PPO | 300k | 1 | **10%** | 4.6 ± 143.4 | 0.161 | 0% | 12.3 | `models_zoo/ReachDR-v0__PPO/model.zip` |
| 18 | `ReachDR-v0` | SAC | 100k | 3 | **75 ± 8%** | 366.3 ± 38.3 | 0.031 | 2% | 37.6 | `models_zoo/ReachDR-v0__SAC/model.zip` |
|  | `ReachMoving-v0` | random (기준선) | - | - | 0% | -249.4 ± 74.2 | 0.534 | 85% | - | - |
| 21 | `ReachMoving-v0` | SAC | 100k | 3 | **90 ± 7%** | 408.0 ± 20.8 | 0.042 | 5% | 37.2 | `models_zoo/ReachMoving-v0__SAC/model.zip` |
|  | `Push-v0` | random (기준선) | - | - | 5% | -172.2 ± 104.5 | 0.175 | 0% | - | - |
| 24 | `Push-v0` | SAC | 100k | 3 | **63 ± 12%** | 205.0 ± 38.9 | 0.061 | 0% | 37.6 | `models_zoo/Push-v0__SAC/model.zip` |
|  | `Lift-v0` | random (기준선) | - | - | 0% | -73.8 ± 76.5 | 0.150 | 0% | - | - |
| 25 | `Lift-v0` | SAC | 300k | 1 | **30%** | 426.9 ± 159.3 | 0.083 | 0% | 110.0 | `models_zoo/Lift-v0__SAC/model.zip` |
|  | `TrackTrajectory-v0` | random (기준선) | - | - | 0% | -740.3 ± 29.6 | 2.378 | 85% | - | - |
| 26 | `TrackTrajectory-v0` | PPO | 300k | 1 | **80%** | 83.4 ± 164.9 | 0.167 | 0% | 12.4 | `models_zoo/TrackTrajectory-v0__PPO/model.zip` |
| 27 | `TrackTrajectory-v0` | SAC | 100k | 1 | **50%** | 53.3 ± 162.2 | 0.196 | 0% | 42.5 | `models_zoo/TrackTrajectory-v0__SAC/model.zip` |
|  | `HoldPose-v0` | random (기준선) | - | - | 0% | -705.2 ± 152.1 | 2.511 | 90% | - | - |
| 30 | `HoldPose-v0` | PPO | 300k | 3 | **88 ± 2%** | 152.3 ± 6.5 | 0.045 | 0% | 11.0 | `models_zoo/HoldPose-v0__PPO/model.zip` |
|  | `ApproachObjects-v0` | random (기준선) | - | - | 5% | 124.9 ± 57.9 | 0.147 | 0% | - | - |
| 31 | `ApproachObjects-v0` | SAC | 100k | 1 | **25%** | 274.1 ± 54.3 | 0.099 | 0% | 39.4 | `models_zoo/ApproachObjects-v0__SAC/model.zip` |
| 32 | ApproachObjects-v0 | PPO (PyTorch 자체 구현) | - | - | - | 219.7 ± 34.1 | - | - | - | `models/ppo_policy.pt` (results/compare_results.csv 기록 (200 step 에피소드)) |
| 33 | ApproachObjects-v0 | Diffusion Policy (TemporalUNet+DDPM) | - | - | - | 50.4 ± 107.3 | - | - | - | `models/diffusion_policy.pt` (results/compare_results.csv 기록 (200 step 에피소드)) |
| 34 | ApproachObjects-v0 | Diffusion Policy + Domain Randomization | - | - | - | - | - | - | - | `models_dr/diffusion_policy_dr.pt` (평가 기록 없음) |
| 35 | TrajectoryPlanning (기둥 1~3개) | Learned Trajectory Planner (self-supervised) | - | - | 99.7% | - | 0.0001 | 0.3% | - | `models_tp/planner.pt` (results_tp/eval_summary.json, 300문제 MuJoCo 롤아웃) |

총 35개 모델 (SB3 zoo 31개 = 21개 (env, algo) 조합 × seed + 기존 4개).

![success](zoo_success.png)

![returns](zoo_returns.png)
