# RL model zoo evaluation

결정적 정책, 환경당 20 에피소드(200 step), 평가 seed 10000. success는 마지막 step의 과제 성공 조건.
학습 seed가 여러 개인 모델은 seed별 성공률의 평균 ± 표준편차(n = seed 수)로 표시합니다.

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

| # | env | model | steps | seeds | success | mean return ± std | final dist (m) | collision | train (min) | 파일 |
|---|---|---|---|---|---|---|---|---|---|---|
|  | `Reach-v0` | random (기준선) | - | - | 0% | -260.2 ± 75.5 | 0.623 | 90% | - | - |
| 1 | `Reach-v0` | A2C | 300k | 1 | **0%** | -174.8 ± 34.5 | 0.400 | 0% | 8.9 | `models_zoo/Reach-v0__A2C/model.zip` |
| 2 | `Reach-v0` | PPO | 300k | 1 | **0%** | -133.4 ± 47.9 | 0.335 | 0% | 12.4 | `models_zoo/Reach-v0__PPO/model.zip` |
| 5 | `Reach-v0` | SAC | 100k | 3 | **70 ± 11%** | 349.5 ± 52.9 | 0.058 | 2% | 37.5 | `models_zoo/Reach-v0__SAC/model.zip` |
| 6 | `Reach-v0` | TD3 | 100k | 1 | **30%** | 174.5 ± 218.5 | 0.085 | 5% | 29.3 | `models_zoo/Reach-v0__TD3/model.zip` |
|  | `ReachSparse-v0` | random (기준선) | - | - | 0% | 0.0 ± 0.0 | 0.672 | 80% | - | - |
| 7 | `ReachSparse-v0` | PPO | 300k | 1 | **0%** | 0.0 ± 0.0 | 0.524 | 60% | 12.5 | `models_zoo/ReachSparse-v0__PPO/model.zip` |
| 8 | `ReachSparse-v0` | SAC | 300k | 1 | **15%** | 28.9 ± 60.7 | 0.293 | 40% | 110.3 | `models_zoo/ReachSparse-v0__SAC/model.zip` |
| 9 | `ReachSparse-v0` | SAC_HER | 300k | 1 | **65%** | 116.8 ± 84.5 | 0.055 | 5% | 118.1 | `models_zoo/ReachSparse-v0__SAC_HER/model.zip` |
|  | `ReachObstacle-v0` | random (기준선) | - | - | 0% | -305.2 ± 58.0 | 0.595 | 90% | - | - |
| 10 | `ReachObstacle-v0` | PPO | 300k | 1 | **0%** | -149.2 ± 88.0 | 0.323 | 25% | 12.5 | `models_zoo/ReachObstacle-v0__PPO/model.zip` |
| 11 | `ReachObstacle-v0` | SAC | 300k | 1 | **30%** | 163.9 ± 310.6 | 0.083 | 30% | 110.7 | `models_zoo/ReachObstacle-v0__SAC/model.zip` |
|  | `ReachDR-v0` | random (기준선) | - | - | 0% | -254.0 ± 59.4 | 0.646 | 85% | - | - |
| 12 | `ReachDR-v0` | PPO | 300k | 1 | **10%** | 4.6 ± 143.4 | 0.161 | 0% | 12.3 | `models_zoo/ReachDR-v0__PPO/model.zip` |
| 15 | `ReachDR-v0` | SAC | 100k | 3 | **75 ± 8%** | 366.3 ± 38.3 | 0.031 | 2% | 37.6 | `models_zoo/ReachDR-v0__SAC/model.zip` |
|  | `ReachMoving-v0` | random (기준선) | - | - | 0% | -262.0 ± 63.0 | 0.599 | 95% | - | - |
| 18 | `ReachMoving-v0` | SAC | 100k | 3 | **90 ± 7%** | 408.1 ± 20.9 | 0.042 | 5% | 37.2 | `models_zoo/ReachMoving-v0__SAC/model.zip` |
|  | `Push-v0` | random (기준선) | - | - | 10% | -156.4 ± 124.2 | 0.162 | 0% | - | - |
| 21 | `Push-v0` | SAC | 100k | 3 | **58 ± 8%** | 185.5 ± 21.0 | 0.064 | 0% | 37.6 | `models_zoo/Push-v0__SAC/model.zip` |
|  | `Lift-v0` | random (기준선) | - | - | 0% | -94.6 ± 55.2 | 0.150 | 0% | - | - |
| 22 | `Lift-v0` | SAC | 300k | 1 | **35%** | 461.5 ± 132.5 | 0.073 | 0% | 110.0 | `models_zoo/Lift-v0__SAC/model.zip` |
|  | `TrackTrajectory-v0` | random (기준선) | - | - | 0% | -747.9 ± 39.4 | 2.742 | 95% | - | - |
| 23 | `TrackTrajectory-v0` | PPO | 300k | 1 | **80%** | 83.4 ± 165.3 | 0.167 | 0% | 12.4 | `models_zoo/TrackTrajectory-v0__PPO/model.zip` |
| 24 | `TrackTrajectory-v0` | SAC | 100k | 1 | **50%** | 53.2 ± 162.5 | 0.196 | 0% | 42.5 | `models_zoo/TrackTrajectory-v0__SAC/model.zip` |
|  | `HoldPose-v0` | random (기준선) | - | - | 0% | -714.7 ± 120.0 | 2.354 | 85% | - | - |
| 27 | `HoldPose-v0` | PPO | 300k | 3 | **88 ± 2%** | 152.6 ± 6.5 | 0.045 | 0% | 11.0 | `models_zoo/HoldPose-v0__PPO/model.zip` |
|  | `ApproachObjects-v0` | random (기준선) | - | - | 10% | 144.6 ± 77.2 | 0.144 | 0% | - | - |
| 28 | `ApproachObjects-v0` | SAC | 100k | 1 | **25%** | 273.4 ± 54.3 | 0.100 | 0% | 39.4 | `models_zoo/ApproachObjects-v0__SAC/model.zip` |
| 29 | ApproachObjects-v0 | PPO (PyTorch 자체 구현) | - | - | - | 157.2 ± 62.6 | - | - | - | `models/ppo_policy.pt` (results/compare_results.csv 기록 (200 step 에피소드)) |
| 30 | ApproachObjects-v0 | Diffusion Policy (TemporalUNet+DDPM) | - | - | - | 37.0 ± 95.9 | - | - | - | `models/diffusion_policy.pt` (results/compare_results.csv 기록 (200 step 에피소드)) |
| 31 | ApproachObjects-v0 | Diffusion Policy + Domain Randomization | - | - | - | - | - | - | - | `models_dr/diffusion_policy_dr.pt` (평가 기록 없음) |
| 32 | TrajectoryPlanning (기둥 1~3개) | Learned Trajectory Planner (self-supervised) | - | - | 99.7% | - | 0.0001 | 0.3% | - | `models_tp/planner.pt` (results_tp/eval_summary.json, 300문제 MuJoCo 롤아웃) |

총 32개 모델 (SB3 zoo 28개 = 18개 (env, algo) 조합 × seed + 기존 4개).

![success](zoo_success.png)

![returns](zoo_returns.png)
