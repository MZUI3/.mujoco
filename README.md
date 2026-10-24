# MuJoCo Manipulator Learning

MuJoCo 시뮬레이션에서 4자유도 로봇팔과 그리퍼(`yang_manipulator.xml`)를 대상으로 여러 학습 기반 제어 방법을 실험하는 저장소입니다.

| 트랙 | 내용 | 문서 |
|---|---|---|
| Learned Trajectory Planning | 장애물을 피해 목표까지 가는 관절 궤적을 신경망이 생성합니다. MuJoCo 동역학으로 평가하고 시각화합니다. | [trajectory_planning/README.md](trajectory_planning/README.md) |
| Diffusion Policy | 궤적 수집, TemporalUNet + DDPM 학습, receding-horizon 실행, Domain Randomization 재학습으로 이어지는 6단계 파이프라인입니다. | [README_PIPELINE.md](README_PIPELINE.md) |
| RL baselines | PPO(PyTorch 구현)와 SAC(stable-baselines3)로 학습하고 Diffusion 정책과 비교합니다. | [README_PIPELINE.md](README_PIPELINE.md#추가-rl-baseline--ppo-proximal-policy-optimization) |
| 과제 환경 + RL Model Zoo | Gymnasium 과제 환경 10종과 SAC/TD3/PPO/A2C 모델 매트릭스. 한 명령으로 학습·평가·GIF/뷰어 시각화. | [rl_zoo/README.md](rl_zoo/README.md) |

## Learned Trajectory Planning

시작 관절각, 목표 위치, 기둥 장애물(1~3개, 반지름 3~7 cm, 높이 15~40 cm 무작위)이 주어지면 2초 길이의 B-spline 관절 궤적을 한 번에 출력합니다.
전문가 데모 없이 학습합니다. MuJoCo 모델에서 뽑아낸 미분 가능한 순기구학으로 궤적 비용(목표 오차, 충돌, 관절 한계, 가속도)을 계산해 직접 최소화합니다.
장애물은 집합 인코더(기둥별 공유 MLP + max/mean 풀링)로 처리해서 개수가 달라도 같은 네트워크를 씁니다.

![learned vs straight line](results_tp/videos/problem_004_learned_vs_linear.gif)

왼쪽은 학습 플래너로, 기둥을 피해 목표에 도달합니다. 오른쪽은 관절공간 직선 경로로, 기둥에 부딪힙니다.
얇은 선은 계획한 경로이고 굵은 선은 MuJoCo에서 실제로 추종한 경로입니다.

테스트 300문제(기둥 1~3개, 크기 무작위)를 MuJoCo 동역학(computed-torque 추종 제어)으로 실행한 결과입니다.

| 방법 | 성공률 | 충돌률 | 기둥 3개 성공률 | 문제당 계획 시간 |
|---|---|---|---|---|
| learned (학습 플래너) | 99.7% | 0.3% | 100.0% | 101 ms |
| learned+refine (학습 결과 + 최적화 30 step) | 100.0% | 0.0% | 100.0% | 475 ms |
| trajopt (문제마다 최적화) | 100.0% | 0.0% | 100.0% | 7,253 ms |
| linear (IK + 관절공간 직선) | 70.0% | 30.0% | 60.8% | 74 ms |

학습 플래너는 문제마다 최적화하는 방식과 같은 수준의 성공률을 약 72배 빠르게 냅니다.
기둥 1개 고정 크기(v1) 분포에서도 99.7%로 이전 단일 기둥 모델(100%)과 같은 수준입니다.

![success rates](results_tp/success_by_obstacles.png)

```bash
python trajectory_planning/train_planner.py                       # 학습 (CPU 약 20분)
python trajectory_planning/evaluate_planner.py --n 300             # MuJoCo 롤아웃 평가 + 그래프
python trajectory_planning/visualize_planner.py --problems 4 19 29 --methods learned linear   # GIF
python trajectory_planning/visualize_planner.py --viewer --methods learned trajopt           # 인터랙티브 뷰어
# 학습된 모델을 원하는 목표·장애물로 MuJoCo에서 실행 (뷰어: --viewer, GIF: --gif 경로, 궤적 저장: --save-npz)
python trajectory_planning/simulate_planner.py --goal 0.35 -0.25 0.20 --obstacle 0.30 0.05 0.04 0.30 --viewer
```

그래프와 지표 정의, 시각화 옵션, 파이썬 API, 설계 과정은 [trajectory_planning/README.md](trajectory_planning/README.md)에 정리했습니다.

## 과제 환경 10종 + RL Model Zoo

`envs/`에 같은 로봇팔로 만든 Gymnasium 과제 환경 10종(도달, 희소 보상 도달, 장애물 회피 도달, 도메인 랜덤화,
이동 목표 추적, 밀기, 들기, 궤적 추종, 자세 유지, 기존 물체 접근 환경)이 있고, `rl_zoo/`가 그 위에서
stable-baselines3(SAC/TD3/PPO/A2C, 희소 보상용 SAC+HER)로 모델 매트릭스를 학습·평가·시각화합니다.

환경 10개 × 알고리즘 매트릭스로 모델 18개를 학습했고, 기존 모델 4개(PPO, Diffusion Policy, Diffusion+DR, 학습 궤적 플래너)와 합쳐 22개입니다. 아래는 환경별 최고 모델입니다 (20 에피소드, 결정적 정책, MuJoCo 롤아웃).

| 환경 | 과제 | 무작위 성공률 | 최고 모델 | 성공률 | 평균 리턴 |
|---|---|---|---|---|---|
| `Reach-v0` | 무작위 목표 위치에 EE 도달 | 0% | SAC (100k) / A2C 0%, PPO 0%, TD3 30% | **55%** | 275 |
| `ReachSparse-v0` | Reach와 같은 과제, 희소 보상 | 0% | SAC_HER (300k) / PPO 0%, SAC 15% | **65%** | 117 |
| `ReachObstacle-v0` | 시작-목표 사이 기둥(4 cm × 30 cm)을 피해 도달, 접촉 시 패널티·실패 | 0% | SAC (300k) / PPO 0% | **30%** | 164 |
| `ReachDR-v0` | Reach + 질량·댐핑·마찰 ±20~30 %, 행동 스케일 ±15 %, 관측 노이즈 | 0% | SAC (100k) / PPO 10% | **75%** | 363 |
| `ReachMoving-v0` | 반지름 5~10 cm 원을 1~2 rad/s로 도는 목표 추적 | 0% | SAC (100k) | **80%** | 389 |
| `Push-v0` | 바닥의 상자(6 cm)를 10~18 cm 떨어진 목표로 밀기 | 0% | SAC (100k) | **70%** | 213 |
| `Lift-v0` | 상자를 잡아 15 cm 이상 들기, 그리퍼 열기/닫기 포함 | 0% | SAC (300k) | **35%** | 462 |
| `TrackTrajectory-v0` | 무작위 2초 관절 궤적 추종 | 0% | PPO (300k) / SAC 50% | **80%** | 83 |
| `HoldPose-v0` | 0.5초마다 손목에 5~15 N 외력, 목표 자세 유지 | 0% | PPO (300k) | **90%** | 161 |
| `ApproachObjects-v0` | 기존 DomainRandomizedEnv: 무작위 물체 2개에 접근, 토크 6, 동역학 랜덤화 | 15% | SAC (100k) | **25%** | 273 |

![zoo success](results_zoo/zoo_success.png)

| Push-v0 / SAC | ReachMoving-v0 / SAC | HoldPose-v0 / PPO |
|---|---|---|
| ![push](results_zoo/videos/Push-v0__SAC.gif) | ![moving](results_zoo/videos/ReachMoving-v0__SAC.gif) | ![hold](results_zoo/videos/HoldPose-v0__PPO.gif) |

![zoo success](results_zoo/zoo_success.png)

```bash
python envs/manipulator_tasks.py                      # 환경 10개 스모크 테스트
python rl_zoo/train_zoo.py --all --parallel 9         # 모델 매트릭스 학습 (CPU 약 1.5시간)
python rl_zoo/evaluate_zoo.py --episodes 20           # 전체 평가 표 + 그래프
python rl_zoo/visualize_zoo.py --run Reach-v0__SAC    # GIF (--viewer 로 MuJoCo 뷰어)
```

환경 정의, 모델 표, 사용법은 [rl_zoo/README.md](rl_zoo/README.md)에 있습니다.

## Diffusion Policy와 RL baselines

```bash
python run_pipeline.py all                           # Step 1~6 전체 실행
python train_ppo.py --total_timesteps 50000          # PPO baseline
python evaluate_compare.py --diffusion_model models/diffusion_policy.pt \
    --diffusion_norm models/normalization.json --ppo_model models/ppo_policy.pt --episodes 10
```

단계별 설명, 헤드리스 서버 실행(`run_on_server.sh` / `.bat`), 문제 해결 방법은 [README_PIPELINE.md](README_PIPELINE.md)를 참고하세요.

## 저장소 구조

```
.mujoco/
├── trajectory_planning/          # 학습 기반 궤적 계획 (학습/평가/시각화/시뮬 스크립트 + README)
├── envs/                         # Gymnasium 과제 환경 10종 (manipulator_tasks.py)
├── rl_zoo/                       # 모델 zoo 학습/평가/시각화 + README
├── models_zoo/, results_zoo/     # zoo 모델, 평가 표, 그래프, GIF
├── models_tp/, results_tp/       # 궤적 계획 모델, 평가 결과, 그래프, GIF
├── manipulator_env.py            # Gymnasium 환경 (DomainRandomizedEnv)
├── collect_trajectories.py       # Diffusion Step 2: 데이터 수집
├── train_diffusion_policy.py     # Diffusion Step 3: TemporalUNet + DDPM 학습
├── validate_policy.py            # Diffusion Step 4: 검증
├── policy_node.py                # Diffusion Step 5: receding-horizon 실행
├── train_dr_policy.py            # Diffusion Step 6: Domain Randomization 재학습
├── train_ppo.py, train_sb3_sac.py, evaluate_compare.py, record_episodes.py
├── run_pipeline.py / .sh / .bat, run_on_server.sh / .bat
├── data/, models/, models_dr/, results/, trajectories/
└── mujoco200_win64/model/yang/yang_manipulator.xml   # 로봇 모델
```

## 설치

```bash
python -m pip install -r requirements.txt pillow
```

`mujoco` 3.x, `torch`, `gymnasium`, `numpy`, `scipy`, `matplotlib`가 필요합니다. 궤적 계획 트랙은 Python 3.14, MuJoCo 3.11, PyTorch 2.13(CPU)에서 확인했습니다.
모든 스크립트는 저장소 루트에서 실행합니다.

## 시각화 스크립트

- `visualize_manipulator.py`: 학습된 PPO 정책(`models/ppo_policy.pt`)을 MuJoCo 뷰어에서 실행합니다. `--condition domain_randomized`, `--model`, `--steps` 옵션이 있고, 뷰어를 열 수 없으면 텍스트 출력으로 대신합니다.
- `visualize_policy_sb3.py`: SAC 모델(`models/sac_manipulator.zip`, `train_sb3_sac.py`로 학습)을 실행합니다. `--render`를 주면 뷰어를 띄웁니다. stable-baselines3나 모델 파일이 없으면 랜덤 행동으로 장면만 보여 줍니다.
