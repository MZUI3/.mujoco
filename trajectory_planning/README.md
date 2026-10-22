# Learned Trajectory Planning (MuJoCo)

`yang_manipulator.xml` 로봇팔(4 DoF 팔 + 그리퍼)이 **시작 관절각 → 목표 위치**까지
**장애물(기둥)·바닥·베이스와 충돌 없이** 이동하는 관절 궤적을 신경망이 한 번의 forward로 생성하도록 학습하고,
MuJoCo 동역학 시뮬레이션에서 성능을 평가·시각화합니다.

## 문제 정의

| 입력 | 출력 |
|---|---|
| 시작 관절각 `q0` (4), 목표 EE 위치 `goal` (xyz), 장애물 위치 `obstacle` (xy) | 2초 길이 관절 궤적 `q(t)` |

- 궤적 표현: clamped cubic **B-spline** (제어점 10개). `c0=c1=q0`, `c8=c9=q_end`로 고정해서 시작·끝 속도가 0입니다.
- 장애물: 반지름 4 cm, 높이 30 cm 기둥. 무작위 문제의 절반은 "관절공간 직선 경로" 위에 놓여 단순 경로를 실제로 가로막습니다.
- 문제 샘플링: 시작 자세와 목표 자세(목표 위치를 만드는 자세)가 둘 다 충돌이 없는 경우만 사용합니다. 그래서 모든 문제는 풀 수 있습니다.

## 학습 방법 (self-supervised, 데모 불필요)

1. MuJoCo 모델에서 기구학 체인과 충돌 구(geom → sphere 근사)를 자동으로 뽑아 **미분 가능한 순기구학(PyTorch)**을 만듭니다.
   MuJoCo `xpos`와의 오차는 1e-7 m 수준입니다 (`python trajectory_planning/tp_core.py`로 확인).
2. 궤적 비용 = 목표 오차 + 충돌 침투(안전여유 2 cm, 장애물/베이스/바닥 SDF) + 관절 한계 + 가속도 + 속도.
3. 목표 위치의 **충돌 인지 IK 해 `q_ik`**(DLS IK + 랜덤 재시작, 시작 자세에 가장 가까운 충돌 없는 해)를 먼저 구합니다.
4. `PlannerNet`(MLP 512×4)은 `(q0, goal, obstacle, q_ik)`를 받아 `q_end` 보정량과 **경유 제어점 오프셋**(장애물 우회)을 출력합니다.
   `q_end`는 **미분 가능한 DLS-IK 층**(해석적 자코비안, 5 스텝)을 거쳐 목표에 정확히 붙습니다.
   출력층이 0으로 초기화되어 있어서, 학습 전 네트워크는 `linear` 기준선과 같은 궤적을 냅니다. 학습하면서 충돌을 피하도록 경로를 휘게 됩니다.
5. 문제 10만 개와 그 IK 해를 미리 계산한 풀에서 512개씩 배치를 뽑아 비용을 직접 최소화합니다(3000 iter).

> 설계 메모: 처음에는 MLP가 IK까지 직접 출력하게 했는데, 목표 정밀도가 cm 수준에 머물렀습니다(성공률 약 5%).
> IK 층을 넣은 뒤에도 베이스를 관통하는 IK 분기에 갇히는 문제가 남았습니다(성공률 약 60%). IK 분기 선택은
> 이산적인 문제라 경사하강으로 넘어가기 어렵습니다. 그래서 분기는 충돌 인지 IK로 정하고, 네트워크는 경로 형상을 학습하도록 역할을 나눴습니다.

## 비교 방법 (평가)

| 이름 | 설명 |
|---|---|
| `learned` | 목표 IK + 신경망 1회 forward |
| `learned+refine` | 신경망 출력을 초기값으로 같은 비용을 30 step 최적화 |
| `trajopt` | 문제마다 같은 비용을 처음부터 최적화 (초기값 4개 × 400 step). 품질 기준선 |
| `linear` | 충돌 인지 IK 후 관절공간 직선 보간 (경로상 장애물 무시). 하한 기준선 |

모든 궤적은 MuJoCo에서 **computed-torque 제어기**(질량행렬 + 중력/코리올리 보상 + PD, 액추에이터 한계로 클리핑)로 추종합니다.
지표는 **실제 시뮬레이션 결과**로 계산합니다.

- **success**: 최종 EE 오차 < 2 cm 이고 로봇(링크2 이후)이 장애물·바닥·베이스와 한 번도 접촉하지 않음
- final EE error, 장애물 최소거리(`mj_geomDistance`), EE 경로 길이, 계획 궤적의 적분 제곱 jerk, 추종 RMSE, 토크 포화 비율
- planning latency: 문제 1개를 단독으로 계획할 때 걸린 시간

## 실행

저장소 루트(`.mujoco/`)에서 실행합니다. 의존성: `mujoco>=3.2 torch numpy scipy matplotlib pillow`

```bash
# 1) 학습 (CPU 약 15분). 빠른 확인은 --iters 300 --pool 8192
python trajectory_planning/train_planner.py
# 2) 평가: 테스트 문제 × 4개 방법을 MuJoCo로 롤아웃 (100문제 약 4분, 아래 결과는 --n 300)
python trajectory_planning/evaluate_planner.py --n 100
# 3) 시각화: 방법별 화면을 나란히 붙인 GIF (headless OK)
python trajectory_planning/visualize_planner.py --start 7 --n 1 --methods learned linear
# 3') 인터랙티브 MuJoCo 뷰어로 실시간 재생 (GUI 필요)
python trajectory_planning/visualize_planner.py --viewer --methods learned trajopt
```

`evaluate_planner.py`와 `visualize_planner.py`는 기본 seed(2026)가 같아서 같은 테스트 문제를 사용합니다.
`--start k`로 평가 CSV의 `problem=k` 문제를 바로 영상으로 볼 수 있습니다(예: #1, #7, #11은 직선 경로가 실패하는 문제).
샘플러와 IK는 결정적이라서 문제 개수나 배치 크기가 달라도 같은 문제에 같은 궤적이 나옵니다.

## 결과 (MuJoCo 동역학 롤아웃, 테스트 300문제, seed 2026)

| method | success | collision | median err (mm) | median clearance (cm) | median jerk | track RMSE (deg) | latency (ms) |
|---|---|---|---|---|---|---|---|
| learned | **100.0%** | 0.0% | 0.13 | 7.81 | 122.7 | 0.09 | **66.7** |
| learned+refine | 100.0% | 0.0% | 0.88 | 5.03 | 93.9 | 0.09 | 469.8 |
| trajopt | 100.0% | 0.0% | 0.81 | 5.41 | 97.4 | 0.08 | 5685.2 |
| linear | 70.3% | 29.7% | 0.12 | 4.66 | 2093.2 | 3.54 | 82.8 |

- 학습 플래너는 직선 경로가 실패하는 30%의 문제를 모두 풀었습니다. 문제별 최적화(trajopt)와 같은 성공률을 **약 85배 빠르게** 냅니다.
  learned 지연시간의 대부분은 목표 IK(약 65 ms)이고, 네트워크 forward는 약 1 ms입니다.
- 학습 플래너는 trajopt보다 장애물과 거리를 더 두고(중앙값 7.8 cm 대 5.4 cm) 경로가 약간 깁니다. `learned+refine`이 그 차이를 줄여 줍니다.
- 학습 곡선: 학습 전(iter 0, 직선 경로와 동일) 검증 성공률 70.3%에서 100 iter 만에 99%까지 올라갑니다(`models_tp/training_curves.png`).
- 테스트 문제는 학습과 같은 분포(장애물 1개, 고정 크기)에서 뽑았습니다. 분포 밖 상황(장애물 여러 개, 다른 크기)에서는 검증하지 않았습니다.

## 출력물

| 경로 | 내용 |
|---|---|
| `models_tp/planner.pt` | 검증 성공률 기준 best 체크포인트 (`planner_last.pt`는 마지막) |
| `models_tp/training_curves.png`, `train_history.json`, `train_log.txt` | 비용 항목별 학습 곡선, 검증 성공률·목표 오차 |
| `results_tp/eval_summary.md`, `eval_summary.json`, `eval_results.csv` | 방법별 요약 / 문제×방법 단위 원시 지표 |
| `results_tp/summary_rates.png` | 성공률·목표 도달률·충돌률 |
| `results_tp/distributions.png` | 오차, 장애물 거리, 경로 길이, jerk, 추종 오차, 계획 시간 분포 |
| `results_tp/example_paths_3d.png` | 직선 경로가 실패한 문제에서 방법별 실제 EE 경로 (3D) |
| `results_tp/tracking_example.png` | 계획 vs MuJoCo 실제 관절각 |
| `results_tp/videos/*.gif` | MuJoCo 렌더링 (얇은 선 = 계획 경로, 굵은 선 = 실제 경로) |

## 파일 구성

- `tp_core.py`: 씬 구성(MjSpec으로 기존 XML 로드 → 자유 물체 제거, 기둥 장애물·목표 마커 추가), 미분 가능 FK/자코비안, B-spline, 비용, 문제 샘플러, 네트워크, 플래너들
- `tp_sim.py`: MuJoCo 추종 제어기, 접촉/거리 지표, 렌더링 헬퍼
- `train_planner.py`, `evaluate_planner.py`, `visualize_planner.py`

## 한계와 확장 아이디어

- 장애물은 기둥 1개이고 크기가 고정이며, 목표는 위치만 있습니다(방향 없음). 장애물 개수·크기를 입력에 넣거나
  occupancy/point-cloud 인코더를 붙이면 일반화할 수 있습니다.
- 회귀형 MLP는 문제마다 해를 하나만 냅니다(좌/우 우회 같은 다중 해 중 하나). Diffusion/CVAE 플래너로 여러 후보를 샘플링하고
  비용으로 고르면 성공률을 더 올릴 수 있습니다. 기존 `train_diffusion_policy.py`와 연결할 수 있는 지점입니다.
- `learned+refine`처럼 학습 플래너를 최적화의 warm-start로 쓰면 짧은 시간 안에 trajopt 수준 품질에 가까워집니다.
