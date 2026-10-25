# 플래너 ↔ RL 브리지: `ReachMultiObstacle-v0`

같은 20문제(평가 seed 10000~, 기둥 수 1/2/3개 = 12/4/4), 결정적 정책, 200 step(2 s). 성공 = 마지막 step에서 목표 3 cm 이내 **그리고** 에피소드 내 팔-기둥/바닥 접촉 0회.

| 정책 | 학습 데이터 / 예산 | 성공률 | 충돌률 | 최종 거리 (cm) | 평균 리턴 | 비고 |
|---|---|---|---|---|---|---|
| 무작위 행동 | - | 0% | 90% | 50.9 | -255.1 ± 61.4 | 기준선 |
| SAC 100k | 100k env steps, 21 min | **0%** | 15% | 11.0 | -7.1 ± 47.8 | `models_zoo/ReachMultiObstacle-v0__SAC` |
| BC (순수 행동 복제, obs만) | 데모 1997 에피소드 / 399k 전이, 1 min | **0%** | 40% | 33.8 | -211.2 ± 73.7 | `models_zoo/ReachMultiObstacle-v0__BC_naive`, MLP [256, 256], 입력 obs만 |
| BC + DAgger (플래너 데모 + 전문가 라벨링) | 데모 1997 에피소드 / 399k 전이 + DAgger 8×400 에피소드 라벨링, 14 min | **25%** | 10% | 4.7 | -159.3 ± 51.2 | `models_zoo/ReachMultiObstacle-v0__BC`, MLP [256, 256], 입력 obs + 시간 + q0 |
| 학습 플래너 (SimScene.rollout, 참고) | 데모 없음 (self-supervised) | 100% | 0% | 0.0 | - | 2 ms computed-torque, 0.5 s settle; 플래너 기준(2 cm) 성공 100%, 계획 70 ms/문제 |
| 학습 플래너 + computed torque (env.step, 데모 생성기) | 데모 없음 (self-supervised) | 100% | 0% | 0.0 | -120.0 ± 46.8 | 10 ms 제어, BC 데모와 같은 추종기 = BC의 상한 |

참고: 기둥 1개 고정 크기(4 cm × 30 cm)인 `ReachObstacle-v0`에서 SAC 300k는 성공 30%, 충돌 30% 였습니다 (`results_zoo/zoo_summary.md`).

재현:

```bash
python rl_zoo/collect_planner_demos.py                               # 플래너 데모 2000문제 -> data/planner_demos.npz
python rl_zoo/train_bc.py                                            # BC + DAgger -> models_zoo/ReachMultiObstacle-v0__BC
python rl_zoo/train_bc.py --dagger 0 --no-context --run ReachMultiObstacle-v0__BC_naive --algo-label BC_naive   # 순수 BC 대조군
python rl_zoo/train_zoo.py --env ReachMultiObstacle-v0 --algo SAC    # SAC 100k
python rl_zoo/evaluate_bridge.py --episodes 20                           # 이 표
python rl_zoo/visualize_zoo.py --run ReachMultiObstacle-v0__BC       # GIF
```
