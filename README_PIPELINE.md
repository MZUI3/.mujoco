# Diffusion Policy Pipeline - Automation Script

완전 자동화된 6단계 MuJoCo 조작 학습 파이프라인입니다.

## 실행 방법

### 1. Python 버전 (권장)

### Manipulator 시각화 (MuJoCo)
- MuJoCo 뷰어로 로봇을 직접 확인하려면 GUI 가능한 환경에서 아래 스크립트를 실행하세요.

```bash
python visualize_manipulator.py
```

- GUI가 없거나 viewer 호출에 실패하면 스크립트가 대체 텍스트 출력(몇 스텝의 상태/리턴)을 제공합니다.
- MuJoCo 및 OpenGL이 설치되어 있어야 뷰어가 정상 동작합니다.


#### 대화형 모드 (메뉴 선택)
```bash
python run_pipeline.py
```

#### 명령줄 모드

전체 파이프라인 실행:
```bash
python run_pipeline.py all
```

특정 Step만 실행:
```bash
python run_pipeline.py step1    # Step 1만 실행
python run_pipeline.py step2    # Step 2만 실행
python run_pipeline.py step3    # Step 3만 실행
python run_pipeline.py step4    # Step 4만 실행
python run_pipeline.py step5    # Step 5만 실행
python run_pipeline.py step6    # Step 6만 실행
```

의존성만 설치:
```bash
python run_pipeline.py deps
```

### 2. Bash 스크립트 버전 (Linux/Mac/WSL)

#### 대화형 모드
```bash
bash run_pipeline.sh
```

#### 명령줄 모드
```bash
bash run_pipeline.sh all        # 전체 실행
bash run_pipeline.sh step1      # Step 1만 실행
bash run_pipeline.sh deps       # 의존성 설치만
```

### 3. Windows Batch 파일 버전

#### 대화형 모드
```cmd
run_pipeline.bat
```

#### 명령줄 모드
```cmd
run_pipeline.bat all            # 전체 실행
run_pipeline.bat step1          # Step 1만 실행
run_pipeline.bat deps           # 의존성 설치만
```

## 파이프라인 구조

### Step 1: 환경 테스트 (Environment Testing)
- MuJoCo 환경 초기화 확인
- Gymnasium API 호환성 검증
- 10 스텝 시뮬레이션 실행
- **실행 시간**: ~10초

**파일**: `manipulator_env.py`

### Step 2: 데이터 수집 (Data Collection)
- 500 에피소드에서 궤적 수집
- 무작위, 노이즈, IK 기반 정책 사용
- 50,000개 전이 생성
- **실행 시간**: ~30초
- **생성 파일**: `data/trajectories.npz`

**파일**: `collect_trajectories.py`

### Step 3: 정책 학습 (Policy Training)
- TemporalUNet + DDPM 모델 학습
- 15 에포크 학습
- CUDA GPU 활용
- **실행 시간**: ~2-3분
- **생성 파일**: 
  - `models/diffusion_policy.pt` (모델 가중치)
  - `models/normalization.json` (정규화 파라미터)
  - `models/training_loss.png` (손실 곡선)

**파일**: `train_diffusion_policy.py`

### Step 4: 정책 검증 (Policy Validation)
- 기본 환경에서 10 에피소드 검증
- Domain Randomization 환경에서 5 에피소드 검증
- 성능 저하율 계산
- **실행 시간**: ~5-6분
- **성능 메트릭**: Mean return, Std deviation

**파일**: `validate_policy.py`

### Step 5: 정책 노드 통합 (Policy Node Integration)
- 실시간 정책 실행
- Receding-horizon control (8 스텝 계획)
- 3 에피소드 실행 및 궤적 저장
- **실행 시간**: ~1-2분
- **생성 파일**: `trajectories/episode_*.npz`

**파일**: `policy_node.py`

### Step 6: Domain Randomization 재학습
- DR 환경에서 200 에피소드 수집
- 30 에포크 재학습
- 견고성 향상
- **실행 시간**: ~5-6분
- **생성 파일**: 
  - `models_dr/diffusion_policy_dr.pt`
  - `models_dr/normalization.json`
  - `models_dr/training_loss_dr.png`

**파일**: `train_dr_policy.py`

## 요구사항

### Python 패키지
- Python 3.8+
- mujoco >= 3.0
- gymnasium >= 1.0
- numpy >= 1.20
- torch >= 2.0
- tqdm
- matplotlib

### 하드웨어
- CPU: 최소 4 코어
- RAM: 최소 8GB
- GPU (선택): CUDA 지원 GPU 권장 (학습 속도 10배 향상)

## 출력 디렉토리 구조

```
.mujoco/
├── data/
│   └── trajectories.npz              # Step 2 생성
├── models/
│   ├── diffusion_policy.pt           # Step 3 생성
│   ├── normalization.json            # Step 3 생성
│   ├── policy_epoch_10.pt            # Step 3 체크포인트
│   └── training_loss.png             # Step 3 생성
├── models_dr/
│   ├── diffusion_policy_dr.pt        # Step 6 생성
│   ├── normalization.json            # Step 6 생성
│   ├── policy_epoch_20.pt            # Step 6 체크포인트
│   ├── policy_epoch_30.pt            # Step 6 체크포인트
│   └── training_loss_dr.png          # Step 6 생성
├── trajectories/
│   ├── episode_1.npz                 # Step 5 생성
│   ├── episode_2.npz                 # Step 5 생성
│   └── episode_3.npz                 # Step 5 생성
└── run_pipeline.py                   # 메인 실행 스크립트
```

## 성능 지표

### 기본 모델 (Step 3)
- 모델 파라미터: 185,030
- 학습 손실: 0.3142 → 0.2738 (13% 감소)
- 검증 리턴: 170.46 ± 21.84

### Domain Randomization 모델 (Step 6)
- 학습 손실: 0.7609 → 0.4510 (41% 감소)
- 견고성 향상: 성능 저하율 7.2%

## 문제 해결

### 메모리 부족
- Step 3에서 메모리 부족 발생 시:
  - 배치 크기 감소: `train_diffusion_policy.py`에서 `batch_size=16`으로 수정
  - 에포크 감소: `num_epochs=10`으로 수정

### GPU 메모리 부족
- PyTorch에서 "CUDA out of memory" 발생 시:
  - CPU 모드로 실행: 스크립트 시작 전 `export CUDA_VISIBLE_DEVICES=""`
  - 배치 크기 감소

### 느린 실행 속도
- 데이터 수집 (Step 2) 가속화:
  - `collect_trajectories.py`에서 `num_episodes=250` 감소
- 학습 (Step 3) 가속화:
  - 에포크 감소 또는 배치 크기 증가

## 각 Step 간 의존성

```
Step 1 (환경 테스트)
    ↓
Step 2 (데이터 수집) ← Step 1 필요
    ↓
Step 3 (정책 학습) ← Step 2 필요
    ↓
Step 4 (정책 검증) ← Step 3 필요
    ↓
Step 5 (정책 노드) ← Step 3 필요
    ↓
Step 6 (DR 재학습) ← Step 3 필요
```

### 건너뛸 수 있는 경우
- Step 1은 선택사항 (환경 검증용)
- Step 2의 데이터가 있으면 Step 3부터 시작 가능
- Step 4, 5는 독립적으로 실행 가능 (Step 3 필요)

## 스크립트 선택 가이드

| 상황 | 권장 스크립트 |
|------|------------|
| 처음 설정 | `run_pipeline.py all` |
| 특정 Step만 | `run_pipeline.py step3` |
| 메뉴 방식 선택 | `python run_pipeline.py` |
| Linux/Mac | `bash run_pipeline.sh` |
| Windows (관리자 권한) | `run_pipeline.bat` |

## 서버 자동 실행 및 헤드리스 시각화

레포지토리에는 서버에서 전체 파이프라인을 자동으로 실행하는 스크립트를 포함합니다:

- Linux/WSL/macOS: `run_on_server.sh`
- Windows: `run_on_server.bat`

이 스크립트는 다음 단계를 순차 실행합니다: 의존성 설치 → (옵션) 시각화 → 데이터 수집 → diffusion 학습 → diffusion 검증 → PPO 학습 → 비교 평가 → 비디오 녹화(가능 시).

활성화(가상환경) 옵션
- CONDA 환경 사용 시 (권장): 실행 전에 환경 변수 `CONDA_ENV`를 설정하면 스크립트가 자동으로 conda 환경을 활성화합니다.
  - 예: `CONDA_ENV=myenv ./run_on_server.sh`
- 가상env(virtualenv/venv) 사용 시: 환경 변수 `VENV_PATH`에 가상환경 경로를 지정하면 activate 스크립트를 시도합니다.
  - 예: `VENV_PATH=/home/user/venv ./run_on_server.sh`
- Windows에서는 `set CONDA_ENV=myenv` 또는 `set VENV_PATH=C:\path\to\venv` 후 `run_on_server.bat` 실행

헤드리스(그래픽 없는 서버)에서의 시각화 및 비디오 녹화
- MuJoCo의 GUI 뷰어는 OpenGL/디스플레이가 필요합니다. 헤드리스 서버에서는 다음 중 하나를 권장합니다:
  - X 포워딩: `ssh -X user@server` (실시간 GUI 확인)
  - xvfb: `xvfb-run --auto-servernum --server-args='-screen 0 1400x900x24' python visualize_manipulator.py`
  - MuJoCo offscreen(egl/osmesa) 사용: `export MUJOCO_GL=egl` 또는 `export MUJOCO_GL=osmesa` (서버에서 지원되는 경우)
- 스크립트 `record_episodes.py`는 먼저 MuJoCo의 렌더 API를 시도하고, 실패하면 matplotlib 기반의 관절 위치 바 플롯 대체 시각화를 생성하여 비디오를 만듭니다. 따라서 헤드리스 서버에서도 동작하지 않는 환경보다 더 높은 성공률로 비디오를 생성할 수 있습니다.

성공적으로 비디오를 생성하려면 `imageio`와 `imageio-ffmpeg`가 필요합니다. run_on_server 스크립트가 설치를 시도합니다.

## 주의사항

1. **첫 실행 시간**: 전체 파이프라인 완료에 약 15-20분 소요
2. **GPU 사용**: CUDA 설치 필수 (CPU로도 가능하지만 느림)
3. **작업 중단**: Ctrl+C로 중단 가능 (다시 실행하면 Step부터 재개)
4. **재실행**: 기존 모델/데이터는 덮어쓰기됨
5. **경로**: 반드시 `.mujoco` 디렉토리에서 실행

## 추가 최적화 옵션

### Step 2 (데이터 수집) 최적화
- `num_episodes=500` → `num_episodes=1000` (더 다양한 데이터)

### Step 3 (학습) 최적화
- `num_epochs=15` → `num_epochs=30` (더 좋은 수렴)
- `batch_size=32` → `batch_size=64` (더 빠른 학습, 더 많은 메모리)

### Step 6 (DR 재학습) 최적화
- `num_episodes=200` → `num_episodes=500` (더 견고한 정책)
- `num_epochs=30` → `num_epochs=50` (더 나은 수렴)

### 추가: RL baseline — PPO (Proximal Policy Optimization)

이 저장소에는 diffusion 기반 정책 학습 파이프라인이 포함되어 있습니다. diffusion과 비교하기 위한 RL baseline(PPO)을 추가했고, 학습·검증·비교·시각화까지 쉽게 실행할 수 있도록 스크립트를 제공합니다.

특징
- 경량 PyTorch 구현 (파일: `train_ppo.py`)
- 연속 동작(continuous action)을 다루며, `DomainRandomizedEnv`에 대해 학습합니다
- 학습 중 에피소드 리턴을 `models/ppo_returns.png`로 저장
- 최종 정책 가중치는 `models/ppo_policy.pt`에 저장
- diffusion 정책과 동일한 평가 루틴을 수행하는 `evaluate_compare.py` 제공 (결과 CSV/PNG 생성)

빠른 실행 가이드 (내일 회사 서버에서 바로 실행할 수 있도록)

1) 의존성 설치 (한 번만)

```bash
python run_pipeline.py deps
# 또는
python -m pip install mujoco gymnasium numpy torch tqdm matplotlib
```

2) Manipulator 시각화 (옵션)
- GUI가 가능한 환경에서:

```bash
python visualize_manipulator.py
```

- 헤드리스(서버) 환경에서 시각화가 필요하면 다음 중 하나 사용:
  - X 포워딩: ssh -X user@server 그리고 위 명령 실행
  - xvfb (Linux): xvfb-run --auto-servernum --server-args='-screen 0 1400x900x24' python visualize_manipulator.py
  - 또는 MuJoCo가 EGL/OSMesa를 지원하면 환경변수 설정: `export MUJOCO_GL=egl` 또는 `export MUJOCO_GL=osmesa`

3) Diffusion 파이프라인 실행 (권장 순서)
- 데이터 수집 → 학습 → 검증

```bash
python run_pipeline.py step2   # collect_trajectories.py -> data/trajectories.npz
python run_pipeline.py step3   # train_diffusion_policy.py -> models/diffusion_policy.pt + training_loss.png
python run_pipeline.py step4   # validate_policy.py -> 검증 통계 출력
```

4) PPO 학습 (baseline)

```bash
python train_ppo.py --total_timesteps 50000
```

- `--total_timesteps`는 서버 자원에 맞춰 50k~200k 권장
- GPU가 있으면 자동으로 사용합니다
- 출력: `models/ppo_policy.pt`, `models/ppo_returns.png`

5) 동일 평가(비교)
- diffusion과 PPO 모델을 동일한 환경(무작위성 off)에서 평가하여 per-episode 리턴을 비교합니다.

```bash
python evaluate_compare.py \
  --diffusion_model models/diffusion_policy.pt \
  --diffusion_norm models/normalization.json \
  --ppo_model models/ppo_policy.pt \
  --episodes 10
```

- 출력: `results/compare_results.csv`, `results/compare_plot.png`

재현성 및 평가 팁
- 평가 시 무작위성 통제: `random seed`를 통일(예: 스크립트 실행 전에 PYTHONHASHSEED, numpy.random.seed, torch.manual_seed 설정)
- 동일한 평가 환경을 쓰려면 `DomainRandomizedEnv(randomize=False)`로 실행
- 데이터 예산(총 timesteps 또는 수집된 전이 수)을 같게 해서 공정 비교

결과 파일 정리(한눈에 보기)
- Diffusion
  - models/diffusion_policy.pt
  - models/training_loss.png
  - models/normalization.json
- PPO
  - models/ppo_policy.pt
  - models/ppo_returns.png
- 비교
  - results/compare_results.csv
  - results/compare_plot.png

문제 발생 시 빠른 체크
- 모델 파일이 없으면 학습 스텝(step2/step3 또는 train_ppo)이 제대로 실행됐는지 확인
- GPU 메모리 부족: 배치 크기 또는 에포크 감소
- MuJoCo viewer 관련 오류: 헤드리스일 가능성 — X 포워딩 또는 xvfb 사용

이 섹션에 언급된 모든 스크립트는 레포지토리 루트(예: `.mujoco` 디렉토리)에서 실행하세요.

## 라이센스

이 파이프라인은 교육 및 연구 목적으로 제공됩니다.

## 지원

문제 발생 시:
1. 에러 메시지 확인
2. 의존성 재설치: `python run_pipeline.py deps`
3. Step별로 문제 원인 파악
4. 로그 파일 확인 (stdout/stderr)
