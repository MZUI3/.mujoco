# Diffusion Policy Pipeline - Automation Script

완전 자동화된 6단계 MuJoCo 조작 학습 파이프라인입니다.

## 실행 방법

### 1. Python 버전 (권장)

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

## 라이센스

이 파이프라인은 교육 및 연구 목적으로 제공됩니다.

## 지원

문제 발생 시:
1. 에러 메시지 확인
2. 의존성 재설치: `python run_pipeline.py deps`
3. Step별로 문제 원인 파악
4. 로그 파일 확인 (stdout/stderr)
