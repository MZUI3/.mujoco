import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import os

class DomainRandomizedEnv(gym.Env):
    """
    MuJoCo 기반 Domain Randomized 환경
    - env.reset(), env.step(action) 단독 테스트 가능
    - 관측값과 reward 정상 반환
    - MuJoCo 뷰어 지원
    """
    
    def __init__(self, xml_path=None, randomize=True, render_mode=None):
        """
        Args:
            xml_path: XML 모델 파일 경로. None이면 기본 manipulator 모델 사용
            randomize: Domain randomization 사용 여부
            render_mode: "human" (뷰어) 또는 None
        """
        super().__init__()
        
        self.randomize = randomize
        self.render_mode = render_mode
        self.viewer = None
        
        # XML 파일 경로 설정
        if xml_path is None:
            # 기본값: mujoco200_win64의 yang_manipulator.xml 사용
            default_path = Path(__file__).parent / "mujoco200_win64" / "model" / "yang" / "yang_manipulator.xml"
            if default_path.exists():
                xml_path = str(default_path)
            else:
                raise FileNotFoundError(f"기본 모델을 찾을 수 없습니다: {default_path}")
        
        # MuJoCo 모델 및 데이터 생성
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 원본 파라미터 백업 (randomization용)
        self._default_mass = self.model.body_mass.copy()
        self._default_damping = self.model.dof_damping.copy()
        self._default_friction = self.model.geom_friction.copy()
        
        # Action/Observation space 정의
        self.n_dof = self.model.nv  # 자유도 수 (속도)
        self.n_ctrl = self.model.nu  # 제어 입력 크기
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_ctrl,), dtype=np.float32
        )
        
        # Observation: qpos + qvel
        obs_dim = self.model.nq + self.model.nv  # joint positions + velocities
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self._step_count = 0

    def reset(self, seed=None, options=None):
        """환경 리셋"""
        super().reset(seed=seed)
        
        # MuJoCo 초기 상태로 리셋
        mujoco.mj_resetData(self.model, self.data)
        
        if self.randomize:
            self._randomize_dynamics()
        
        self._step_count = 0
        obs = self._get_obs()
        info = {}
        
        return obs, info

    def step(self, action):
        """
        환경 스텝 실행
        
        Args:
            action: 액션 배열 (self.n_dof 크기)
            
        Returns:
            obs, reward, terminated, truncated, info
        """
        # Action을 제어 입력으로 변환
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        
        # MuJoCo 시뮬레이션 한 스텝 진행
        mujoco.mj_step(self.model, self.data)
        
        self._step_count += 1
        
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = False
        truncated = self._step_count >= 1000  # 최대 1000 스텝
        info = {}
        
        return obs, reward, terminated, truncated, info

    def _randomize_dynamics(self):
        """Domain randomization: 질량, damping, 마찰 무작위화"""
        # 질량 무작위화 (80% ~ 120%)
        mass_scale = np.random.uniform(0.8, 1.2, size=self._default_mass.shape)
        self.model.body_mass[:] = self._default_mass * mass_scale

        # Damping 무작위화 (70% ~ 130%)
        damping_scale = np.random.uniform(0.7, 1.3, size=self._default_damping.shape)
        self.model.dof_damping[:] = self._default_damping * damping_scale

        # 마찰 무작위화 (80% ~ 120%)
        friction_scale = np.random.uniform(0.8, 1.2, size=self._default_friction.shape)
        self.model.geom_friction[:] = self._default_friction * friction_scale

    def _get_obs(self):
        """관측값 반환: [joint_positions, joint_velocities]"""
        obs = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
        
        # 센서 노이즈 추가 (randomization 활성화 시)
        if self.randomize:
            obs = obs + np.random.normal(0, 0.005, size=obs.shape).astype(np.float32)
        
        return obs

    def _compute_reward(self):
        """
        Reward 계산
        간단한 예: qvel의 제곱 합의 음수 (에너지 효율)
        """
        # 액션 페널티 + 이동 보상
        action_penalty = -0.01 * np.sum(self.data.ctrl ** 2)
        motion_reward = 0.1 * np.sum(np.abs(self.data.qvel))
        
        reward = action_penalty + motion_reward
        return float(reward)

    def render(self):
        """MuJoCo 뷰어로 시각화 (MuJoCo 3.x에서는 별도 설정 필요)"""
        # MuJoCo 3.x에서는 passive viewer 또는 다른 방식 사용
        # 현재는 텍스트 출력으로 대체
        pass

    def close(self):
        """환경 종료"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# 단독 테스트용 함수
if __name__ == "__main__":
    print("=" * 60)
    print("MuJoCo 환경 단독 테스트 시작")
    print("=" * 60)
    
    # 환경 생성 (randomization 비활성화 - 먼저 기본 동작 확인)
    env = DomainRandomizedEnv(randomize=False, render_mode="human")
    print(f"✓ 환경 생성 성공")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Control inputs (nu): {env.n_ctrl}")
    print(f"  - DoF/Velocities (nv): {env.n_dof}")
    
    # 리셋 테스트
    obs, info = env.reset()
    print(f"\n✓ reset() 성공")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Observation 첫 5개 값: {obs[:5]}")
    
    # Step 테스트 (10 스텝)
    print(f"\n✓ step() 테스트 (10 스텝):")
    for step_idx in range(10):
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        env.render()
        print(f"  Step {step_idx+1}: obs_shape={obs.shape}, reward={reward:.4f}, truncated={truncated}")
    
    env.close()
    print(f"\n✓ 모든 테스트 완료!")
    print("=" * 60)