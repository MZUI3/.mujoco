"""
Step 2: 데이터 수집 스크립트
========================
IK 기반 또는 랜덤/노이즈 정책으로 (obs, action) 궤적을 수집하여 저장합니다.
최소 300~500 에피소드 수집 권장.
"""

import numpy as np
import pickle
import os
from pathlib import Path
from manipulator_env import DomainRandomizedEnv
from tqdm import tqdm


class TrajectoryCollector:
    """궤적 수집 클래스"""
    
    def __init__(self, env, num_episodes=500, max_steps_per_episode=100):
        """
        Args:
            env: gym 환경
            num_episodes: 수집할 에피소드 수
            max_steps_per_episode: 에피소드당 최대 스텝
        """
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        
        # 궤적 저장소
        self.trajectories = []
    
    def random_policy(self, obs):
        """
        랜덤 정책: 액션 공간에서 무작위 샘플링
        """
        return self.env.action_space.sample()
    
    def noisy_policy(self, obs, noise_std=0.1):
        """
        노이즈 정책: 0 근처에서 노이즈를 섞은 액션
        obs를 사용하지 않는 간단한 정책
        """
        action = np.random.normal(0, noise_std, size=self.env.action_space.shape)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)
    
    def ik_based_policy(self, obs, target_pos=None):
        """
        IK 기반 정책 (간단한 예시)
        실제로는 역기구학 계산이 필요하지만, 여기서는 관절 속도 피드백 사용
        
        Args:
            obs: 관측값 [qpos, qvel]
            target_pos: 목표 위치 (없으면 현재 위치 유지)
        """
        n_q = self.env.model.nq
        n_v = self.env.model.nv
        
        qpos = obs[:n_q]
        qvel = obs[n_q:n_q+n_v]
        
        # 목표: 관절이 중립 위치(0)으로 돌아가도록
        # P 제어 + D 제어
        Kp = 0.5
        Kd = 0.1
        
        # 목표로부터의 오차
        q_error = -qpos[:self.env.n_ctrl]  # 처음 n_ctrl개 관절만
        
        # PD 제어
        action = Kp * q_error - Kd * qvel[:self.env.n_ctrl]
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        return action.astype(np.float32)
    
    def collect_episode(self, policy_type="random"):
        """
        한 에피소드 수집
        
        Args:
            policy_type: "random", "noisy", "ik"
            
        Returns:
            trajectory: {
                "obs": (T, obs_dim),
                "actions": (T, action_dim),
                "next_obs": (T, obs_dim),
                "rewards": (T,),
                "dones": (T,),
            }
        """
        trajectory = {
            "obs": [],
            "actions": [],
            "next_obs": [],
            "rewards": [],
            "dones": [],
        }
        
        obs, _ = self.env.reset()
        
        for step in range(self.max_steps_per_episode):
            # 정책에 따라 액션 선택
            if policy_type == "random":
                action = self.random_policy(obs)
            elif policy_type == "noisy":
                action = self.noisy_policy(obs, noise_std=0.15)
            elif policy_type == "ik":
                action = self.ik_based_policy(obs)
            else:
                raise ValueError(f"Unknown policy type: {policy_type}")
            
            # 한 스텝 실행
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # 궤적에 저장
            trajectory["obs"].append(obs.copy())
            trajectory["actions"].append(action.copy())
            trajectory["next_obs"].append(next_obs.copy())
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done)
            
            obs = next_obs
            
            if done:
                break
        
        # 리스트를 numpy 배열로 변환
        trajectory["obs"] = np.array(trajectory["obs"], dtype=np.float32)
        trajectory["actions"] = np.array(trajectory["actions"], dtype=np.float32)
        trajectory["next_obs"] = np.array(trajectory["next_obs"], dtype=np.float32)
        trajectory["rewards"] = np.array(trajectory["rewards"], dtype=np.float32)
        trajectory["dones"] = np.array(trajectory["dones"], dtype=bool)
        
        return trajectory
    
    def collect(self, policy_type="random"):
        """
        여러 에피소드를 수집
        
        Args:
            policy_type: "random", "noisy", "ik", 또는 "mixed"
        """
        print(f"\n{'='*60}")
        print(f"데이터 수집 시작 (정책: {policy_type})")
        print(f"목표: {self.num_episodes} 에피소드, 에피소드당 최대 {self.max_steps_per_episode} 스텝")
        print(f"{'='*60}")
        
        self.trajectories = []
        
        for ep in tqdm(range(self.num_episodes), desc="수집 중"):
            # 혼합 정책: 무작위로 정책 선택
            if policy_type == "mixed":
                current_policy = np.random.choice(["random", "noisy", "ik"])
            else:
                current_policy = policy_type
            
            trajectory = self.collect_episode(current_policy)
            self.trajectories.append(trajectory)
        
        print(f"\n✓ {len(self.trajectories)} 에피소드 수집 완료!")
    
    def save(self, filepath, format="npz"):
        """
        궤적 저장
        
        Args:
            filepath: 저장할 파일 경로
            format: "npz" 또는 "pkl"
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "npz":
            # NPZ 형식: 여러 에피소드를 하나의 파일에 저장
            data = {
                "trajectories": self.trajectories,
                "num_episodes": len(self.trajectories),
                "action_dim": self.trajectories[0]["actions"].shape[1],
                "obs_dim": self.trajectories[0]["obs"].shape[1],
            }
            np.savez_compressed(filepath, **data)
            print(f"\n✓ NPZ 형식으로 저장: {filepath}")
            print(f"  파일 크기: {filepath.stat().st_size / (1024*1024):.2f} MB")
        
        elif format == "pkl":
            # Pickle 형식
            with open(filepath, "wb") as f:
                pickle.dump(self.trajectories, f)
            print(f"\n✓ Pickle 형식으로 저장: {filepath}")
            print(f"  파일 크기: {filepath.stat().st_size / (1024*1024):.2f} MB")
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def load(self, filepath, format="npz"):
        """
        궤적 로드
        
        Args:
            filepath: 로드할 파일 경로
            format: "npz" 또는 "pkl"
        """
        filepath = Path(filepath)
        
        if format == "npz":
            data = np.load(filepath, allow_pickle=True)
            self.trajectories = list(data["trajectories"])
            print(f"\n✓ NPZ 로드: {filepath}")
            print(f"  에피소드: {len(self.trajectories)}")
        
        elif format == "pkl":
            with open(filepath, "rb") as f:
                self.trajectories = pickle.load(f)
            print(f"\n✓ Pickle 로드: {filepath}")
            print(f"  에피소드: {len(self.trajectories)}")
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def print_statistics(self):
        """수집된 궤적의 통계 출력"""
        if not self.trajectories:
            print("수집된 궤적이 없습니다!")
            return
        
        print(f"\n{'='*60}")
        print("궤적 통계")
        print(f"{'='*60}")
        
        total_steps = sum(len(traj["obs"]) for traj in self.trajectories)
        avg_episode_length = total_steps / len(self.trajectories)
        
        all_rewards = np.concatenate([traj["rewards"] for traj in self.trajectories])
        
        print(f"에피소드 수: {len(self.trajectories)}")
        print(f"총 스텝: {total_steps}")
        print(f"평균 에피소드 길이: {avg_episode_length:.1f}")
        print(f"Reward 평균: {all_rewards.mean():.4f}")
        print(f"Reward 표준편차: {all_rewards.std():.4f}")
        print(f"Reward min/max: {all_rewards.min():.4f} / {all_rewards.max():.4f}")
        
        # 관측값 통계
        all_obs = np.concatenate([traj["obs"] for traj in self.trajectories])
        print(f"\nObservation 통계:")
        print(f"  shape: {all_obs.shape}")
        print(f"  mean: {all_obs.mean():.4f}")
        print(f"  std: {all_obs.std():.4f}")
        
        # 액션 통계
        all_actions = np.concatenate([traj["actions"] for traj in self.trajectories])
        print(f"\nAction 통계:")
        print(f"  shape: {all_actions.shape}")
        print(f"  mean: {all_actions.mean():.4f}")
        print(f"  std: {all_actions.std():.4f}")


if __name__ == "__main__":
    # 환경 생성 (randomization 비활성화)
    env = DomainRandomizedEnv(randomize=False, render_mode=None)
    
    # 수집기 생성
    collector = TrajectoryCollector(
        env,
        num_episodes=500,  # 최소 300~500 권장
        max_steps_per_episode=100
    )
    
    # 데이터 수집 (혼합 정책 사용)
    collector.collect(policy_type="mixed")
    
    # 통계 출력
    collector.print_statistics()
    
    # 저장
    output_dir = Path(__file__).parent / "data"
    collector.save(output_dir / "trajectories.npz", format="npz")
    collector.save(output_dir / "trajectories.pkl", format="pkl")
    
    print("\n" + "="*60)
    print("✓ Step 2 완료: 데이터 수집 스크립트 실행 완료!")
    print("="*60)
    
    env.close()
