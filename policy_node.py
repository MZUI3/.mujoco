import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import torch
import numpy as np

class PolicyInferenceNode(Node):
    def __init__(self, policy_path, action_dim=6, plan_horizon=16, exec_horizon=8):
        super().__init__('diffusion_policy_node')

        self.model = torch.jit.load(policy_path)  # TorchScript로 export 권장
        self.model.eval()

        self.action_dim = action_dim
        self.plan_horizon = plan_horizon
        self.exec_horizon = exec_horizon
        self.action_buffer = None
        self.buffer_idx = 0

        self.latest_obs = None
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.action_pub = self.create_publisher(Float64MultiArray, '/joint_target', 10)

        control_hz = 50.0  # 실제 하드웨어 제어 주기에 맞게 조정
        self.create_timer(1.0 / control_hz, self.control_loop)

    def joint_cb(self, msg: JointState):
        self.latest_obs = np.concatenate([msg.position, msg.velocity]).astype(np.float32)

    def control_loop(self):
        if self.latest_obs is None:
            return

        if self.action_buffer is None or self.buffer_idx >= self.exec_horizon:
            self.action_buffer = self._sample_action_chunk(self.latest_obs)
            self.buffer_idx = 0

        action = self.action_buffer[:, self.buffer_idx]
        self.buffer_idx += 1

        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.action_pub.publish(msg)

    @torch.no_grad()
    def _sample_action_chunk(self, obs):
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        action_chunk = self.model(obs_t)  # 학습된 diffusion sampler를 forward에 감싸서 export
        return action_chunk.squeeze(0).numpy()


def main():
    rclpy.init()
    node = PolicyInferenceNode(policy_path="diffusion_policy.pt")
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()