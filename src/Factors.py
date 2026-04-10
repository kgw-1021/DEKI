from src.Node import VNode, FNode
import numpy as np
from abc import *

class DynamicsFactor(FNode):
    def __init__(self, name: str, dims: list) -> None:
        super().__init__(name, dims)

    @abstractmethod
    def dynamics(self, x_prev):
        px, py, theta, v = x_prev
        pred_px = px + v * np.cos(theta) * self.dt
        pred_py = py + v * np.sin(theta) * self.dt
        pred_theta = theta
        pred_v = v

        pred_next = np.stack([pred_px, pred_py, pred_theta, pred_v], axis=0)
        return error

    
    def error_function(self, x_prev, x_curr):
        predicted = self.dynamics(x_prev)  # 
        error = x_curr - predicted
        return error
    
class ObstacleFactor(FNode):
    def __init__(self, name: str, dims: list, obstacle_center: np.ndarray, obstacle_radius: float) -> None:
        super().__init__(name, dims)
        self.obstacle_center = obstacle_center
        self.obstacle_radius = obstacle_radius

    def error_function(self, x_curr):
        distance = np.linalg.norm(x_curr - self.obstacle_center)
        error = max(0, self.obstacle_radius - distance)  # 장애물 반경 내에 있을 때만 에러 발생
        return error
    
class StartFactor(FNode):
    def __init__(self, name: str, dims: list, start_state: np.ndarray) -> None:
        super().__init__(name, dims)
        self.start_state = start_state

    def error_function(self, x_curr):
        error = x_curr - self.start_state
        return error
    
class GoalFactor(FNode):
    def __init__(self, name: str, dims: list, goal_state: np.ndarray) -> None:
        super().__init__(name, dims)
        self.goal_state = goal_state

    def error_function(self, x_curr):
        error = x_curr - self.goal_state
        return error

class CollisionFactor(FNode):
    """분산형 충돌 회피 팩터 (공유 메모리 활용)"""

    def __init__(self, name: str, safe_dist: float, weight: float = 1e-2):
        gamma = np.eye(1) * weight
        super().__init__(name, dims=[1], gamma=gamma)
        self.safe_dist = safe_dist
        self.other_pos_mean = None

    def _error_function(self, x: np.ndarray) -> np.ndarray:
        if self.other_pos_mean is None:
            return np.zeros((1, x.shape[1]))

        my_pos = x[:2, :]
        target_pos = self.other_pos_mean[:2].reshape(2, 1)

        dists = np.linalg.norm(my_pos - target_pos, axis=0)
        dists = np.maximum(dists, 1e-6)

        error = np.maximum(0.0, self.safe_dist - dists)
        return error.reshape(1, -1)

class VelocityConstraintFNode(FNode):
    """속도 한계 제약"""

    def __init__(
        self,
        name: str,
        v_max: float = 0.1,
        v_min: float = -0.05,
        weight: float = 1e-4,
    ):
        gamma = np.eye(1) * weight
        super().__init__(name, dims=[1], gamma=gamma)
        self.v_max = v_max
        self.v_min = v_min

    def _error_function(self, x: np.ndarray) -> np.ndarray:
        v = x[3, :]

        # [변경] 부호 통일: 두 항 모두 위반량을 양수로 표현
        # 기존: err_lower = minimum(0, v - v_min) → 음수, abs로 뒤집음
        # 수정: err_lower = maximum(0, v_min - v) → 처음부터 양수
        # 의미가 동일하지만 코드 의도가 명확해지고 부호 실수 방지
        err_upper = np.maximum(0.0, v - self.v_max)   # v > v_max 위반량
        err_lower = np.maximum(0.0, self.v_min - v)   # v < v_min 위반량
        error = err_upper + err_lower
        return error.reshape(1, -1)
    
class ControlSmoothnessFNode(FNode):
    """가속도/각속도 제한 및 궤적 부드러움 제약"""

    def __init__(
        self,
        name: str,
        dt: float = 0.1,
        w_smooth: float = 1e-1,
        w_limit: float = 1e-4,
    ):
        # [변경] 잔차 벡터를 4차원 → 2차원으로 축소
        # 기존: [accel, omega, accel_lim, omega_lim] 4차원
        # 수정: [accel, omega] 2차원 (평활화 항만 유지)
        #
        # 기존 구현에서 평활화 항(accel, omega)과 한계 항(accel_lim, omega_lim)이
        # 같은 물리량에서 파생된 중복 정보였습니다.
        # w_smooth=1e-1이 w_limit=1e-4보다 1000배 크므로 한계 항은
        # 사실상 EKI 교차공분산에 미치는 영향이 무시되었습니다.
        # 한계 초과 페널티는 VelocityConstraintFNode가 별도로 담당하므로
        # 여기서는 평활화만 담당하도록 역할을 분리합니다.
        gamma = np.diag([w_smooth, w_smooth])
        super().__init__(name, dims=[4], gamma=gamma)

        self.dt = dt
        self.accel_max = 1.0
        self.omega_max = 1.0

    def _error_function(self, x_prev: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        dt = self.dt
        v_prev, v_next = x_prev[3, :], x_next[3, :]
        accel = (v_next - v_prev) / dt

        theta_prev, theta_next = x_prev[2, :], x_next[2, :]
        d_theta = (theta_next - theta_prev + np.pi) % (2 * np.pi) - np.pi
        omega = d_theta / dt

        # [변경] 평활화 항만 반환 (한계 항 제거)
        # 가속도/각속도 자체를 0에 가깝게 만드는 것이 이 팩터의 역할
        error = np.stack([err_accel := accel, err_omega := omega], axis=0)
        return error