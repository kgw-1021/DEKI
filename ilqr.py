import numpy as np
import matplotlib.pyplot as plt
from src.Graph import Graph
from src.Node import VNode, FNode, FactorGraph

class PriorFactor(FNode):
    def __init__(self, name: str, dims: list, gamma: np.ndarray, target_state: np.ndarray):
        super().__init__(name, dims, gamma)
        self.target_state = target_state.reshape(2, 1)

    def error_function(self, local_ensembles: list) -> np.ndarray:
        return local_ensembles[0] - self.target_state

class DynamicsFactor(FNode):
    def __init__(self, name: str, dims: list, gamma: np.ndarray, dt: float = 0.1):
        super().__init__(name, dims, gamma)
        self.dt = dt

    def error_function(self, local_ensembles: list) -> np.ndarray:
        X_t = local_ensembles[0]
        U_t = local_ensembles[1]
        X_next = local_ensembles[2]
        
        pos = X_t[0, :]
        vel = X_t[1, :]
        acc = U_t[0, :]
        
        # 비선형 동역학 방정식 (iLQR과 완벽히 동일)
        next_pos_pred = pos + vel * self.dt
        next_vel_pred = vel + (acc - 0.1 * vel**2) * self.dt
        
        X_next_pred = np.vstack([next_pos_pred, next_vel_pred])
        return X_next_pred - X_next

class StateCostFactor(FNode):
    def __init__(self, name: str, dims: list, gamma: np.ndarray, target_state: np.ndarray):
        super().__init__(name, dims, gamma)
        self.target_state = target_state.reshape(2, 1)

    def error_function(self, local_ensembles: list) -> np.ndarray:
        return local_ensembles[0] - self.target_state

class ControlCostFactor(FNode):
    def error_function(self, local_ensembles: list) -> np.ndarray:
        return local_ensembles[0]


# ---------------------------------------------------------
# [2] 정통 iLQR 알고리즘 직접 구현 (Analytical Jacobians 사용)
# ---------------------------------------------------------
#@title Classical iLQR Implmenetation (you can skip this if you want)
import numpy as np

def f(x, u, dt=0.1):
    pos, vel = x
    acc = u[0] - 0.1 * vel ** 2
    next_pos = pos + vel * dt
    next_vel = vel + acc * dt
    return np.array([next_pos, next_vel])

def linearize_dynamics(x, u, dt=0.1):
    n = x.shape[0]
    m = u.shape[0]
    eps = 1e-5
    fx = f(x, u, dt)
    A = np.zeros((n, n))
    B = np.zeros((n, m))
    for i in range(n):
        dx = x.copy()
        dx[i] += eps
        A[:, i] = (f(dx, u, dt) - fx) / eps
    for j in range(m):
        du = u.copy()
        du[j] += eps
        B[:, j] = (f(x, du, dt) - fx) / eps
    return A, B

def ilqr_numpy(x0, u_init, Q, R, Qf, x_goal, N, max_iter=50, tol=1e-4, dt=0.1):
    n, m = Q.shape[0], R.shape[0]
    u_traj = u_init.copy()

    for iteration in range(max_iter):
        # Rollout current trajectory
        x_traj = [x0]
        for t in range(N):
            x_traj.append(f(x_traj[-1], u_traj[t], dt))

        V_x = Qf @ (x_traj[N] - x_goal)
        V_xx = Qf.copy()

        k_list = []
        K_list = []

        for t in reversed(range(N)):
            x = x_traj[t]
            u = u_traj[t]
            A, B = linearize_dynamics(x, u, dt)

            dx = x - x_goal
            l_x = Q @ dx
            l_u = R @ u
            l_xx = Q
            l_uu = R
            l_ux = np.zeros((m, n))

            Q_x = l_x + A.T @ V_x
            Q_u = l_u + B.T @ V_x
            Q_xx = l_xx + A.T @ V_xx @ A
            Q_uu = l_uu + B.T @ V_xx @ B
            Q_ux = l_ux + B.T @ V_xx @ A

            Q_uu_inv = np.linalg.inv(Q_uu + 1e-8 * np.eye(m))

            k = -Q_uu_inv @ Q_u
            K = -Q_uu_inv @ Q_ux

            V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
            V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K

            k_list.insert(0, k)
            K_list.insert(0, K)

        # Forward pass
        x_new = [x0]
        u_new = []
        x = x0.copy()
        alpha = 1.0
        for t in range(N):
            dx = x - x_traj[t]
            du = k_list[t] + K_list[t] @ dx
            u = u_traj[t] + alpha * du
            x = f(x, u, dt)
            x_new.append(x)
            u_new.append(u)

        x_diff = sum(np.linalg.norm(x_new[t] - x_traj[t]) for t in range(N+1))
        if x_diff < tol:
            break
        u_traj = u_new

    return x_new, u_traj


# ---------------------------------------------------------
# [3] 메인 최적화 실행 및 결과 비교
# ---------------------------------------------------------
def main():
    dt = 0.1
    x0 = np.array([0.0, 1.0])
    x_goal = np.array([10.0, 0.0])
    N = 30
    n_particles = 1000

    # 1. 비용/공분산 설정
    Q = np.diag([1.0, 0.1])
    R = np.array([[0.01]])
    Qf = np.diag([10.0, 1.0])

    gamma_Q = np.linalg.inv(Q)
    gamma_R = np.linalg.inv(R)
    gamma_Qf = np.linalg.inv(Qf)
    
    # [수정됨] 동역학 제약(Hard Constraint) 역할을 강화하기 위해 페널티를 극대화 (1e-6)
    gamma_prior = np.eye(2) * 1e-4
    gamma_dyn = np.eye(2) * 1e-4

    # --- [수정됨] iLQR과 완전히 동일한 Forward Rollout 초기 궤적 생성 ---
    u_init_trj = np.zeros((N, 1))
    x_init_trj = np.zeros((N + 1, 2))
    x_init_trj[0] = x0
    for t in range(N):
        pos, vel = x_init_trj[t]
        acc = u_init_trj[t, 0] - 0.1 * vel**2
        x_init_trj[t+1] = np.array([pos + vel * dt, vel + acc * dt])

    # 2. EKI-ADMM 프레임워크 설정
    print(" EKI-ADMM 최적화 시작...")
    graph = FactorGraph()
    X_nodes, U_nodes = [], []

    # [수정됨] 생성된 Forward Rollout 궤적(x_init_trj)을 초기값으로 사용, 초기 노이즈(init_std)를 0.05로 대폭 감소
    for t in range(N + 1):
        vn_x = VNode(f"X_{t}", [2], init_z=x_init_trj[t], init_std=1.0, n_particles=n_particles, rho_max= 10000.0, alpha_cov = 100.0)
        X_nodes.append(vn_x)

    for t in range(N):
        vn_u = VNode(f"U_{t}", [1], init_z=u_init_trj[t], init_std=1.0, n_particles=n_particles, rho_max= 10000.0, alpha_cov = 100.0)
        U_nodes.append(vn_u)

    # 그래프 엣지 연결
    prior_factor = PriorFactor("Prior", [2], gamma_prior, x0)
    graph.connect(prior_factor, X_nodes[0], dim=2, n_particles=n_particles)

    for t in range(N):
        dyn_factor = DynamicsFactor(f"Dyn_{t}", [2], gamma_dyn, dt)
        graph.connect(dyn_factor, X_nodes[t], dim=2, n_particles=n_particles)
        graph.connect(dyn_factor, U_nodes[t], dim=1, n_particles=n_particles)
        graph.connect(dyn_factor, X_nodes[t+1], dim=2, n_particles=n_particles)

        state_cost = StateCostFactor(f"Cost_X_{t}", [2], gamma_Q, x_goal)
        graph.connect(state_cost, X_nodes[t], dim=2, n_particles=n_particles)

        ctrl_cost = ControlCostFactor(f"Cost_U_{t}", [1], gamma_R)
        graph.connect(ctrl_cost, U_nodes[t], dim=1, n_particles=n_particles)

    term_cost = StateCostFactor("Cost_X_N", [2], gamma_Qf, x_goal)
    graph.connect(term_cost, X_nodes[N], dim=2, n_particles=n_particles)

    # 최적화 수행
    for i in range(1000):
        graph.iterate(n_iter=1)
        
    states_eki = np.array([np.mean(n.z_consensus, axis=1) for n in X_nodes])
    controls_eki = np.array([np.mean(n.z_consensus, axis=1) for n in U_nodes])
    print(f"EKI-ADMM 완료 | 최종 상태 오차: {np.linalg.norm(states_eki[-1] - x_goal):.4f}")

    print(f"results | {states_eki} , {controls_eki}")

    # 3. 정통 iLQR 실행
    states_ilqr, controls_ilqr = ilqr_numpy(x0, u_init_trj, Q, R, Qf, x_goal, N)
    states_ilqr = np.array(states_ilqr)
    controls_ilqr = np.array(controls_ilqr)
    print(f" iLQR 완료 | 최종 상태 오차: {np.linalg.norm(states_ilqr[-1] - x_goal):.4f}")


    # 4. 결과 비교 시각화
    plt.figure(figsize=(15, 8))

    # 위치 비교
    plt.subplot(2, 3, 1)
    plt.plot(states_eki[:, 0], label='EKI-ADMM Pos', lw=2)
    plt.plot(states_ilqr[:, 0], label='Standard iLQR Pos', lw=2, linestyle='--')
    plt.axhline(y=x_goal[0], color='r', linestyle=':', label='Goal')
    plt.title("Position Trajectory")
    plt.legend()
    plt.grid(True)

    # 속도 비교
    plt.subplot(2, 3, 2)
    plt.plot(states_eki[:, 1], label='EKI-ADMM Vel', lw=2)
    plt.plot(states_ilqr[:, 1], label='Standard iLQR Vel', lw=2, linestyle='--')
    plt.axhline(y=x_goal[1], color='r', linestyle=':', label='Goal')
    plt.title("Velocity Trajectory")
    plt.legend()
    plt.grid(True)

    # 제어 입력 비교
    plt.subplot(2, 3, 3)
    plt.plot(controls_eki[:, 0], label='EKI-ADMM Ctrl', lw=2)
    plt.plot(controls_ilqr[:, 0], label='Standard iLQR Ctrl', lw=2, linestyle='--')
    plt.title("Control Input (Accel)")
    plt.legend()
    plt.grid(True)

    # 위치 오차(차이)
    plt.subplot(2, 3, 4)
    plt.plot(np.abs(states_eki[:, 0] - states_ilqr[:, 0]), color='purple')
    plt.title("Position Difference (|EKI - iLQR|)")
    plt.grid(True)

    # 속도 오차(차이)
    plt.subplot(2, 3, 5)
    plt.plot(np.abs(states_eki[:, 1] - states_ilqr[:, 1]), color='purple')
    plt.title("Velocity Difference (|EKI - iLQR|)")
    plt.grid(True)

    # 제어 오차(차이)
    plt.subplot(2, 3, 6)
    plt.plot(np.abs(controls_eki[:, 0] - controls_ilqr[:, 0]), color='purple')
    plt.title("Control Difference (|EKI - iLQR|)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()