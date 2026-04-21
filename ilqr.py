import numpy as np
import matplotlib.pyplot as plt
from src.Graph import Graph
from src.Node import VNode, FNode, FactorGraph

# ---------------------------------------------------------
# [1] Factor 노드들 (동역학, 비용 함수) 정의 (이전과 동일)
# ---------------------------------------------------------
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
# [2] 정통 iLQR 알고리즘 (이전과 동일)
# ---------------------------------------------------------
def f(x, u, dt=0.1):
    pos, vel = x
    acc = u[0] - 0.1 * vel ** 2
    next_pos = pos + vel * dt
    next_vel = vel + acc * dt
    return np.array([next_pos, next_vel])

def linearize_dynamics(x, u, dt=0.1):
    n = x.shape[0]; m = u.shape[0]; eps = 1e-5
    fx = f(x, u, dt)
    A = np.zeros((n, n)); B = np.zeros((n, m))
    for i in range(n):
        dx = x.copy(); dx[i] += eps
        A[:, i] = (f(dx, u, dt) - fx) / eps
    for j in range(m):
        du = u.copy(); du[j] += eps
        B[:, j] = (f(x, du, dt) - fx) / eps
    return A, B

def ilqr_numpy(x0, u_init, Q, R, Qf, x_goal, N, max_iter=50, tol=1e-4, dt=0.1):
    n, m = Q.shape[0], R.shape[0]
    u_traj = u_init.copy()

    for iteration in range(max_iter):
        x_traj = [x0]
        for t in range(N): x_traj.append(f(x_traj[-1], u_traj[t], dt))

        V_x = Qf @ (x_traj[N] - x_goal); V_xx = Qf.copy()
        k_list = []; K_list = []

        for t in reversed(range(N)):
            x = x_traj[t]; u = u_traj[t]
            A, B = linearize_dynamics(x, u, dt)

            Q_x = Q @ (x - x_goal) + A.T @ V_x
            Q_u = R @ u + B.T @ V_x
            Q_xx = Q + A.T @ V_xx @ A
            Q_uu = R + B.T @ V_xx @ B
            Q_ux = np.zeros((m, n)) + B.T @ V_xx @ A

            Q_uu_inv = np.linalg.inv(Q_uu + 1e-8 * np.eye(m))
            k = -Q_uu_inv @ Q_u
            K = -Q_uu_inv @ Q_ux

            V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
            V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K

            k_list.insert(0, k); K_list.insert(0, K)

        x_new = [x0]
        u_new = []
        x = x0.copy()
        for t in range(N):
            dx = x - x_traj[t]
            du = k_list[t] + K_list[t] @ dx
            u = u_traj[t] + 1.0 * du
            x = f(x, u, dt)
            x_new.append(x); u_new.append(u)

        if sum(np.linalg.norm(x_new[t] - x_traj[t]) for t in range(N+1)) < tol: break
        u_traj = u_new

    return x_new, u_traj


# ---------------------------------------------------------
# [3] 메인 최적화 실행
# ---------------------------------------------------------
def main():
    dt = 0.1
    x0 = np.array([0.0, 1.0])
    x_goal = np.array([10.0, 0.0])
    N = 30
    n_particles = 100

    Q = np.diag([1.0, 0.1])
    R = np.array([[0.01]])
    Qf = np.diag([10.0, 1.0])

    # [해결책 1] Q, R, Qf 행렬의 대각 성분을 rho 초기값으로 추출
    rho_init_X = np.diag(Q).reshape(2, 1)
    rho_init_U = np.diag(R).reshape(1, 1)
    rho_init_Xf = np.diag(Qf).reshape(2, 1)

    gamma_Q = np.linalg.inv(Q)
    gamma_R = np.linalg.inv(R)
    gamma_Qf = np.linalg.inv(Qf)
    gamma_prior = np.eye(2) * 1e-5
    gamma_dyn = np.eye(2) * 1e-4

    u_init_trj = np.zeros((N, 1))
    x_init_trj = np.zeros((N + 1, 2))
    x_init_trj[0] = x0
    for t in range(N):
        pos, vel = x_init_trj[t]
        acc = u_init_trj[t, 0] - 0.1 * vel**2
        x_init_trj[t+1] = np.array([pos + vel * dt, vel + acc * dt])

    print("\n EKI-ADMM (Scale Synchronized & EMA) 최적화 시작...")
    graph = FactorGraph()
    X_nodes, U_nodes = [], []

    # alpha_cov 제거, 순수 스케일과 분산 감소 비율로 동작
    for t in range(N + 1):
        vn_x = VNode(f"X_{t}", [2], init_z=x_init_trj[t], init_std=1.0, n_particles=n_particles, rho_max=100.0)
        X_nodes.append(vn_x)

    for t in range(N):
        vn_u = VNode(f"U_{t}", [1], init_z=u_init_trj[t], init_std=1.0, n_particles=n_particles, rho_max=100.0)
        U_nodes.append(vn_u)

    # [해결책 1] 각 노드의 성격에 맞는 rho_init 동기화 주입
    prior_factor = PriorFactor("Prior", [2], gamma_prior, x0)
    graph.connect(prior_factor, X_nodes[0], dim=2, n_particles=n_particles, rho_init=rho_init_X)

    for t in range(N):
        dyn_factor = DynamicsFactor(f"Dyn_{t}", [2], gamma_dyn, dt)
        graph.connect(dyn_factor, X_nodes[t], dim=2, n_particles=n_particles, rho_init=rho_init_X)
        graph.connect(dyn_factor, U_nodes[t], dim=1, n_particles=n_particles, rho_init=rho_init_U)
        graph.connect(dyn_factor, X_nodes[t+1], dim=2, n_particles=n_particles, rho_init=rho_init_X)

        state_cost = StateCostFactor(f"Cost_X_{t}", [2], gamma_Q, x_goal)
        graph.connect(state_cost, X_nodes[t], dim=2, n_particles=n_particles, rho_init=rho_init_X)

        ctrl_cost = ControlCostFactor(f"Cost_U_{t}", [1], gamma_R)
        graph.connect(ctrl_cost, U_nodes[t], dim=1, n_particles=n_particles, rho_init=rho_init_U)

    term_cost = StateCostFactor("Cost_X_N", [2], gamma_Qf, x_goal)
    graph.connect(term_cost, X_nodes[N], dim=2, n_particles=n_particles, rho_init=rho_init_Xf)

    # --- 2. 수렴 종료 조건 및 데이터 로깅 ---
    all_vnodes = X_nodes + U_nodes
    max_iter = 1000
    eps_abs = 1e-4
    eps_rel = 1e-3

    history_r = []
    history_s = []
    history_eps_pri = []
    history_eps_dual = []
    history_rho_avg = [] 

    print("\n" + "="*80)
    print("🚀 EKI-ADMM (Variable Metric + Residual Balancing) 최적화 시작")
    print("="*80)

    for i in range(max_iter):
        z_old = {vn: np.mean(vn.z_consensus, axis=1, keepdims=True) for vn in all_vnodes}
        
        # [수정] 디버그 로그 출력 조건 (처음 3번, 이후 10번마다)
        is_debug_step = (i < 3) or (i % 10 == 0)
        
        if is_debug_step:
            print(f"\n\n▶ Iteration: {i+1} ◀")
            
        graph.iterate(n_iter=1)
        
        r_sq, s_sq = 0.0, 0.0
        nx, nz, sum_lambda_sq = 0.0, 0.0, 0.0
        N_dim = 0
        avg_rho = 0.0
        edge_count = 0
        
        for vn in all_vnodes:
            z_mean = np.mean(vn.z_consensus, axis=1, keepdims=True)
            for edge in vn.edges:
                x_mean = np.mean(edge.local_ensemble, axis=1, keepdims=True)
                lam_mean = np.mean(edge.dual_lambda, axis=1, keepdims=True)
                rho_vec = edge.rho
                
                r_sq += np.sum((x_mean - z_mean)**2)
                s_sq += np.sum((rho_vec * (z_mean - z_old[vn]))**2)
                
                nx += np.sum(x_mean**2)
                nz += np.sum(z_mean**2)
                sum_lambda_sq += np.sum(lam_mean**2)
                N_dim += x_mean.size
                
                avg_rho += np.mean(rho_vec)
                edge_count += 1
                
        r = np.sqrt(r_sq)
        s = np.sqrt(s_sq)
        eps_pri = np.sqrt(N_dim) * eps_abs + eps_rel * max(np.sqrt(nx), np.sqrt(nz))
        eps_dual = np.sqrt(N_dim) * eps_abs + eps_rel * np.sqrt(sum_lambda_sq)
        
        history_r.append(r)
        history_s.append(s)
        history_eps_pri.append(eps_pri)
        history_eps_dual.append(eps_dual)
        history_rho_avg.append(avg_rho / edge_count)
        
        if is_debug_step:
            print(f"\n[Global Metrics] Primal Res: {r:.4f} (Tol: {eps_pri:.4f}), Dual Res: {s:.4f} (Tol: {eps_dual:.4f})")
            print(f"[Global Metrics] Average Rho: {avg_rho/edge_count:.4f}")
            print("="*80)
        
        if r < eps_pri and s < eps_dual:
            print(f"\n🎉 EKI-ADMM 조기 수렴 성공! (Iteration: {i+1}/{max_iter})")
            break

    states_eki = np.array([np.mean(n.z_consensus, axis=1) for n in X_nodes])
    controls_eki = np.array([np.mean(n.z_consensus, axis=1) for n in U_nodes])
    print(f" EKI-ADMM 완료 | 최종 상태 오차: {np.linalg.norm(states_eki[-1] - x_goal):.4f}")

    # --- 3. 정통 iLQR 실행 ---
    states_ilqr, controls_ilqr = ilqr_numpy(x0, u_init_trj, Q, R, Qf, x_goal, N)
    states_ilqr = np.array(states_ilqr)
    controls_ilqr = np.array(controls_ilqr)
    print(f" iLQR 완료 | 최종 상태 오차: {np.linalg.norm(states_ilqr[-1] - x_goal):.4f}")

    # --- 4. 시각화 ---
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.plot(states_eki[:, 0], label='EKI-ADMM Pos', lw=2)
    plt.plot(states_ilqr[:, 0], label='Standard iLQR Pos', lw=2, linestyle='--')
    plt.axhline(y=x_goal[0], color='r', linestyle=':', label='Goal')
    plt.title("Position Trajectory")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(states_eki[:, 1], label='EKI-ADMM Vel', lw=2)
    plt.plot(states_ilqr[:, 1], label='Standard iLQR Vel', lw=2, linestyle='--')
    plt.axhline(y=x_goal[1], color='r', linestyle=':', label='Goal')
    plt.title("Velocity Trajectory")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(controls_eki[:, 0], label='EKI-ADMM Ctrl', lw=2)
    plt.plot(controls_ilqr[:, 0], label='Standard iLQR Ctrl', lw=2, linestyle='--')
    plt.title("Control Input (Accel)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(np.abs(states_eki[:, 0] - states_ilqr[:, 0]), color='purple')
    plt.title("Position Difference (|EKI - iLQR|)")
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(np.abs(states_eki[:, 1] - states_ilqr[:, 1]), color='purple')
    plt.title("Velocity Difference (|EKI - iLQR|)")
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(np.abs(controls_eki[:, 0] - controls_ilqr[:, 0]), color='purple')
    plt.title("Control Difference (|EKI - iLQR|)")
    plt.grid(True)

    plt.tight_layout()
    
    # --- 5. 수렴 과정 (Residual & Rho) 시각화 ---
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history_r, label='Primal Residual (r)', color='blue', lw=2)
    plt.plot(history_eps_pri, label='Primal Tolerance', color='blue', linestyle='--', alpha=0.7)
    plt.title("Primal Residual vs Tolerance")
    plt.xlabel("Iteration")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history_s, label='Dual Residual (s)', color='red', lw=2)
    plt.plot(history_eps_dual, label='Dual Tolerance', color='red', linestyle='--', alpha=0.7)
    plt.title("Dual Residual vs Tolerance")
    plt.xlabel("Iteration")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history_rho_avg, label='Average Rho', color='green', lw=2)
    plt.title("Average Penalty (Rho) Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Rho (Scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()