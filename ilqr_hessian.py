import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# [1] Hessian-based ADMM 프레임워크 임포트
# ---------------------------------------------------------
from src.Graph_hessian import Graph, Node, Edge
from src.Node_hessian import VNode, FNode, FactorGraph 

# ---------------------------------------------------------
# [2] Factor Nodes 정의 (자코비안 함수 포함, 수정 불필요)
# ---------------------------------------------------------
class PriorFactor(FNode):
    def __init__(self, name: str, dims: list, gamma: np.ndarray, target_state: np.ndarray):
        super().__init__(name, dims, gamma)
        self.target_state = target_state.reshape(2, 1)

    def error_function(self, local_xs: list) -> np.ndarray:
        return local_xs[0] - self.target_state

    def jacobian_function(self, local_xs: list) -> list:
        return [np.eye(2)]

class DynamicsFactor(FNode):
    def __init__(self, name: str, dims: list, gamma: np.ndarray, dt: float = 0.1):
        super().__init__(name, dims, gamma)
        self.dt = dt

    def error_function(self, local_xs: list) -> np.ndarray:
        X_t = local_xs[0]
        U_t = local_xs[1]
        X_next = local_xs[2]
        
        pos = X_t[0, 0]
        vel = X_t[1, 0]
        acc = U_t[0, 0]
        
        next_pos_pred = pos + vel * self.dt
        next_vel_pred = vel + (acc - 0.1 * vel**2) * self.dt
        
        X_next_pred = np.array([[next_pos_pred], [next_vel_pred]])
        return X_next_pred - X_next

    def jacobian_function(self, local_xs: list) -> list:
        X_t = local_xs[0]
        vel = X_t[1, 0]
        
        J_Xt = np.array([
            [1.0, self.dt],
            [0.0, 1.0 - 0.2 * vel * self.dt]
        ])
        
        J_Ut = np.array([
            [0.0],
            [self.dt]
        ])
        
        J_Xnext = np.array([
            [-1.0, 0.0],
            [0.0, -1.0]
        ])
        
        return [J_Xt, J_Ut, J_Xnext]

class StateCostFactor(FNode):
    def __init__(self, name: str, dims: list, gamma: np.ndarray, target_state: np.ndarray):
        super().__init__(name, dims, gamma)
        self.target_state = target_state.reshape(2, 1)

    def error_function(self, local_xs: list) -> np.ndarray:
        return local_xs[0] - self.target_state

    def jacobian_function(self, local_xs: list) -> list:
        return [np.eye(2)]

class ControlCostFactor(FNode):
    def error_function(self, local_xs: list) -> np.ndarray:
        return local_xs[0]

    def jacobian_function(self, local_xs: list) -> list:
        return [np.eye(1)]


# ---------------------------------------------------------
# [3] 정통 iLQR 알고리즘
# ---------------------------------------------------------
def compute_cost(x_trj, u_trj, x_goal, N, Q, R, Qf):
    cost = 0.5 * (x_trj[N] - x_goal).T @ Qf @ (x_trj[N] - x_goal)
    for t in range(N):
        cost += 0.5 * (x_trj[t] - x_goal).T @ Q @ (x_trj[t] - x_goal)
        cost += 0.5 * u_trj[t].T @ R @ u_trj[t]
    return cost

def run_standard_ilqr(x0, x_goal, N, Q, R, Qf, dt=0.1, max_iter=100):
    u_trj = np.zeros((N, 1))
    x_trj = np.zeros((N + 1, 2))
    x_trj[0] = x0
    
    for t in range(N):
        pos, vel = x_trj[t]
        acc = u_trj[t, 0] - 0.1 * vel**2
        x_trj[t+1] = np.array([pos + vel * dt, vel + acc * dt])
        
    for i in range(max_iter):
        V_x = Qf @ (x_trj[N] - x_goal)
        V_xx = Qf
        
        K_gains = np.zeros((N, 1, 2))
        k_gains = np.zeros((N, 1))
        
        for t in range(N - 1, -1, -1):
            xt = x_trj[t]
            ut = u_trj[t]
            
            A = np.array([[1.0, dt], [0.0, 1.0 - 0.2 * xt[1] * dt]])
            B = np.array([[0.0], [dt]])
            
            lx = Q @ (xt - x_goal)
            lu = R @ ut
            Q_x = lx + A.T @ V_x
            Q_u = lu + B.T @ V_x
            Q_xx = Q + A.T @ V_xx @ A
            Q_uu = R + B.T @ V_xx @ B
            Q_ux = np.zeros((1, 2)) + B.T @ V_xx @ A
            
            Q_uu_inv = np.linalg.inv(Q_uu + 1e-6 * np.eye(1))
            k_gains[t] = -Q_uu_inv @ Q_u
            K_gains[t] = -Q_uu_inv @ Q_ux
            
            V_x = Q_x + K_gains[t].T @ Q_uu @ k_gains[t] + K_gains[t].T @ Q_u + Q_ux.T @ k_gains[t]
            V_xx = Q_xx + K_gains[t].T @ Q_uu @ K_gains[t] + K_gains[t].T @ Q_ux + Q_ux.T @ K_gains[t]
            V_xx = 0.5 * (V_xx + V_xx.T)
            
        current_cost = compute_cost(x_trj, u_trj, x_goal, N, Q, R, Qf)
        x_new = np.zeros((N + 1, 2))
        u_new = np.zeros((N, 1))
        
        for alpha in [1.0, 0.5, 0.1, 0.05, 0.01]:
            x_new[0] = x0
            for t in range(N):
                u_new[t] = u_trj[t] + alpha * k_gains[t] + K_gains[t] @ (x_new[t] - x_trj[t])
                pos, vel = x_new[t]
                acc = u_new[t, 0] - 0.1 * vel**2
                x_new[t+1] = np.array([pos + vel * dt, vel + acc * dt])
            if compute_cost(x_new, u_new, x_goal, N, Q, R, Qf) < current_cost:
                break
                
        x_trj = x_new
        u_trj = u_new
        
    return x_trj, u_trj

def compute_total_cost(states, controls, Q, R, Qf, x_goal):
    N = len(controls)
    total_cost = 0.0
    for t in range(N):
        x = states[t]
        u = controls[t]
        x_error = x - x_goal
        total_cost += 0.5 * x_error.T @ Q @ x_error
        total_cost += 0.5 * u.T @ R @ u
    xN_error = states[N] - x_goal
    total_cost += 0.5 * xN_error.T @ Qf @ xN_error
    return total_cost

# ---------------------------------------------------------
# [4] 메인 최적화 실행 및 결과 비교
# ---------------------------------------------------------
def main():
    dt = 0.1
    x0 = np.array([0.0, 1.0])
    x_goal = np.array([10.0, 0.0])
    N = 30

    Q = np.diag([1.0, 0.1])
    R = np.array([[0.01]])
    Qf = np.diag([10.0, 1.0])

    gamma_Q = np.linalg.inv(Q)
    gamma_R = np.linalg.inv(R)
    gamma_Qf = np.linalg.inv(Qf)
    gamma_prior = np.eye(2) * 1e-6
    gamma_dyn = np.eye(2) * 1e-6

    # 공통 초기 궤적 생성
    u_init_trj = np.zeros((N, 1))
    x_init_trj = np.zeros((N + 1, 2))
    x_init_trj[0] = x0
    for t in range(N):
        pos, vel = x_init_trj[t]
        acc = u_init_trj[t, 0] - 0.1 * vel**2
        x_init_trj[t+1] = np.array([pos + vel * dt, vel + acc * dt])

    # --- 1. 정통 iLQR Baseline ---
    print("\n iLQR (Line Search) 최적화 시작...")
    states_ilqr, controls_ilqr = run_standard_ilqr(x0, x_goal, N, Q, R, Qf, dt)
    print(f"iLQR 완료 | 최종 상태 오차: {np.linalg.norm(states_ilqr[-1] - x_goal):.4f}")

    # --- 2. Hessian-based ADMM (Factor Graph) 초기화 ---
    print("\n Hessian-based ADMM 최적화 시작...")
    graph = FactorGraph()
    X_nodes, U_nodes = [], []

    # 수정됨: VNode에서 mu_res, tau_res 제거됨
    for t in range(N + 1):
        vn_x = VNode(f"X_{t}", [2], init_z=x_init_trj[t])
        X_nodes.append(vn_x)

    for t in range(N):
        vn_u = VNode(f"U_{t}", [1], init_z=u_init_trj[t])
        U_nodes.append(vn_u)

    prior_factor = PriorFactor("Prior", [2], gamma_prior, x0)
    graph.connect(prior_factor, X_nodes[0], dim=2, rho_init=1.0, init_val=x_init_trj[0])

    for t in range(N):
        dyn_factor = DynamicsFactor(f"Dyn_{t}", [2], gamma_dyn, dt)
        graph.connect(dyn_factor, X_nodes[t], dim=2, rho_init=1.0, init_val=x_init_trj[t])
        graph.connect(dyn_factor, U_nodes[t], dim=1, rho_init=1.0, init_val=u_init_trj[t])
        graph.connect(dyn_factor, X_nodes[t+1], dim=2, rho_init=1.0, init_val=x_init_trj[t+1])

        state_cost = StateCostFactor(f"Cost_X_{t}", [2], gamma_Q, x_goal)
        graph.connect(state_cost, X_nodes[t], dim=2, rho_init=1.0, init_val=x_init_trj[t])

        ctrl_cost = ControlCostFactor(f"Cost_U_{t}", [1], gamma_R)
        graph.connect(ctrl_cost, U_nodes[t], dim=1, rho_init=1.0, init_val=u_init_trj[t])

    term_cost = StateCostFactor("Cost_X_N", [2], gamma_Qf, x_goal)
    graph.connect(term_cost, X_nodes[N], dim=2, rho_init=1.0, init_val=x_init_trj[N])


    # ---------------------------------------------------------
    # [수정] ADMM 수렴 판정 로직 및 데이터 로깅 (Hessian Matrix P 대응)
    # ---------------------------------------------------------
    eps_abs = 1e-5  
    eps_rel = 1e-4  
    max_iter = 1000  
    
    all_vnodes = X_nodes + U_nodes

    history_r = []
    history_s = []
    history_eps_pri = []
    history_eps_dual = []
    history_P_mean = [] # 수정: rho 대신 P 행렬의 크기 추적

    for i in range(max_iter):
        z_old = {vn: vn.z.copy() for vn in all_vnodes}
        
        graph.iterate()
        
        r_sq, s_sq = 0.0, 0.0
        nx, nz = 0.0, 0.0
        sum_lambda_sq = 0.0
        N_dim = 0
        
        for vn in all_vnodes:
            for edge in vn.edges:
                # Primal Residual
                r_sq += np.sum((edge.local_x - vn.z) ** 2)
                
                # Dual Residual (수정됨: edge.rho 스칼라 곱 대신 행렬 edge.P 곱셈)
                s_vec = edge.P @ (vn.z - z_old[vn])
                s_sq += np.sum(s_vec ** 2)
                
                nx += np.sum(edge.local_x ** 2)
                nz += np.sum(vn.z ** 2)
                sum_lambda_sq += np.sum(edge.dual_lambda ** 2)
                N_dim += edge.local_x.size
                
        r = np.sqrt(r_sq)
        s = np.sqrt(s_sq)
        
        eps_pri = np.sqrt(N_dim) * eps_abs + eps_rel * max(np.sqrt(nx), np.sqrt(nz))
        eps_dual = np.sqrt(N_dim) * eps_abs + eps_rel * np.sqrt(sum_lambda_sq)
        
        history_r.append(r)
        history_s.append(s)
        history_eps_pri.append(eps_pri)
        history_eps_dual.append(eps_dual)
        
        # 첫 번째 X 노드의 첫 번째 엣지의 P 행렬 대각 성분 평균 기록
        history_P_mean.append(np.mean(np.diag(all_vnodes[0].edges[0].P)))
        
        if r < eps_pri and s < eps_dual:
            print(f"ADMM 수렴 성공! (Iteration: {i+1}/{max_iter})")
            print(f"- Primal Residual : {r:.6f} < {eps_pri:.6f}")
            print(f"- Dual Residual   : {s:.6f} < {eps_dual:.6f}")
            break
    else:
        print(f"ADMM 최대 반복 횟수 도달 ({max_iter} iter). 완전한 수렴 실패.")

    states_admm = np.array([n.z.flatten() for n in X_nodes])
    controls_admm = np.array([n.z.flatten() for n in U_nodes])
    print(f"Hessian ADMM 완료 | 최종 상태 오차: {np.linalg.norm(states_admm[-1] - x_goal):.4f}")

    cost_ilqr = compute_total_cost(states_ilqr, controls_ilqr, Q, R, Qf, x_goal)
    cost_admm = compute_total_cost(states_admm, controls_admm, Q, R, Qf, x_goal)
    print(f"iLQR 최종 비용: {cost_ilqr:.4f}")
    print(f"ADMM 최종 비용: {cost_admm:.4f}")

    # --- 3. 결과 비교 시각화 ---
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.plot(states_admm[:, 0], label='Hessian ADMM Pos', lw=2)
    plt.plot(states_ilqr[:, 0], label='Standard iLQR Pos', lw=2, linestyle='--')
    plt.axhline(y=x_goal[0], color='r', linestyle=':', label='Goal')
    plt.title("Position Trajectory")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(states_admm[:, 1], label='Hessian ADMM Vel', lw=2)
    plt.plot(states_ilqr[:, 1], label='Standard iLQR Vel', lw=2, linestyle='--')
    plt.axhline(y=x_goal[1], color='r', linestyle=':', label='Goal')
    plt.title("Velocity Trajectory")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(controls_admm[:, 0], label='Hessian ADMM Ctrl', lw=2)
    plt.plot(controls_ilqr[:, 0], label='Standard iLQR Ctrl', lw=2, linestyle='--')
    plt.title("Control Input (Accel)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(np.abs(states_admm[:, 0] - states_ilqr[:, 0]), color='purple')
    plt.title("Position Diff (|ADMM - iLQR|)")
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(np.abs(states_admm[:, 1] - states_ilqr[:, 1]), color='purple')
    plt.title("Velocity Diff (|ADMM - iLQR|)")
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(np.abs(controls_admm[:, 0] - controls_ilqr[:, 0]), color='purple')
    plt.title("Control Diff (|ADMM - iLQR|)")
    plt.grid(True)

    plt.tight_layout()
    
    # --- 4. 시각화: ADMM 수렴 과정 (Residual vs Tolerance & Hessian P) ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history_r, label='Primal Residual (r)', color='blue', lw=2)
    plt.plot(history_eps_pri, label='Primal Tolerance (eps_pri)', color='blue', linestyle='--', alpha=0.7)
    plt.title("Primal Residual vs Tolerance")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.subplot(1, 3, 2)
    plt.plot(history_s, label='Dual Residual (s)', color='red', lw=2)
    plt.plot(history_eps_dual, label='Dual Tolerance (eps_dual)', color='red', linestyle='--', alpha=0.7)
    plt.title("Dual Residual vs Tolerance")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.subplot(1, 3, 3)
    plt.plot(history_P_mean, label='Mean(diag(P))', color='green', lw=2)
    plt.title("Hessian (P) Magnitude Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Mean value of Matrix P")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()