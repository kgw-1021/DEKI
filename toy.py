import numpy as np
import matplotlib.pyplot as plt

# 새로 작성해주신 프레임워크 임포트 (경로는 실제에 맞게 수정)
from src.Node import VNode, FNode, FactorGraph

# -----------------------------------------------------------------------------
# 1. 2D Nonlinear Factors 정의 (새 프레임워크 구조)
# -----------------------------------------------------------------------------
class DistanceFactor(FNode):
    """ 두 변수 노드 사이의 유클리디안 거리를 제약하는 2D 팩터 """
    def __init__(self, name, dims, target_dist, gamma):
        super().__init__(name, dims, gamma)
        self.target_dist = target_dist

    def error_function(self, local_ensembles):
        """
        local_ensembles = [x_i ensemble, x_j ensemble]
        residual: ||x_i - x_j|| - target_dist
        """
        x_i = local_ensembles[0]
        x_j = local_ensembles[1]

        # 두 앙상블 간의 거리 계산 (차원별 차이의 노름)
        diff = x_i - x_j
        dist = np.linalg.norm(diff, axis=0, keepdims=True) # Shape: (1, N)
        
        return dist - self.target_dist
    
class PriorFactor(FNode):
    """ 변수 노드의 위치에 대한 Prior 제약 """
    def __init__(self, name, dims, target_pos, gamma):
        super().__init__(name, dims, gamma)
        self.target_pos = target_pos

    def error_function(self, local_ensembles):
        """
        local_ensembles = [x_i ensemble]
        residual: x_i - target_pos
        """
        x_i = local_ensembles[0] # Shape: (2, N)
        return x_i - self.target_pos.reshape(-1, 1) # Shape: (2, N)


# -----------------------------------------------------------------------------
# 2. Ground Truth 및 설정
# -----------------------------------------------------------------------------
np.random.seed(1)

# 정답 (정삼각형 형태의 3개 노드 위치)
gt_x1 = np.array([0.0, 0.0])
gt_x2 = np.array([10.0, 0.0])
gt_x3 = np.array([5.0, 8.66025]) 
gt_x4 = np.array([5.0, 8.66025* np.sqrt(3)]) 

# 측정된 거리
d23 = np.linalg.norm(gt_x2 - gt_x3)
d31 = np.linalg.norm(gt_x3 - gt_x1)
d34 = np.linalg.norm(gt_x3 - gt_x4)

# 측정 노이즈 공분산
gamma_dist = np.array([[0.1]])       # 거리 제약은 1차원이므로 1x1
gamma_prior = np.eye(2) * 0.001       # 위치 Prior는 2차원이므로 2x2

# -----------------------------------------------------------------------------
# 3. 그래프 구성
# -----------------------------------------------------------------------------
graph = FactorGraph()
N_particles = 100
# init_pos = np.array([[0, 0], [0, 0], [0, 0], [0, 0]]) + np.random.normal(0, 1.0, (4, 2))  # 초기 추정값 (약간의 노이즈 포함)
init_pos = np.array([gt_x1, gt_x2, gt_x3, gt_x4]) + np.random.normal(0, 5.0, (4, 2))  # 초기 추정값 (약간의 노이즈 포함)

# Variable Nodes (2D)
v1 = VNode("x1", dims=[2], rho_method='covariance', init_z=init_pos[0].reshape(-1, 1), n_particles=N_particles, alpha_cov=10.0, rho_max=100.0)
v2 = VNode("x2", dims=[2], rho_method='covariance', init_z=init_pos[1].reshape(-1, 1), n_particles=N_particles, alpha_cov=10.0, rho_max=100.0)
v3 = VNode("x3", dims=[2], rho_method='covariance', init_z=init_pos[2].reshape(-1, 1), n_particles=N_particles, alpha_cov=10.0, rho_max=100.0)
v4 = VNode("x4", dims=[2], rho_method='covariance', init_z=init_pos[3].reshape(-1, 1), n_particles=N_particles, alpha_cov=10.0, rho_max=100.0)


anchor_prior_v1 = PriorFactor("anchor", dims=[2], target_pos=gt_x1, gamma=gamma_prior)
anchor_prior_v2 = PriorFactor("anchor", dims=[2], target_pos=gt_x2, gamma=gamma_prior)
anchor_prior_v4 = PriorFactor("anchor", dims=[2], target_pos=gt_x4, gamma=gamma_prior)


f23 = DistanceFactor("f23", dims=[1], target_dist=d23, gamma=gamma_dist)
f31 = DistanceFactor("f31", dims=[1], target_dist=d31, gamma=gamma_dist)
f43 = DistanceFactor("f43", dims=[1], target_dist=d34, gamma=gamma_dist)
# f14 = DistanceFactor("f14", dims=[1], target_dist=np.linalg.norm(gt_x1 - gt_x4), gamma=gamma_dist)
# f24 = DistanceFactor("f24", dims=[1], target_dist=np.linalg.norm(gt_x2 - gt_x4), gamma=gamma_dist)
# f12 = DistanceFactor("f12", dims=[1], target_dist=np.linalg.norm(gt_x1 - gt_x2), gamma=gamma_dist)



graph.connect(v1, anchor_prior_v1, dim=2, n_particles=N_particles)
graph.connect(v2, anchor_prior_v2, dim=2, n_particles=N_particles)
graph.connect(v4, anchor_prior_v4, dim=2, n_particles=N_particles)

graph.connect(v2, f23, dim=2, n_particles=N_particles)
graph.connect(v3, f23, dim=2, n_particles=N_particles)

graph.connect(v3, f31, dim=2, n_particles=N_particles)
graph.connect(v1, f31, dim=2, n_particles=N_particles)

graph.connect(v4, f43, dim=2, n_particles=N_particles)
graph.connect(v3, f43, dim=2, n_particles=N_particles)

# graph.connect(v1, f14, dim=2, n_particles=N_particles)
# graph.connect(v4, f14, dim=2, n_particles=N_particles)

# graph.connect(v2, f24, dim=2, n_particles=N_particles)
# graph.connect(v4, f24, dim=2, n_particles=N_particles)

# graph.connect(v1, f12, dim=2, n_particles=N_particles)
# graph.connect(v2, f12, dim=2, n_particles=N_particles)
# -----------------------------------------------------------------------------
# 4. 실시간 시각화를 포함한 반복 수행 (Iteration)
# -----------------------------------------------------------------------------
n_iter = 100

plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))

hist_v1, hist_v2, hist_v3, hist_v4 = [], [], [], []
total_errors = []
err1_history, err2_history, err3_history, err4_history = [], [], [], []   

for i in range(n_iter):
    # 1 스텝 업데이트 (EKI x-update -> Z-update & Dual-update)
    graph.iterate(1)
    
    # Z-Consensus 값을 추출
    m1 = np.mean(v1.z_consensus, axis=1)
    m2 = np.mean(v2.z_consensus, axis=1)
    m3 = np.mean(v3.z_consensus, axis=1)
    m4 = np.mean(v4.z_consensus, axis=1)
    
    hist_v1.append(m1)
    hist_v2.append(m2)
    hist_v3.append(m3)
    hist_v4.append(m4)
    
    # 에러 계산 (대표 좌표와 Ground Truth 사이의 거리)
    err1 = np.linalg.norm(m1 - gt_x1)
    err2 = np.linalg.norm(m2 - gt_x2)
    err3 = np.linalg.norm(m3 - gt_x3)
    err4 = np.linalg.norm(m4 - gt_x4)

    total_errors.append(err1 + err2 + err3 + err4)
    err1_history.append(err1)
    err2_history.append(err2)
    err3_history.append(err3)
    err4_history.append(err4)

    # ---------------- 플로팅 업데이트 ----------------
    ax.clear()
    
    # 1. Ground Truth 플롯 (별 모양)
    ax.scatter(*gt_x1, c='red', marker='*', s=300, zorder=5, label='Ground Truth')
    ax.scatter(*gt_x2, c='red', marker='*', s=300, zorder=5)
    ax.scatter(*gt_x3, c='red', marker='*', s=300, zorder=5)
    ax.scatter(*gt_x4, c='red', marker='*', s=300, zorder=5)
    
    # 2. 로컬 앙상블(파티클) 스캐터 플롯 
    ens1 = v1.edges[0].local_ensemble
    ens2 = v2.edges[0].local_ensemble
    ens3 = v3.edges[0].local_ensemble
    ens4 = v4.edges[0].local_ensemble
    ax.scatter(ens1[0, :], ens1[1, :], c='blue', alpha=0.1, s=10)
    ax.scatter(ens2[0, :], ens2[1, :], c='green', alpha=0.1, s=10)
    ax.scatter(ens3[0, :], ens3[1, :], c='magenta', alpha=0.1, s=10)
    ax.scatter(ens4[0, :], ens4[1, :], c='cyan', alpha=0.1, s=10)

    # -------------------------------------------------------------------------
    # [수정 2] 보너스: Z 합의점의 앙상블 분포도 시각화 ('x' 마커)
    # 로컬 앙상블(동그라미)들이 Z 앙상블(X 표시)을 향해 어떻게 융합되는지 볼 수 있습니다.
    # -------------------------------------------------------------------------
    z_ens1 = v1.z_consensus
    z_ens2 = v2.z_consensus
    z_ens3 = v3.z_consensus
    z_ens4 = v4.z_consensus 
    ax.scatter(z_ens1[0, :], z_ens1[1, :], c='darkblue', alpha=0.5, s=15, marker='x')
    ax.scatter(z_ens2[0, :], z_ens2[1, :], c='darkgreen', alpha=0.5, s=15, marker='x')
    ax.scatter(z_ens3[0, :], z_ens3[1, :], c='purple', alpha=0.5, s=15, marker='x')
    ax.scatter(z_ens4[0, :], z_ens4[1, :], c='darkcyan', alpha=0.5, s=15, marker='x')
    
    # 3. 합의점(Z)의 이동 경로(Trajectory) 플롯
    h1 = np.array(hist_v1)
    h2 = np.array(hist_v2)
    h3 = np.array(hist_v3)
    h4 = np.array(hist_v4)
    ax.plot(h1[:, 0], h1[:, 1], 'b--', linewidth=2, label='x1 Path')
    ax.plot(h2[:, 0], h2[:, 1], 'g--', linewidth=2, label='x2 Path')
    ax.plot(h3[:, 0], h3[:, 1], 'm--', linewidth=2, label='x3 Path')
    ax.plot(h4[:, 0], h4[:, 1], 'c--', linewidth=2, label='x4 Path')
    
    # 4. 현재 대표 합의점 플롯
    ax.scatter(*m1, c='blue', s=150, edgecolors='black', zorder=4)
    ax.scatter(*m2, c='green', s=150, edgecolors='black', zorder=4)
    ax.scatter(*m3, c='magenta', s=150, edgecolors='black', zorder=4)
    ax.scatter(*m4, c='cyan', s=150, edgecolors='black', zorder=4)

    # UI 정리
    ax.set_title(f"Iteration: {i+1}/{n_iter}\nTotal Error: {total_errors[-1]:.4f}", fontsize=14)
    ax.set_xlim(-5, 20)
    ax.set_ylim(-5, 20)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right')
    
    plt.pause(0.1)

plt.ioff()
print("Optimization Complete!")
print(f"Final Estimated Positions:\n x1: {m1}\n x2: {m2}\n x3: {m3}\n x4: {m4}")
print(f"Final Errors:\n x1: {err1_history[-1]}\n x2: {err2_history[-1]}\n x3: {err3_history[-1]}\n x4: {err4_history[-1]}")
print(f"Final Total Error: {total_errors[-1]:.4f}")

plt.figure(figsize=(8, 5))
plt.plot()
plt.plot(total_errors, 'k-', linewidth=2, label='Total Error')
plt.plot(err1_history, 'b--', label='Error x1')
plt.plot(err2_history, 'g--', label='Error x2')
plt.plot(err3_history, 'm--', label='Error x3')
plt.plot(err4_history, 'c--', label='Error x4')
plt.legend()
plt.title("Total Error over Iterations", fontsize=14)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Total Error", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()
