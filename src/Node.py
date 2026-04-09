import numpy as np
from abc import abstractmethod
from typing import List, Tuple
from scipy.linalg import block_diag
from src.Graph import Node, Edge, Graph

class VNode(Node):
    def __init__(self, name: str, dims: list, 
                 rho_method: str = 'residual',
                 rho_max: float = 100.0,
                 alpha_cov: float = 1.0,
                 mu_res: float = 1.5,
                 init_z: np.ndarray = None,
                 init_std: float = 1.0,
                 n_particles: int = 100,
                 tau_res: float = 1.5) -> None:
        super().__init__(name, dims)
        self.dim = int(np.prod(dims)) if dims else 1
        
        if init_z is None:
            self.z_consensus = np.random.randn(self.dim, n_particles) * init_std
        else:
            base_z = init_z.reshape(-1, 1)
            # init_z를 중심으로 init_std만큼 퍼진 앙상블 생성
            self.z_consensus = base_z + np.random.randn(self.dim, n_particles) * init_std
        
        # ADMM 페널티 조절 파라미터
        self.rho_method = rho_method.lower()
        self.rho_max = rho_max
        self.alpha_cov = alpha_cov
        self.mu_res = mu_res
        self.tau_res = tau_res

    def update_consensus_and_dual(self):
        # 앙상블 파티클 개수 N 가져오기
        N = self.edges[0].local_ensemble.shape[1] 
        
        # 합의점 Z 역시 (dim, N) 차원의 앙상블 행렬이 됨
        rho_sum = 0.0
        weighted_val_sum = np.zeros((self.dim, N)) # (dim, 1)이 아니라 (dim, N)
        
        # 1. Ensemble Z-update
        for edge in self.edges:
            # [핵심] 평균(x_mean)을 내지 않고 파티클(X_local) 전체를 그대로 사용!
            X_local = edge.local_ensemble 
            rho = edge.rho
            
            adjusted_ensemble = X_local + (edge.dual_lambda / rho)
            
            rho_sum += rho
            weighted_val_sum += rho * adjusted_ensemble
            
        # Z 앙상블 도출! (100개의 파티클이 각각 합의점을 가짐)
        self.z_consensus = weighted_val_sum / rho_sum
        
        # 2. Ensemble Dual-update
        for edge in self.edges:
            edge.z_target_prev = edge.z_target.copy()
            edge.z_target = self.z_consensus.copy() 
            
            # 람다 업데이트도 파티클별로 독립적으로 진행
            edge.dual_lambda += edge.rho * (edge.local_ensemble - self.z_consensus)
            
        # 3. 페널티 조절 (단, 앙상블이 죽지 않으므로 rho_method를 훨씬 부드럽게 세팅 가능)
        self._update_penalties()

    def _update_penalties(self):
        for edge in self.edges:
            x_mean = np.mean(edge.local_ensemble, axis=1, keepdims=True)
            
            if self.rho_method == 'covariance':
                # 제안 기법: 퍼짐 정도(Covariance Trace)에 반비례하게 조절
                N = edge.local_ensemble.shape[1]
                Xc = edge.local_ensemble - x_mean
                C_xx = (Xc @ Xc.T) / (N - 1)
                trace_c = float(np.trace(C_xx))
                
                new_rho = self.rho_max / (1.0 + self.alpha_cov * trace_c)
                edge.rho = min(new_rho, self.rho_max)
                
            elif self.rho_method == 'residual':
                # 기존 기법: Primal(r) vs Dual(s) 잔차 밸런싱
                r = float(np.linalg.norm(x_mean - edge.z_target))
                s = float(np.linalg.norm(edge.rho * (edge.z_target - edge.z_target_prev)))
                
                if r > self.mu_res * s:
                    edge.rho = min(edge.rho * self.tau_res, self.rho_max)
                elif s > self.mu_res * r:
                    edge.rho = max(edge.rho / self.tau_res, 1e-4) # 0 방지


class FNode(Node):
    def __init__(self, name: str, dims: list, gamma: np.ndarray) -> None:
        super().__init__(name, dims)
        self.gamma = gamma
        self.noise_scale = 1.0

    @abstractmethod
    def error_function(self, local_ensembles: List[np.ndarray]) -> np.ndarray:
        pass

    def eki_x_update(self):
        local_ensembles = [edge.local_ensemble for edge in self.edges]
        E_phys = self.error_function(local_ensembles) 
        
        E_admm_list = []
        gamma_admm_list = []
        X_stacked_list = []
        
        for edge in self.edges:
            X = edge.local_ensemble
            N = X.shape[1]
            X_stacked_list.append(X)
            
            # 개별 에지의 rho를 사용하여 가상 관측치와 공분산 세팅
            virtual_obs = edge.z_target - (edge.dual_lambda / edge.rho)
            E_admm = X - virtual_obs 
            E_admm_list.append(E_admm)
            
            # ADMM 패널티 강도에 따른 가상 노이즈 공분산 (강할수록 분산이 작아져 더 강하게 끌어당김)
            gamma_admm_list.append(np.eye(X.shape[0]) / edge.rho)

        X_total = np.vstack(X_stacked_list)
        E_total = np.vstack([E_phys] + E_admm_list)
        Gamma_total = block_diag(self.gamma, *gamma_admm_list)
        
        N = X_total.shape[1]
        X_mean = np.mean(X_total, axis=1, keepdims=True)
        E_mean = np.mean(E_total, axis=1, keepdims=True)

        C_xy = ((X_total - X_mean) @ (E_total - E_mean).T) / (N - 1)
        C_yy = ((E_total - E_mean) @ (E_total - E_mean).T) / (N - 1)

        K = C_xy @ np.linalg.inv(C_yy + Gamma_total + 1e-8 * np.eye(Gamma_total.shape[0]))
        
        zero_mean = np.zeros(Gamma_total.shape[0])
        noise = np.random.multivariate_normal(zero_mean, Gamma_total, N).T
        X_new_total = X_total + K @ (-E_total + noise)

        expected_variance = np.trace(C_yy) + np.trace(Gamma_total)
        actual_error_sq = np.sum(E_mean ** 2)
        eta = actual_error_sq / (expected_variance + 1e-8)

        # print(f"{self.name} | E_mean norm: {np.linalg.norm(E_mean):.4f}, Expected Var: {expected_variance:.4f}, Actual Error^2: {actual_error_sq:.4f}, Eta: {eta:.4f}, Rho: {[edge.rho for edge in self.edges]}")
        
        # [핵심] Eta를 이용한 동적 노이즈 스케일링
        # 초기 노이즈(self.initial_noise_scale)에 eta의 제곱근을 곱해줍니다.
        # - eta > 1 이면 노이즈 증폭 (탐색 강화)
        # - eta < 1 이면 노이즈 축소 (수렴 강화)
        dynamic_noise_scale = self.noise_scale * np.sqrt(eta)
        
        # 안전장치 1: 노이즈가 무한히 커지는 것을 방지 (예: 초기 노이즈의 최대 3배까지만 허용)
        dynamic_noise_scale = min(dynamic_noise_scale, self.noise_scale * 1.0)
        
        # 안전장치 2: 시간이 지날수록 전반적인 탐색 반경을 서서히 좁히기 (선택 사항)
        # decay_factor = 0.99 ** self.current_iteration 
        # dynamic_noise_scale *= decay_factor
        
        # 안전장치 3: 완벽한 수렴을 위해 노이즈가 특정 임계치 이하면 완전히 꺼버림
        if dynamic_noise_scale < 1e-4:
            dynamic_noise_scale = 0.0

        # 로그 출력 (디버깅용으로 켜두시면 수렴 과정을 관찰하기 매우 좋습니다)
        # print(f"[{self.name}] Eta: {eta:.4f} -> Noise Scale: {dynamic_noise_scale:.4f}")

        # 노이즈 주입!
        if dynamic_noise_scale > 0:
            X_new_total += dynamic_noise_scale * np.random.randn(*X_new_total.shape)


        # X_new_total += self.noise_scale * np.random.randn(*X_new_total.shape)

        idx = 0
        for edge in self.edges:
            dim = edge.local_ensemble.shape[0]
            edge.local_ensemble = X_new_total[idx:idx+dim, :]
            idx += dim


class FactorGraph(Graph):
    def iterate(self, n_iter: int = 1) -> None:
        fnodes = [n for n in self.nodes if isinstance(n, FNode)]
        vnodes = [n for n in self.nodes if isinstance(n, VNode)]
        
        for _ in range(n_iter):
            
            # Synchronous update
            for fn in fnodes:
                fn.eki_x_update()
                fn.noise_scale *= 0.95
                
            for vn in vnodes:
                vn.update_consensus_and_dual()