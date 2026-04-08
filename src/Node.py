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
                 tau_res: float = 1.5) -> None:
        super().__init__(name, dims)
        self.dim = int(np.prod(dims)) if dims else 1
        self.z_consensus = np.zeros((self.dim, 1)) if init_z is None else init_z.reshape(-1, 1)
        
        # ADMM 페널티 조절 파라미터
        self.rho_method = rho_method.lower()
        self.rho_max = rho_max
        self.alpha_cov = alpha_cov
        self.mu_res = mu_res
        self.tau_res = tau_res

    def update_consensus_and_dual(self):
        weight_sum = np.zeros((self.dim, self.dim))
        weighted_val_sum = np.zeros((self.dim, 1))
        
        # 1. Z-update: 공분산 및 개별 rho를 모두 고려한 가중 평균
        # 수식: Z = (\sum \rho_i \Sigma_i^{-1})^{-1} \sum \rho_i \Sigma_i^{-1} (x_i + \lambda_i / \rho_i)
        for edge in self.edges:
            X_local = edge.local_ensemble
            N = X_local.shape[1]
            x_mean = np.mean(X_local, axis=1, keepdims=True)
            
            Xc = X_local - x_mean
            cov = (Xc @ Xc.T) / (N - 1)
            precision = np.linalg.inv(cov + 1e-6 * np.eye(self.dim))
            
            
            W = edge.rho * precision 
            adjusted_mean = x_mean + (edge.dual_lambda / edge.rho)
            
            weight_sum += W
            weighted_val_sum += W @ adjusted_mean
            
        self.z_consensus = np.linalg.inv(weight_sum) @ weighted_val_sum
        
        # 2. Dual-update 및 z_target 동기화
        for edge in self.edges:
            edge.z_target_prev = edge.z_target.copy()
            edge.z_target = self.z_consensus.copy()
            
            x_mean = np.mean(edge.local_ensemble, axis=1, keepdims=True)
            edge.dual_lambda += edge.rho * (x_mean - self.z_consensus)
            
        # 3. 각 에지별로 적응형 패널티(rho) 조절
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

        # expected_variance = np.trace(C_yy) + np.trace(Gamma_total)
        # actual_error_sq = np.sum(E_mean ** 2)
        # eta = actual_error_sq / (expected_variance + 1e-8)

        # print(f"{self.name} | E_mean norm: {np.linalg.norm(E_mean):.4f}, Expected Var: {expected_variance:.4f}, Actual Error^2: {actual_error_sq:.4f}, Eta: {eta:.4f}, Rho: {[edge.rho for edge in self.edges]}")
        
        X_new_total += self.noise_scale * np.random.randn(*X_new_total.shape)
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
                fn.noise_scale *= 0.98
                
            for vn in vnodes:
                vn.update_consensus_and_dual()