import numpy as np
from abc import abstractmethod
from typing import List, Tuple
from src.Graph_hessian import Node, Edge, Graph

class VNode(Node):
    def __init__(self, name: str, dims: list, init_z: np.ndarray = None,
                 mu_res: float = 10.0, tau_res: float = 2.0, rho_max: float = 1e3) -> None:
        super().__init__(name, dims)
        self.dim = int(np.prod(dims)) if dims else 1
        self.z = init_z.reshape(-1, 1).copy() if init_z is not None else np.zeros((self.dim, 1))
        self.z_prev = self.z.copy()
        self.mu_res = mu_res
        self.tau_res = tau_res
        self.rho_max = rho_max

    def update_consensus_and_dual(self):
        self.z_prev = self.z.copy()
        sum_Px = np.zeros((self.dim, 1))
        sum_P = np.zeros((self.dim, self.dim))
        
        for edge in self.edges:
            # 안전장치: P가 없으면 초기화
            if not hasattr(edge, 'P'):
                edge.rho = getattr(edge, 'rho', 1.0)
                edge.S = np.eye(self.dim)
                edge.P = edge.rho * edge.S
                
            sum_Px += edge.P @ edge.local_x + edge.dual_lambda
            sum_P += edge.P
        
        sum_P += 1e-6 * np.eye(self.dim)
        
        # 1. z-update: 정밀도 가중 평균
        self.z = np.linalg.solve(sum_P, sum_Px)
        
        for edge in self.edges:
            # Dual variable 업데이트 (Unscaled)
            edge.dual_lambda += edge.P @ (edge.local_x - self.z)
            
            # --- [핵심] 잔차 기반 스케일(rho) 조절 ---
            # Primal residual (합의 오차)
            r_norm = np.linalg.norm(edge.local_x - self.z)
            # Dual residual (z의 변동성)
            s_norm = np.linalg.norm(edge.P @ (self.z - self.z_prev))
            
            if r_norm > self.mu_res * s_norm:
                edge.rho *= self.tau_res      # 합의가 안되면 페널티 스케일업
            elif s_norm > self.mu_res * r_norm:
                edge.rho /= self.tau_res      # 듀얼이 너무 튀면 페널티 스케일다운
                
            # 스칼라 발산 방지 (안전범위)
            edge.rho = np.clip(edge.rho, 1e-4, self.rho_max)
            
            # 3. 새로운 rho를 반영하여 Penalty 행렬 P 업데이트
            edge.P = edge.rho * edge.S


class FNode(Node):
    def __init__(self, name: str, dims: list, gamma: np.ndarray, 
                 max_inner_iter: int = 1, inner_tol: float = 1e-5):
        super().__init__(name, dims)
        self.inv_gamma = np.linalg.inv(gamma)
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol

    @abstractmethod
    def error_function(self, local_xs: list) -> np.ndarray: pass

    @abstractmethod
    def jacobian_function(self, local_xs: list) -> list: pass

    def admm_x_update(self):
        for i in range(self.max_inner_iter):
            local_xs = [edge.local_x for edge in self.edges]
            f = self.error_function(local_xs)
            js = self.jacobian_function(local_xs)
            
            J_total = np.hstack(js)
            X_total = np.vstack(local_xs)
            
            # Gauss-Newton Hessian
            H = J_total.T @ self.inv_gamma @ J_total
            
            idx = 0
            P_blocks = []
            lambda_total = []
            Z_total = []
            
            for edge in self.edges:
                dim = edge.local_x.shape[0]
                H_block = H[idx:idx+dim, idx:idx+dim]
                
                scale_curvature = np.trace(H_block) + 1e-8
                edge.S = H_block / scale_curvature
                
                edge.P = edge.rho * edge.S
                
                P_blocks.append(edge.P)
                lambda_total.append(edge.dual_lambda)
                Z_total.append(edge.get_other(self).z)
                idx += dim
            
            P_total = np.zeros(H.shape)
            idx = 0
            for P in P_blocks:
                dim = P.shape[0]
                P_total[idx:idx+dim, idx:idx+dim] = P
                idx += dim
                
            Lambda_total = np.vstack(lambda_total)
            Z_total = np.vstack(Z_total)
            
            # KKT System Solver
            lhs = H + P_total
            lhs += 1e-6 * np.eye(lhs.shape[0]) 
            
            Grad_f = J_total.T @ self.inv_gamma @ f
            rhs = -Grad_f - Lambda_total - P_total @ (X_total - Z_total)
            
            delta_x = np.linalg.solve(lhs, rhs)
            step_norm = np.linalg.norm(delta_x) 
            idx = 0
            for edge in self.edges:
                dim = edge.local_x.shape[0]
                edge.local_x += delta_x[idx:idx+dim, :]
                idx += dim
            
            if step_norm < self.inner_tol:
                break

class FactorGraph(Graph):
    def iterate(self):
        for node in self.nodes:
            if isinstance(node, FNode):
                node.admm_x_update()
        for node in self.nodes:
            if isinstance(node, VNode):
                node.update_consensus_and_dual()