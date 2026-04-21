import numpy as np
from abc import abstractmethod
from typing import List, Tuple
from src.Graph_hessian import Node, Edge, Graph

class VNode(Node):
    def __init__(self, name: str, dims: list, init_z: np.ndarray = None) -> None:
        super().__init__(name, dims)
        self.dim = int(np.prod(dims)) if dims else 1
        self.z = init_z.reshape(-1, 1).copy() if init_z is not None else np.zeros((self.dim, 1))
        self.z_prev = self.z.copy()

    def update_consensus_and_dual(self):
        self.z_prev = self.z.copy()
        sum_Px = np.zeros((self.dim, 1))
        sum_P = np.zeros((self.dim, self.dim))
        
        for edge in self.edges:
            # 안전장치: P가 아직 초기화되지 않았다면 단위행렬 사용
            if not hasattr(edge, 'P'):
                edge.P = np.eye(self.dim)
            sum_Px += edge.P @ edge.local_x + edge.dual_lambda
            sum_P += edge.P
        
        sum_P += 1e-6 * np.eye(self.dim)
        # 1. z-update: 정밀도 가중 평균
        self.z = np.linalg.solve(sum_P, sum_Px)
        
        # 2. Dual-update
        for edge in self.edges:
            edge.dual_lambda += edge.P @ (edge.local_x - self.z)

class FNode(Node):
    def __init__(self, name: str, dims: list, gamma: np.ndarray, 
                 max_inner_iter: int = 1, inner_tol: float = 1e-5):
        super().__init__(name, dims)
        self.inv_gamma = np.linalg.inv(gamma)
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol

    @abstractmethod
    def error_function(self, local_xs: list) -> np.ndarray:
        pass

    @abstractmethod
    def jacobian_function(self, local_xs: list) -> list:
        pass

    def admm_x_update(self):
        for i in range(self.max_inner_iter):
            local_xs = [edge.local_x for edge in self.edges]
            f = self.error_function(local_xs)
            js = self.jacobian_function(local_xs)
            
            J_total = np.hstack(js)
            X_total = np.vstack(local_xs)
            
            # 실제 Gauss-Newton Hessian 계산
            H = J_total.T @ self.inv_gamma @ J_total
            
            idx = 0
            P_blocks = []
            lambda_total = []
            Z_total = []
            
            for edge in self.edges:
                dim = edge.local_x.shape[0]
                H_block = H[idx:idx+dim, idx:idx+dim]
                
                # --- [핵심 디버깅 해결 파트] ---
                # iLQR 제어 문제 특성상 하드 제약(Dynamics)은 H가 1e6 수준으로 폭발함.
                # 곡률 방향은 유지하되 최대 Stiffness 제한을 두어 시스템의 마비를 방지.
                try:
                    eigvals, eigvecs = np.linalg.eigh(H_block)
                    # 고유값 제한: 패널티 스케일이 [0.1, 10.0] 범위를 넘지 않게 조절
                    eigvals_clipped = np.clip(eigvals, 0.1, 0.5)
                    P_safe = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
                except np.linalg.LinAlgError:
                    # 행렬 분해 실패 시 안전한 백업
                    P_safe = 1.0 * np.eye(dim)
                
                # Dual 공간 붕괴 방지: P행렬에 EMA(지수 이동 평균) 적용하여 부드럽게 전환
                if not hasattr(edge, 'P'):
                    edge.P = P_safe
                else:
                    edge.P = 0.5 * edge.P + 0.5 * P_safe 
                # --------------------------------
                
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