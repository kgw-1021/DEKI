import numpy as np
from abc import abstractmethod
from typing import List, Tuple
from src.Graph_admm import Node, Edge, Graph

class VNode(Node):
    """
    VNode: 기존과 동일하게 Residual Balancing을 유지함.
    """
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
        sum_x_rho = np.zeros((self.dim, 1))
        sum_rho = 0.0
        
        for edge in self.edges:
            sum_x_rho += edge.rho * (edge.local_x + edge.dual_lambda / edge.rho)
            sum_rho += edge.rho
        
        self.z = sum_x_rho / sum_rho
        
        for edge in self.edges:
            edge.dual_lambda += edge.rho * (edge.local_x - self.z)

        # Residual Balancing
        r_sq, s_sq = 0.0, 0.0
        for edge in self.edges:
            r_sq += np.sum((edge.local_x - self.z) ** 2)
            s_sq += np.sum((edge.rho * (self.z - self.z_prev)) ** 2)
        r, s = np.sqrt(r_sq), np.sqrt(s_sq)

        # print(f"[{self.name}] r: {r:.2e}, s: {s:.2e}, rho: {edge.rho:.2e}")

        if r > self.mu_res * s:
            for edge in self.edges:
                edge.rho = min(edge.rho * self.tau_res, self.rho_max)
                edge.dual_lambda *= self.tau_res
        elif s > self.mu_res * r:
            for edge in self.edges:
                edge.rho = max(edge.rho / self.tau_res, 1e-6)
                edge.dual_lambda /= self.tau_res

class FNode(Node):
    """
    FNode: 내부 루프를 통해 x가 수렴할 때까지 Gauss-Newton을 반복함.
    """
    def __init__(self, name: str, dims: list, gamma: np.ndarray, 
                 max_inner_iter: int = 1, inner_tol: float = 1e-5):
        super().__init__(name, dims)
        self.inv_gamma = np.linalg.inv(gamma)
        self.max_inner_iter = max_inner_iter # 최대 내부 반복 횟수
        self.inner_tol = inner_tol           # 수렴 조건 (delta_x의 크기)

    @abstractmethod
    def error_function(self, local_xs: list) -> np.ndarray:
        pass

    @abstractmethod
    def jacobian_function(self, local_xs: list) -> list:
        pass

    def admm_x_update(self):
        """
        Inner Loop: z와 lambda가 고정된 상태에서 x에 대해 최적화 (Exact Step)
        """
        for i in range(self.max_inner_iter):
            local_xs = [edge.local_x for edge in self.edges]
            f = self.error_function(local_xs)
            js = self.jacobian_function(local_xs)
            
            J_total = np.hstack(js)
            X_total = np.vstack(local_xs)
            
            v_list = []
            rho_diag = []
            for edge in self.edges:
                v_list.append(edge.get_other(self).z - (edge.dual_lambda / edge.rho))
                rho_diag.extend([edge.rho] * edge.local_x.shape[0])
            
            V_total = np.vstack(v_list)
            R = np.diag(rho_diag)
            
            # KKT System (Gauss-Newton step)
            lhs = J_total.T @ self.inv_gamma @ J_total + R
            rhs = -J_total.T @ self.inv_gamma @ f - R @ (X_total - V_total)
            
            lhs += 1e-6 * np.eye(lhs.shape[0]) 
            
            delta_x = np.linalg.solve(lhs, rhs)
            
            # 업데이트 적용
            idx = 0
            for edge in self.edges:
                dim = edge.local_x.shape[0]
                edge.local_x += delta_x[idx:idx+dim, :]
                idx += dim
            
            # 수렴 판정 (변화량이 충분히 작으면 조기 종료)
            if np.linalg.norm(delta_x) < self.inner_tol:
                # print(f"[{self.name}] Inner converged at iter {i+1}")
                break

class FactorGraph(Graph):
    def iterate(self):
        for node in self.nodes:
            if isinstance(node, FNode):
                node.admm_x_update()
        for node in self.nodes:
            if isinstance(node, VNode):
                node.update_consensus_and_dual()