import numpy as np
from typing import List

class Node:
    def __init__(self, name: str, dims: list) -> None:
        self._name = name
        self._dims = dims
        self.edges: List['Edge'] = []

    @property
    def name(self) -> str: return self._name
    @property
    def dims(self) -> list: return self._dims

    def add_edge(self, edge: 'Edge'):
        if edge not in self.edges: self.edges.append(edge)

class Edge:
    def __init__(self, node0: Node, node1: Node, 
                 dim: int = 1, rho_init: float = 1.0,
                 init_val: np.ndarray = None) -> None:
        self._node0 = node0
        self._node1 = node1
        
        if init_val is not None:
            self.local_x = init_val.reshape(-1, 1).copy()
        else:
            self.local_x = np.zeros((dim, 1))
        
        # Unscaled Dual Variable 사용 (lambda)
        self.dual_lambda = np.zeros((dim, 1))
        
        # 스칼라 rho 대신 Penalty Matrix P (Precision 역할) 사용
        self.P = rho_init * np.eye(dim)

        self.rho = rho_init
        
        node0.add_edge(self)
        node1.add_edge(self)

    def get_other(self, node: Node) -> Node:
        return self._node1 if node is self._node0 else self._node0

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def connect(self, node0: Node, node1: Node, **kwargs) -> Edge:
        edge = Edge(node0, node1, **kwargs)
        self.edges.append(edge)
        if node0 not in self.nodes: self.nodes.append(node0)
        if node1 not in self.nodes: self.nodes.append(node1)
        return edge