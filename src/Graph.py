import numpy as np
from typing import List

class Node:
    def __init__(self, name: str, dims: list) -> None:
        self._name = name
        self._dims = dims
        self.edges: List['Edge'] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def dims(self) -> list:
        return self._dims

    def add_edge(self, edge: 'Edge'):
        if edge not in self.edges:
            self.edges.append(edge)
            
    def remove_edge(self, edge: 'Edge'):
        if edge in self.edges:
            self.edges.remove(edge)

class Edge:
    def __init__(self, node0: Node, node1: Node, 
                 dim: int = 1, n_particles: int = 100, 
                 init_std: float = 1.0, rho_init: float = 1.0,
                 init_ensemble: np.ndarray = None) -> None:
        self._node0 = node0
        self._node1 = node1
        self._messages = {} 
        
        vnode = node0 if type(node0).__name__ == 'VNode' else (node1 if type(node1).__name__ == 'VNode' else None)
        
        if init_ensemble is not None:
            self.local_ensemble = init_ensemble.copy()
        elif vnode is not None and hasattr(vnode, 'z_consensus') and vnode.z_consensus.shape[1] == n_particles:
            # [핵심] VNode가 이미 앙상블을 가지고 있다면 그 상태를 완벽히 복제하여 시작!
            self.local_ensemble = vnode.z_consensus.copy()
            self.z_target = vnode.z_consensus.copy()
            self.z_target_prev = vnode.z_consensus.copy()
        else:
            self.local_ensemble = np.random.randn(dim, n_particles) * init_std
            self.z_target = np.zeros((dim, n_particles))
            self.z_target_prev = np.zeros((dim, n_particles))
        
        self.dual_lambda = np.zeros((dim, n_particles))
        
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
        
        if node0 not in self.nodes: 
            self.nodes.append(node0)
        if node1 not in self.nodes: 
            self.nodes.append(node1)
            
        return edge

    def remove_edge(self, edge: Edge):
        if edge in self.edges:
            edge._node0.remove_edge(edge)
            edge._node1.remove_edge(edge)
            self.edges.remove(edge)

    def remove_node(self, node: Node):
        if node in self.nodes:
            for edge in list(node.edges):
                self.remove_edge(edge)
            
            self.nodes.remove(node)