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
                 init_std: float = 1.0, rho_init: float = 1.0) -> None:
        self._node0 = node0
        self._node1 = node1
        self._messages = {} 
        
        self.local_ensemble = np.random.randn(dim, n_particles) * init_std
        
        self.dual_lambda = np.zeros((dim, 1))
        
        self.z_target = np.zeros((dim, 1))
        self.z_target_prev = np.zeros((dim, 1))
        
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