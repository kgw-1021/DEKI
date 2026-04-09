from src.Node import VNode, FNode

class DynamicsFactor(FNode):
    def __init__(self, name: str, dims: list) -> None:
        super().__init__(name, dims)

    def error_function(self, x_prev, x_curr):
        # 간단한 1차 시스템 예시: x_curr = A * x_prev + noise
        A = np.eye(self.dim)  # 단위 행렬로 간단히 표현
        predicted = A @ x_prev
        error = x_curr - predicted
        return error