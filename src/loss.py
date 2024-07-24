from vertex_optim import Vertex


class Loss:
    def __init__(self, thereshold) -> None:
        self.loss = 0.0
        self.thereshold = thereshold

    def __call__(self, vertex1: Vertex, vertex2: Vertex, offset: int = 0) -> float:
        value = self.compute(vertex1, vertex2, offset)
        self.reset()
        return value

    def reset(self):
        self.loss = 0.0

    def compute(self, vertex1: Vertex, vertex2: Vertex, offset: int = 0) -> float:
        self.loss = 0.0
        assert len(vertex1) == len(vertex2)
        for i in range(len(vertex1)):
            diff = abs(
                vertex1[i].angle - vertex2[(i + offset) % len(vertex2)].angle
            )  # offset is used to compare the vertices with different starting points
            if diff > self.thereshold:
                self.loss += 1
            else:
                self.loss += diff
        return self.loss
