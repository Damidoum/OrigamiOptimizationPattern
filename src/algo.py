from dataclasses import dataclass, field
from numpy import pi as PI
from typing import List
from vertex_optim import Vertex, Symmetry, Boundary, DiffAngle, Transformation
from itertools import combinations
from loss import Loss
from utils import Utils


class Algorithm:
    @dataclass
    class Output:
        rotation: float = 0
        angle_adjustments: List[float] = field(default_factory=list)
        cost: float = float("inf")

    def __init__(self, threshold: float = PI / 3, number_of_output: int = 1) -> None:
        self.threshold = threshold
        self.loss = Loss(self.threshold)
        self.output = [self.Output() for _ in range(number_of_output)]
        self.number_of_output = number_of_output

    def __reset_output(self):
        self.output = [self.Output(0) for _ in range(self.number_of_output)]

    def __sort_output(self):
        self.output.sort(key=lambda x: x.cost)

    def __align_vertex(
        self,
        rotated_larger_vertex: Vertex,
        smaller_vertex: Vertex,
        global_rotation: float,
        subset: List[int],
        offset: int,
    ) -> bool:
        subset_larger_vertex = rotated_larger_vertex[subset]  # extract subset vertex
        cost = self.loss(smaller_vertex, subset_larger_vertex, offset)
        if cost < self.output[-1].cost:
            print("Filling the output!")
            self.output[-1].rotation = global_rotation
            self.output[-1].angle_adjustments = [
                angle if abs(angle) <= self.threshold else 0
                for angle in Utils.distance_between_vertices(
                    smaller_vertex, subset_larger_vertex
                )
            ]
            self.output[-1].cost = cost
            self.__sort_output()

    def optimize_pattern(self, vertex1: Vertex, vertex2: Vertex) -> List[Output]:
        """Align two n-degree vertices to maximize the overlap of branches,

        Args:
            vertex1 (Vertex): First vertex.
            vertex2 (Vertex): Second vertex.

        Returns:
            : _description_
        """
        smaller_vertex, larger_vertex = Utils.detect_smaller_vertex(vertex1, vertex2)
        self.__reset_output()  # reset output
        global_rotation = 0

        while global_rotation <= 360 and self.output[-1].cost != 0:
            rotated_larger_vertex = larger_vertex.rotate(global_rotation)
            for subset in combinations(range(len(larger_vertex)), len(smaller_vertex)):
                for offset in range(len(smaller_vertex)):
                    self.__align_vertex(
                        rotated_larger_vertex,
                        smaller_vertex,
                        global_rotation,
                        subset,
                        offset,
                    )
                global_rotation += 1

        return self.output

    def __call__(self, vertex1: Vertex, vertex2: Vertex) -> List[Output]:
        return self.optimize_pattern(vertex1, vertex2)
