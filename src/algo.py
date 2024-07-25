from __future__ import annotations

import copy
from dataclasses import dataclass, field
from numpy import pi as PI
from typing import Any, List
from vertex_optim import Vertex, Symmetry, Boundary, DiffAngle, Transformation
from itertools import combinations
from loss import Loss
from utils import Utils
from numpy import radians, degrees


class Algorithm:
    @dataclass
    class Output:
        rotation: float = 0
        angle_adjustments: List[float] = field(default_factory=list)
        cost: float = float("inf")
        vertex: Vertex = None

        def __str__(self) -> str:
            return f"Rotation: {degrees(self.rotation):.2f}, Angle adjustments: {", ".join([str(round(degrees(angle), 2)) for angle in self.angle_adjustments])}, Cost: {self.cost:.3f}"

        def __repr__(self) -> str:
            return self.__str__()
        
        def __eq__(self, other: Algorithm.Output) -> bool:
            return self.vertex == other.vertex

        def convert_to_vertex(self, vertex: Vertex, already_rotated: bool = False) -> Vertex:
            if not already_rotated:
                new_vertex = vertex.rotate(self.rotation)
            else:
                new_vertex = copy.deepcopy(vertex)
            for i, angle in enumerate(self.angle_adjustments):
                new_vertex[i].angle += angle
            self.vertex = new_vertex
            self.vertex._sort()
            return new_vertex

        def isInOutputList(self, output_list: List[Algorithm.Output], threshold: float = 2 * PI / 360 * 5) -> bool:
            for output in output_list:
                if output == self:
                    return True
            return False

    def __init__(self, threshold: float = PI / 3, number_of_output: int = 1) -> None:
        self.threshold = threshold
        self.loss = Loss(self.threshold)
        self.output = [self.Output() for _ in range(number_of_output)]
        self.number_of_output = number_of_output

    def __reset_output(self) -> None:
        self.output = [self.Output(0) for _ in range(self.number_of_output)]

    def __sort_output(self):
        self.output.sort(key=lambda x: x.cost)

    def __align_vertex(
        self,
        larger_vertex: Vertex,
        rotated_smaller_vertex: Vertex,
        global_rotation: float,
        subset: List[int],
        offset: int,
    ) -> bool:
        subset_larger_vertex = larger_vertex.extract_branches(
            subset
        )  # extract subset vertex
        cost = self.loss(rotated_smaller_vertex, subset_larger_vertex, offset)
        if cost < self.output[-1].cost:
            new_output = self.Output(
                global_rotation,
                [
                    angle if abs(angle) <= self.threshold else 0
                    for angle in Utils.distance_between_vertex(
                        subset_larger_vertex, rotated_smaller_vertex, offset
                    )
                ],
                cost,
            )
            new_output.convert_to_vertex(rotated_smaller_vertex, already_rotated=True)
            if not new_output.isInOutputList(self.output):
                self.output[-1] = new_output
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

        while global_rotation < 360 and self.output[-1].cost != 0:
            rotated_smaller_vertex = smaller_vertex.rotate(radians(global_rotation))
            for subset in combinations(range(len(larger_vertex)), len(smaller_vertex)):
                for offset in range(len(smaller_vertex)):
                    self.__align_vertex(
                        larger_vertex,
                        rotated_smaller_vertex,
                        radians(global_rotation),
                        subset,
                        offset,
                    )
            global_rotation += 1

        return self.output

    def __call__(self, vertex1: Vertex, vertex2: Vertex) -> List[Output]:
        return self.optimize_pattern(vertex1, vertex2)
