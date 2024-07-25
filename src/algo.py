from __future__ import annotations

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

        def __str__(self) -> str:
            return f"Rotation: {degrees(self.rotation):.2f}, Angle adjustments: {", ".join([str(round(degrees(angle), 2)) for angle in self.angle_adjustments])}, Cost: {self.cost:.3f}"

        def __repr__(self) -> str:
            return self.__str__()

        def convert_to_vertex(self, vertex: Vertex) -> Vertex:
            new_vertex = vertex.rotate(self.rotation)
            for i, angle in enumerate(self.angle_adjustments):
                new_vertex[i].angle += angle
            return new_vertex

        def __compare_angle_adjustments(self, output: List[float], eps: float) -> bool:
            if len(output.angle_adjustments) != len(self.angle_adjustments):
                return False
            for i, angle in enumerate(output.angle_adjustments):
                if abs(angle - self.angle_adjustments[i]) > eps:
                    return False
            return True

        def isInOutputList(
            self, output_list: Algorithm.Output, eps: float = 1e-6
        ) -> bool:
            for output in output_list:
                if (
                    self.rotation == output.rotation
                    and self.__compare_angle_adjustments(output, eps)
                    and abs(self.cost - output.cost) < eps
                ):
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
            if not new_output.isInOutputList(self.output):
                print("Filling the output!")
                self.output[-1] = new_output
            else:
                print("Already in the output!")
            """
            self.output[-1].rotation = global_rotation
            self.output[-1].angle_adjustments = [
                angle if abs(angle) <= self.threshold else 0
                for angle in Utils.distance_between_vertex(
                    subset_larger_vertex, rotated_smaller_vertex, offset
                )
            ]
            self.output[-1].cost = cost
            """
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
