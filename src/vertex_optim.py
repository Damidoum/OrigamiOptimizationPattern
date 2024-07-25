from __future__ import annotations
import numpy as np
from typing import List, Tuple, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from utils import Utils


@dataclass
class Symmetry:
    symmetry_angle: float  # angle of symmetry

    def apply(self, vertex):
        if vertex.symmetrize(self.symmetry_angle).is_angle_compatible(vertex):
            return True
        return False


@dataclass
class Boundary:
    index: int  # index of the angle
    min_angle: float  # minimum angle
    max_angle: float  # maximum angle

    def __post_init__(self):
        self.min_angle %= 2 * np.pi
        self.max_angle %= 2 * np.pi

    def apply(self, vertex):
        if vertex.branches[self.index].angle < self.min_angle:
            return False
        if vertex.branches[self.index].angle > self.max_angle:
            return False
        return True


@dataclass
class DiffAngle:
    index1: int  # index of the first angle
    index2: int  # index of the second angle
    min_diff = -float(np.inf)  # minimum difference between the two angles
    max_diff = float(np.inf)  # maximum difference between the two angles

    def apply(self, vertex):
        diff = (
            (vertex.branches[self.index1].angle - vertex.branches[self.index2].angle)
            % 2
            * np.pi
        )
        if diff < self.min_diff:
            return False
        if diff > self.max_diff:
            return False
        return True


@dataclass
class Rotation:
    angle: float  # Angle de rotation en degrés

    def apply(self, point: Tuple[float, float]) -> Tuple[float, float]:
        radians = np.radians(self.angle)
        cos_theta = np.cos(radians)
        sin_theta = np.sin(radians)
        x, y = point
        x_new = cos_theta * x - sin_theta * y
        y_new = sin_theta * x + cos_theta * y
        return (x_new, y_new)


@dataclass
class Translation:
    dx: float  # Déplacement en x
    dy: float  # Déplacement en y

    def apply(self, point: Tuple[float, float]) -> Tuple[float, float]:
        x, y = point
        return (x + self.dx, y + self.dy)


@dataclass
class Transformation:
    rotations: list = None
    translations: list = None

    def __post_init__(self):
        if self.rotations is None:
            self.rotations = []
        if self.translations is None:
            self.translations = []

    def add_rotation(self, rotation: Rotation):
        self.rotations.append(rotation)

    def add_translation(self, translation: Translation):
        self.translations.append(translation)

    def apply(self, point: Tuple[float, float]) -> Tuple[float, float]:
        for rotation in self.rotations:
            point = rotation.apply(point)
        for translation in self.translations:
            point = translation.apply(point)
        return point


@dataclass
class Branch:
    angle: float
    length: float

    def __post_init__(self):
        self.angle %= 2 * np.pi


@dataclass
class Vertex:
    branches: Union[
        Tuple[float, float], List[Tuple[float, float]], Branch, List[Branch]
    ]
    constraints: dict = None
    tesselation_compatibilities: List[Translation] = None

    def __post_init__(self):
        self.branches = Utils.ensure_list(self.branches)
        if (
            len(self) != 0
            and isinstance(self.branches[0], tuple)
            and len(self.branches[0]) == 2
        ):
            self.branches = [Branch(*branch_param) for branch_param in self.branches]
        self._sort()

    def __rpr__(self):
        return f"Vertex({",".join([str(np.degrees(branch.angle)) for branch in self.branches])})"

    def __str__(self):
        return f"Vertex({",".join([str(round(np.degrees(branch.angle), 1)) for branch in self.branches])})"

    def __getitem__(
        self, index: Union[int, List[int], tuple, slice]
    ) -> Union[List[Branch], None]:
        if isinstance(index, int):
            return self.branches[index]
        if isinstance(index, list):
            return [self.branches[i] for i in index]
        if isinstance(index, tuple):
            return [self.branches[i] for i in index]
        if isinstance(index, slice):
            return [
                self.branches[i] for i in range(index.start, index.stop, index.step)
            ]
        print("Invalid index")
        return None

    def __setitem__(self, index: int, value: Branch):
        self.branches[index] = value

    def __delitem__(self, index: int):
        del self.branches[index]

    def __len__(self) -> int:
        return len(self.branches)

    def __eq__(self, other) -> bool:
        if isinstance(other, Vertex):
            return self.branches == other.branches
        return False

    def extract_branches(
        self, index: Union[int, List[int], tuple, slice]
    ) -> Union[List[Branch], None]:
        if isinstance(index, int):
            return Vertex(self.branches[index])
        if isinstance(index, list):
            return Vertex([self.branches[i] for i in index])
        if isinstance(index, tuple):
            return Vertex([self.branches[i] for i in index])
        if isinstance(index, slice):
            return Vertex(
                [self.branches[i] for i in range(index.start, index.stop, index.step)]
            )
        print("Invalid index")
        return None

    def _sort(self):
        self.branches.sort(key=lambda x: x.angle)

    def append_branch(self, branch):
        self.branches.append(branch)
        self._sort()  # very non optimal should be changed later

    def rotate(self, angle: float) -> Vertex:
        """Rotation of the vertex

        Args:
            angle (float): angle of rotation in radians
        """
        new_vertex = Vertex([], self.constraints, self.tesselation_compatibilities)
        for branch in self.branches:
            new_vertex.append_branch(Branch(branch.angle + angle, branch.length))
        return new_vertex

    def _rotate(self, angle: float):
        """Inplace rotation of the vertex

        Args:
            angle (float): angle of rotation in radians
        """
        for branch in self.branches:
            branch.angle += angle
            branch.angle %= 2 * np.pi
        self._sort()

    def symmetrize(self, symmetry_angle: float) -> Vertex:
        new_vertex = Vertex([], self.constraints, self.tesselation_compatibilities)
        for branch in self.branches:
            new_vertex.append_branch(
                Branch(2 * symmetry_angle - branch.angle, branch.length)
            )
        return new_vertex

    def _symmetrize(self, symmetry_angle: float):
        for branch in self.branches:
            branch.angle = 2 * symmetry_angle - branch.angle
            branch.angle %= 2 * np.pi
        self._sort()

    def is_angle_compatible(self, vertex2, eps=1e-6):
        for branch1, branch2 in zip(self.branches, vertex2.branches):
            if not abs(branch1.angle - branch2.angle) < eps:
                return False
        return True

    def plot(self, color: str = "red", alpha: int = 1, linestyle: str = "-", ax=None):
        if ax is None:
            _, ax = plt.subplots()
        for branch in self.branches:
            x = [0, branch.length * np.cos(branch.angle)]
            y = [0, branch.length * np.sin(branch.angle)]
            ax.plot(x, y, marker="o", color=color, alpha=alpha, linestyle=linestyle)

        ax.set_aspect("equal", "box")
        plt.grid(False)
        return ax


if __name__ == "__main__":
    yoshimura = Vertex(
        [
            (0, 1),
            (np.pi / 3, 1),
            (np.pi - np.pi / 3, 1),
            (np.pi, 1),
            (np.pi + np.pi / 3, 1),
            (-np.pi / 3, 1),
        ],
        None,
        None,
    )
    yoshimura_rotate = yoshimura.rotate(np.pi + 0.1)
    yoshimura_sym = yoshimura_rotate.symmetrize(np.pi)
    sym = Symmetry(np.pi)
    bound = Boundary(0, 0, 0.09)
    print(bound.apply(yoshimura))
    print(bound.apply(yoshimura_rotate))
    ax = yoshimura_rotate.plot()
    yoshimura_sym.plot(color="blue", ax=ax)
    plt.show()
