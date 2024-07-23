import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Symmetry:
    symmetry_angle: float  # angle of symmetry

    def apply(self, vertex):
        if vertex.rotate(self.symmetry_angle).is_angle_compatible(vertex):
            return True
        return False


@dataclass
class Boundary:
    index: int  # index of the angle
    min_angle: float  # minimum angle
    max_angle: float  # maximum angle


@dataclass
class DiffAngle:
    index1: int  # index of the first angle
    index2: int  # index of the second angle
    min_diff = -float(np.inf)  # minimum difference between the two angles
    max_diff = float(np.inf)  # maximum difference between the two angles


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


@dataclass
class Vertex:
    branches: List[Tuple[float, float]]
    constraints: dict
    tesselation_compatibilities: List[Translation]

    def __post_init__(self):
        self.branches = [Branch(angle, length) for angle, length in self.branches]

    def append_branch(self, branch):
        self.branches.append(branch)

    def rotate(self, angle: float):
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

    def is_angle_compatible(self, vertex2):
        for branch1, branch2 in zip(self.branches, vertex2.branches):
            if not branch1.angle == branch2.angle:
                return False
        return True

    def plot(self, color="red", ax=None):
        if ax is None:
            _, ax = plt.subplots()
        for branch in self.branches:
            x = [0, branch.length * np.cos(branch.angle)]
            y = [0, branch.length * np.sin(branch.angle)]
            ax.plot(x, y, marker="o", color=color)

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
    yoshimura_rotate = yoshimura.rotate(np.pi / 6)
    ax = yoshimura.plot()
    yoshimura_rotate.plot("blue", ax)
    plt.show()
