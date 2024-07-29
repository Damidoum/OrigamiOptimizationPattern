from __future__ import annotations
from numpy import pi as PI
from numpy import degrees, cos, sin 
from numpy import radians as rad
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
        self.min_angle %= 2 * PI 
        self.max_angle %= 2 * PI
    
    def __repr__(self):
        return f"Boundary({self.index}, {round(degrees(self.min_angle), 1)}, {round(degrees(self.max_angle), 1)})"
    
    def __str__(self):
        return self.__repr__()

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
    min_diff: float = -float("inf")  # minimum difference between the two angles
    max_diff: float = float("inf")  # maximum difference between the two angles

    def __repr__(self):
        return f"DiffAngle({self.index1}, {self.index2}, {round(degrees(self.min_diff), 1)}, {round(degrees(self.max_diff), 1)})"
    
    def __str__(self):
        return self.__repr__()
    
    def __sort_index(self):
        if self.index1 > self.index2:
            self.index1, self.index2 = self.index2, self.index1

    def apply(self, vertex):
        self.__sort_index()
        diff = abs(
            (vertex.branches[self.index1].angle - vertex.branches[self.index2].angle)
            % (2 * PI)
        )
        diff = min(diff, 2 * PI - diff)
        if diff < self.min_diff - 1e-6: # 1e-6 is used to avoid floating point errors
            return False
        if diff > self.max_diff + 1e-6: # 1e-6 is used to avoid floating point errors
            return False
        return True


@dataclass
class Rotation:
    angle: float  # Angle de rotation en degrés

    def apply(self, point: Tuple[float, float]) -> Tuple[float, float]:
        radians = rad(self.angle)
        cos_theta = cos(radians)
        sin_theta = sin(radians)
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
        self.angle %= 2 * PI 
    
    def __eq__(self, other: Branch)  -> bool:
        if isinstance(other, Branch):
            if abs(self.angle - other.angle) < 1e-6:
                return True
            else:
                return False
        return False

    def __repr__(self):
        return f"branch({round(degrees(self.angle), 2), self.length})"
    
    def __str__(self):
        return f"branch({round(degrees(self.angle), 2), self.length})"

    def is_close_to(self, other: Branch, threshold: float = 2 * PI / 360 * 5):
        if isinstance(other, Branch):
            if abs(self.angle - other.angle) < threshold:
                return True
            else:
                return False
        return False


@dataclass
class Vertex:
    branches: Union[
        Tuple[float, float], List[Tuple[float, float]], Branch, List[Branch]
    ]
    constraints: List = None
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
        self.constraints = Utils.ensure_list(self.constraints)

    def __repr__(self):
        return f"Vertex({",".join([str(round(degrees(branch.angle), 1)) for branch in self.branches])}), constraints: {self.constraints}, tesselation_compatibilities: {self.tesselation_compatibilities}"

    def __str__(self):
        return f"Vertex({",".join([str(round(degrees(branch.angle), 1)) for branch in self.branches])}), constraints: {self.constraints}, tesselation_compatibilities: {self.tesselation_compatibilities}"

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

    def __eq__(self, other)  -> bool:
        if isinstance(other, Vertex):
            return self.branches == other.branches
        return False
    
    def __find_offset(self, angle: float) -> int:
        count = 0
        for branch in self.branches[-1::-1]:
            if branch.angle + angle >= 2 * PI - 1e-10:
                count += 1
        return count

    def is_close_to(self, other: Vertex, threshold: float = 2 * PI / 360 * 5):
        if isinstance(other, Vertex):
            if len(self) != len(other):
                return False
            for branch1, branch2 in zip(self.branches, other.branches):
                if not branch1.is_close_to(branch2, threshold):
                    return False
            return True

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

    def check_constraints(self) -> bool:
        for constraint in self.constraints:
            if not constraint.apply(self):
                return False
        return True

    def _sort(self):
        self.branches.sort(key=lambda x: x.angle)
    

    def append_branch(self, branch, sort: bool = True):
        self.branches.append(branch)
        if sort:
            self._sort()  # very non optimal should be changed later

    def rotate(self, angle: float) -> Vertex:
        """Rotation of the vertex

        Args:
            angle (float): angle of rotation in radians
        """
        offset = self.__find_offset(angle)
        new_vertex = Vertex([], [], [])
        for branch in self.branches:
            new_vertex.append_branch(Branch(branch.angle + angle, branch.length), sort = False)
        new_vertex.branches = new_vertex.branches[-offset:] + new_vertex.branches[:-offset]
        if self.constraints is not None:
            for constraint in self.constraints:
                if isinstance(constraint, DiffAngle):
                    new_vertex.constraints.append(DiffAngle((constraint.index1 + offset) % len(self), (constraint.index2 + offset) % len(self), constraint.min_diff, constraint.max_diff))
                if isinstance(constraint, Symmetry):
                    new_vertex.constraints.append(Symmetry(constraint.symmetry_angle + angle))
        return new_vertex

    def _rotate(self, angle: float):
        """Inplace rotation of the vertex

        Args:
            angle (float): angle of rotation in radians
        """
        offset = self.__find_offset(angle)
        for branch in self.branches:
            branch.angle += angle
            branch.angle %= 2 * PI 
        if self.constraints is not None:
                for constraint in self.constraints:
                    if isinstance(constraint, DiffAngle):
                        constraint.index1 = (constraint.index1 + offset) % len(self)
                        constraint.index1 = (constraint.index1 + offset) % len(self)
        self.branches = self.branches[-offset:] + self.branches[:-offset]

    def symmetrize(self, symmetry_angle: float) -> Vertex:
        new_vertex = Vertex([], self.constraints, self.tesselation_compatibilities)
        for branch in self.branches:
            new_vertex.append_branch(
                Branch(2 * symmetry_angle - branch.angle, branch.length), sort = False
            )
        new_vertex._sort()
        return new_vertex

    def _symmetrize(self, symmetry_angle: float):
        for branch in self.branches:
            branch.angle = 2 * symmetry_angle - branch.angle
            branch.angle %= 2 * PI
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
            x = [0, branch.length * cos(branch.angle)]
            y = [0, branch.length * sin(branch.angle)]
            ax.plot(x, y, marker="o", color=color, alpha=alpha, linestyle=linestyle)

        ax.set_aspect("equal", "box")
        plt.grid(False)
        return ax


if __name__ == "__main__":
    yoshimura = Vertex(
        [
            (0, 1),
            (PI / 3, 1),
            (PI - PI / 3, 1),
            (PI, 1),
            (PI + PI / 3, 1),
            (-PI / 3, 1),
        ],
        DiffAngle(1, 4, PI, PI),
        None,
    )
    miura = Vertex(
        [
            (PI / 6, 1),
            (PI / 2, 1),
            (PI - PI / 6, 1),
            (-PI / 2, 1),
        ],
        [DiffAngle(1, 3, PI, PI), DiffAngle(0, 2, 0, PI), Symmetry(PI / 2)],
        None,
    )
    test = miura.rotate(rad(105))
    print(test)
    constraint = Symmetry(PI / 2)
    ax = test.plot()
    test_sym = test.symmetrize(PI/2)
    test_sym.plot(color="blue", ax = ax)
    plt.show()
