from typing import Callable, Tuple
import numpy as np


def vectorize(func: Callable) -> Callable:
    """Dectorator to vectorize a function.

    Args:
        func (Callable): Function to vectorize.

    Returns:
        Callable: Vectorized function.
    """
    return np.vectorize(func)


class Utils:
    @staticmethod
    def detect_smaller_vertex(vertex1, vertex2) -> Tuple:
        """Detect the smaller of two vertices.

        Args:
            vertex1 (Vertex): First vertex.
            vertex2 (Vertex): Second vertex.

        Returns:
            Tuple[Vertex, Vertex]: Smaller vertex, Larger vertex.
        """
        if len(vertex1) < len(vertex2):
            return vertex1, vertex2
        return vertex2, vertex1

    @staticmethod
    @vectorize
    def angle_difference(angle1: float, angle2: float) -> float:
        """Compute the difference between two angles.

        Args:
            angle1 (float): First angle.
            angle2 (float): Second angle.

        Returns:
            float: angle difference
        """
        diff = abs(angle1 - angle2) % 360
        return np.sign(angle1 - angle2) * min(diff, 360 - diff)

    @staticmethod
    def distance_between_vertex(vertex1, vertex2, offset: int) -> Tuple[float]:
        """Compute the distance between two vertices.

        Args:
            vertex1 (Vertex): First vertex.
            vertex2 (Vertex): Second vertex.
            offset (int, optional): Offset to compare the vertices with different starting points. Default to 0.

        Returns:
            Tuple[float]: List of distance between the two vertices.
        """
        assert len(vertex1) == len(vertex2)
        print(offset)
        return list(
            Utils.angle_difference(
                vertex1[(i + offset) % len(vertex1)].angle, vertex2[i].angle
            )
            for i in range(len(vertex1))
        )

    @staticmethod
    def ensure_list(obj: any) -> list:
        if obj is None:
            return []
        if isinstance(obj, list):
            return obj
        return [obj]