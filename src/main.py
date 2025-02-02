import matplotlib.pyplot as plt
from vertex_optim import Vertex, Symmetry, Boundary, DiffAngle
from algo import Algorithm
from numpy import pi as PI
from typing import List
from numpy import cos, sin


def plot_vertex(
    list_vertex: list[Vertex],
    color: List[str] = ["red"],
    linestyle: str = "-",
    alpha: float = 1,
):
    ax = list_vertex[0].plot(color=color[0], linestyle=linestyle, alpha=alpha)
    for i, vertex in enumerate(list_vertex[1:]):
        vertex.plot(
            color=color[i + 1 % len(color)], ax=ax, alpha=alpha, linestyle=linestyle
        )
    plt.show()


def plot_vertices_side_by_side(
    vertices,
    compare_vertex=None,
    spacing=2,
    color: List[str] = ["red", "blue", "green"],
    alpha=1,
    linestyle="-",
):
    """
    Trace une liste de Vertex côte à côte.

    Args:
        vertices (List[Vertex]): Liste des objets Vertex à tracer.
        spacing (float): Espace entre chaque Vertex sur l'axe x.
        color (str): Couleur des lignes.
        alpha (int): Transparence des lignes.
        linestyle (str): Style de ligne.
    """
    _, ax = plt.subplots(figsize=(5, 5))

    # Tracer chaque Vertex
    for i, vertex in enumerate(vertices):
        # Déplacement horizontal pour chaque Vertex
        offset = i * spacing
        for branch in vertex.branches:
            x = [offset, offset + branch.length * cos(branch.angle)]
            y = [0, branch.length * sin(branch.angle)]
            ax.plot(
                x,
                y,
                marker="o",
                color=color[i % len(color)],
                alpha=alpha,
                linestyle=linestyle,
            )
            if compare_vertex is not None:
                for branch2 in compare_vertex.branches:
                    x = [offset, offset + branch2.length * cos(branch2.angle)]
                    y = [0, branch2.length * sin(branch2.angle)]
                    ax.plot(
                        x,
                        y,
                        marker="o",
                        color="grey",
                        alpha=0.15,
                        linestyle="dashed",
                    )

    ax.set_aspect("equal", "box")
    # ax.set_aspect(5)
    plt.grid(True)
    plt.title("Vertices Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


if __name__ == "__main__":
    algo = Algorithm(threshold=2 * PI / 360 * 30, number_of_output=10)
    yoshimura = Vertex(
        [
            (PI / 4, 1),
            (PI / 2, 1),
            (PI - PI / 4, 1),
            (PI + PI / 4, 1),
            (-PI / 2, 1),
            (-PI / 4, 1),
        ],
        None,
        None,
    )
    miura = Vertex(
        [
            (PI / 6, 1),
            (PI / 2, 1),
            (PI - PI / 6, 1),
            (-PI / 2, 1),
        ],
        [Symmetry(PI / 2)],
        None,
    )
    output = algo(yoshimura, miura)
    print(output)
    output = [out.vertex for out in output if out.vertex is not None]
    plot_vertices_side_by_side(
        output,
        compare_vertex=None,
        color=["red", "blue", "black", "green", "yellow"],
        alpha=0.15,
        linestyle="--",
    )
