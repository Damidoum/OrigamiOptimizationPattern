import matplotlib.pyplot as plt
from vertex_optim import Vertex
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
    spacing=5,
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
    _, ax = plt.subplots()

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

    ax.set_aspect("equal", "box")
    plt.grid(True)
    plt.title("Vertices Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


if __name__ == "__main__":
    algo = Algorithm(threshold=2 * PI / 360 * 60, number_of_output=10)
    yoshimura = Vertex(
        [
            (PI / 3, 1),
            (PI / 2, 1),
            (PI - PI / 3, 1),
            (PI + PI / 3, 1),
            (-PI / 2, 1),
            (-PI / 3, 1),
        ],
        None,
        None,
    )
    miura = Vertex(
        [
            (PI / 3 + 0.1, 1),
            (PI / 2, 1),
            (PI - PI / 3 - 0.1, 1),
            (-PI / 2, 1),
        ],
        None,
        None,
    )
    output = algo(yoshimura, miura)
    output1 = output[0].convert_to_vertex(miura)
    output2 = output[1].convert_to_vertex(miura)
    print(output)
    output_vertex = [output[i].convert_to_vertex(miura) for i in range(10)]
    for out in output_vertex:
        print(out)
        print("----")
    plot_vertices_side_by_side(
        output_vertex,
        color=["red", "blue", "black", "green", "yellow"],
        alpha=0.15,
        linestyle="--",
    )
    # print(output2)
    # plot_vertex([miura, output2], color=["red", "blue", "black", "green"])
