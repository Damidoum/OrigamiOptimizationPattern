import matplotlib.pyplot as plt
from vertex_optim import Vertex
from algo import Algorithm
from numpy import pi as PI


def plot_vertex(list_vertex: list[Vertex], color: str = "red"):
    ax = list_vertex[0].plot(color=color)
    for vertex in list_vertex[1:]:
        vertex.plot(color=color, ax=ax)
    plt.show()


if __name__ == "__main__":
    algo = Algorithm(threshold=2 * PI / 360 * 5)
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
    plot_vertex([yoshimura, miura])
    # ax = yoshimura.plot()
    # miura.plot(color="blue", ax=ax)

    # plt.show()
