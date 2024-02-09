from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial


def animate_save(
    frame_data: np.ndarray[float, Any], x_axis: np.ndarray[float, Any]
) -> None:
    fig, ax = plt.subplots()
    (line1,) = ax.plot([], [], "b.")

    def init():
        ax.set_xlim(x_axis[0], x_axis[-1])
        ax.set_ylim(0, 1)

        return (line1,)

    def update(frame):
        line1.set_data(x_axis, frame_data[frame])
        return line1

    ani = FuncAnimation(
        fig, update, frames=np.arange(frame_data.shape[0]), init_func=init, interval=50
    )

    ani.save("problem_e.mp4")
