import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import math as ma

l_1 = 1
l_2 = 1
l_3 = 1

fixed = np.array([0.0, 0.0])

a = 0.1
l_a = 0.05


def plotAngle(angles):

    t1, t2, t3 = angles

    x1 = l_1 * ma.cos(t1)
    y1 = l_1 * ma.sin(t1)

    x2 = x1 + l_2 * ma.cos(t1 + t2)
    y2 = y1 + l_2 * ma.sin(t1 + t2)

    x3 = x2 + l_3 * ma.cos(t1 + t2 + t3)
    y3 = y2 + l_3 * ma.sin(t1 + t2 + t3)

    return x1, y1, x2, y2, x3, y3


def main():

    jointAngles = np.array([(2*ma.pi)/3, -ma.pi/3, -ma.pi/3], dtype=float)
    target = np.array([0.0, 1.5], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(bottom=0.15)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True)

    ax.plot(fixed[0], fixed[1], 'ro')
    target_plot, = ax.plot(target[0], target[1], 'rx', markersize=8)
    arm, = ax.plot([], [], 'o-', color='green', markersize=5, linewidth=2)

    x1, y1, x2, y2, x3, y3 = plotAngle(jointAngles)
    arm.set_data([0, x1, x2, x3], [0, y1, y2, y3])

    keys = set()
    step = 0.05

    def key_press(event):
        if event.key is not None:
            keys.add(event.key.lower())
    def key_release(event):
        if event.key is not None:
            keys.discard(event.key.lower())
    fig.canvas.mpl_connect("key_press_event", key_press)
    fig.canvas.mpl_connect("key_release_event", key_release)
    def update(frame):

        nonlocal jointAngles, target

        if 'i' in keys:
            target[1] += step
        if 'k' in keys:
            target[1] -= step
        if 'j' in keys:
            target[0] -= step
        if 'l' in keys:
            target[0] += step

        target_plot.set_data([target[0]], [target[1]])

        x1, y1, x2, y2, x3, y3 = plotAngle(jointAngles)
        err = np.array([
            target[0] - x3,
            target[1] - y3
        ])
        t1, t2, t3 = jointAngles
        j = np.array([
            [
                -(l_1 * ma.sin(t1)) - (l_2 * ma.sin(t1 + t2)) - (l_3 * ma.sin(t1 + t2 + t3)),
                -(l_2 * ma.sin(t1 + t2)) - (l_3 * ma.sin(t1 + t2 + t3)),
                -(l_3 * ma.sin(t1 + t2 + t3))
            ],
            [
                (l_1 * ma.cos(t1)) + (l_2 * ma.cos(t1 + t2)) + (l_3 * ma.cos(t1 + t2 + t3)),
                (l_2 * ma.cos(t1 + t2)) + (l_3 * ma.cos(t1 + t2 + t3)),
                (l_3 * ma.cos(t1 + t2 + t3))
            ]
        ])
        jointAngles += (j.T @ np.linalg.inv(j @ j.T + (l_a ** 2) * np.identity(2))) @ (a * err)
        x1, y1, x2, y2, x3, y3 = plotAngle(jointAngles)
        arm.set_data([0, x1, x2, x3],[0, y1, y2, y3])
        ax.set_title(f"step: {frame}   err: {np.linalg.norm(err):.4f}")

        return arm, target_plot

    ani = None
    def start(event):
        nonlocal ani
        if ani is not None:
            return
        button.label.set_text("Running")

        ani = FuncAnimation(
            fig,
            update,
            frames=2000,
            interval=40,
            repeat=False
        )
        fig.canvas.draw_idle()

    button_ax = fig.add_axes([0.4, 0.03, 0.2, 0.06])
    button = Button(button_ax, "Button")
    button.on_clicked(start)

    plt.show()


if __name__ == "__main__":
    main()