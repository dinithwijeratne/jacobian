import numpy as np  
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import math as ma
# Each L is length 1
l_1 = 1
l_2 = 1 
fixed = np.array([0,0])
a =  0.1 

def plotAngle(angles): 
    x1 = l_1 * ma.cos(angles[0]) 
    y1 = l_1 * ma.sin(angles[0])
    x2 = l_1 * ma.cos(angles[0]) + l_2 * ma.cos(angles[0] + angles[1]) 
    y2 = l_1 * ma.sin(angles[0]) + l_2 * ma.sin(angles[0] + angles[1]) 
    return x1,y1,x2,y2

def main():
    jointAngles = np.array([0.1,0.1], dtype=float)
    target = np.array([0,1.5])

    fig, ax = plt.subplots(figsize=(6,6))
    fig.subplots_adjust(bottom=0.15)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4,4) 
    ax.set_aspect('equal')
    ax.grid(True)

    ax.plot(target[0], target[1], 'rx', markersize=8)
    ax.plot(fixed[0], fixed[1], 'ro') 
    arm, = ax.plot([],[], 'o-', color='green', markersize=5, linewidth=2)

    # Show the starting pose so it is visible before the animation runs
    x_s1, y_s1, x_s2, y_s2 = plotAngle(jointAngles)
    arm.set_data([0, x_s1, x_s2], [0, y_s1, y_s2])
    ax.set_title("Press Start to begin")

    def update(frame):
        nonlocal jointAngles

        x_1, y_1,x_2,y_2  = plotAngle(jointAngles)
        err = np.array([target[0] - x_2, target[1] - y_2])


        j = np.array([
            [-(l_1*ma.sin(jointAngles[0])) -(l_2*ma.sin(jointAngles[0] + jointAngles[1])), -(l_2*ma.sin(jointAngles[0]+ jointAngles[1]))],
            [(l_1 * ma.cos(jointAngles[0])) + (l_2 * ma.cos(jointAngles[0] + jointAngles[1])),(l_2*ma.cos(jointAngles[0]+ jointAngles[1]))]
        ])
        j_pinv = np.linalg.pinv(j)     
        jointAngles += j_pinv @ (a*err) 
        x_3, y_3, x_4, y_4 = plotAngle(jointAngles)
        arm.set_data([0,x_3, x_4], [0,y_3,y_4]) 
        ax.set_title(f"step: {frame} err: {np.linalg.norm(err):.4f}")
        return (arm,)

    # The animation is only built once Start is clicked, so the figure sits on
    # the initial pose until then
    ani = None

    def start(event):
        nonlocal ani
        if ani is not None:
            return
        button.label.set_text("Running")
        ani = FuncAnimation(fig, update, frames = 200, interval=40, repeat=False)
        fig.canvas.draw_idle()

    button_ax = fig.add_axes([0.4, 0.03, 0.2, 0.06])
    button = Button(button_ax, "Start")
    button.on_clicked(start)

    plt.show()



if __name__ == "__main__": 
    main() 
