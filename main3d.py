import numpy as np  
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import math as ma
# Each L is length 1
l_1 = 1
l_2 = 1 
fixed = np.array([0,0,0])
a =  0.1 
e = 0.0001 
l_a = 0.05 

def plotAngle(angles): 
    x1 = l_1 * ma.cos(angles[0]) 
    y1 = l_1 * ma.sin(angles[0])
    x2 = l_1 * ma.cos(angles[0]) + l_2 * ma.cos(angles[0] + angles[1]) 
    y2 = l_1 * ma.sin(angles[0]) + l_2 * ma.sin(angles[0] + angles[1]) 
    return x1,y1,x2,y2
def htm(angles, joint):
    t_yaw = np.array([
        [ma.cos(angles[0]), -1*ma.sin(angles[0]), 0 ,0],
        [ma.sin(angles[0]), ma.cos(angles[0]), 0 , 0], 
        [0,0,1,0], 
        [0,0,0,1]
    ])
    t_shoulder_pitch = np.array([
        [ma.cos(angles[1]), 0, ma.sin(angles[1]), 0], 
        [0,1,0,0], 
        [-1*ma.sin(angles[1]), 0, ma.cos(angles[1]), 0], 
        [0,0,0,1]
    ])
    t_l1 = np.array([
        [1,0,0,l_1], 
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1] 
    ])
    t_elbow_pitch = np.array([ 
        [ma.cos(angles[2]), 0,ma.sin(angles[2]), 0], 
        [0,1,0,0], 
        [-1*ma.sin(angles[2]), 0,ma.cos(angles[2]), 0], 
        [0,0,0,1]
    ])
    t_l2 = np.array([ 
        [1,0,0,l_2], 
        [0,1,0,0], 
        [0,0,1,0], 
        [0,0,0,1]
    ])
    if joint == 2: 
        return t_yaw @ t_shoulder_pitch @ t_l1 @ t_elbow_pitch @ t_l2
    elif joint == 1: 
        return t_yaw @ t_shoulder_pitch @ t_l1 
    else: 
        return 0 
def main():
    jointAngles = np.array([0.1,ma.pi/2,0.1], dtype=float)
    target = np.array([0.5, 1,1])
    t_elbow = htm(jointAngles, 1)
    t_tip = htm(jointAngles, 2)
    elbowPostion = t_elbow[:3, 3] 
    endPostion = t_tip[:3,3]
    
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.15)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4,4) 
    ax.set_zlim(-4,4)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.plot(target[0], target[1], target[2], 'rx', markersize=8)
    ax.plot(fixed[0], fixed[1], fixed[2], 'ro')
    arm, = ax.plot([],[], [], 'o-', color='green', markersize=5, linewidth=2)

    # Show the starting pose so it is visible before the animation runs
    arm.set_data_3d([0, elbowPostion[0], endPostion[0]], [0, elbowPostion[1], endPostion[1]], [0,elbowPostion[2],endPostion[2]])
    ax.set_title("Press Start to begin")

    def update(frame):
        nonlocal jointAngles
        t_elbow = htm(jointAngles, 1)
        t_end = htm(jointAngles, 2)
        elbowPostion = t_elbow[:3,3]
        endPostion = t_end[:3,3] 
        err = target - endPostion     
        tol = .001   
        errNorm = np.linalg.norm(err) 
        if (errNorm <= tol): 
            if ani is not None:
                ani.event_source.stop() # Halts the animation loop
            ax.set_title(f"Converged at step: {frame} err: {errNorm:.3f}")
            button.label.set_text("Finished")
            return (arm,) 

        max_step = 0.05 
        if (errNorm > max_step):  
            dx = (err / errNorm) * max_step 
        else: 
            dx = err

        d_jointAngles = np.copy(jointAngles)
        d_jointAngles += e 
        j = np.empty((3,3))
        j[::, 0] =  (htm(np.array([d_jointAngles[0], jointAngles[1], jointAngles[2]]), 2)[:3 ,3] - endPostion) / e
        j[::, 1] =  (htm(np.array([jointAngles[0], d_jointAngles[1], jointAngles[2]]), 2)[:3 ,3] - endPostion) / e 
        j[::, 2] =  (htm(np.array([jointAngles[0], jointAngles[1], d_jointAngles[2]]), 2)[:3 ,3] - endPostion) / e 
        
        jointAngles += (j.T @ np.linalg.inv(j @ j.T + (l_a**2) * np.identity(3))) @ dx 
        arm.set_data_3d([0,elbowPostion[0], endPostion[0]], [0,elbowPostion[1],endPostion[1]], [0,elbowPostion[2],endPostion[2]]) 
        ax.set_title(f"step: {frame} err: {errNorm:.4f}")
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
