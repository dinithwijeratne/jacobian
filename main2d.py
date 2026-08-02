import numpy as np  
import matplotlib.pyplot as plt 
import math as ma 
# Each L is length 1 
l_1 = 1
l_2 = 1 
fixed = np.array([0,0])
a =  0.2 

def plotAngle(angles): 
    x1 = l_1 * ma.cos(angles[0]) 
    y1 = l_1 * ma.sin(angles[0])
    x2 = l_1 * ma.cos(angles[0]) + l_2 * ma.cos(angles[0] + angles[1]) 
    y2 = l_1 * ma.sin(angles[0]) + l_2 * ma.sin(angles[0] + angles[1]) 
    return x1,y1,x2,y2 
def main():
    jointAngles = np.array([0,0], dtype=float)
    target = np.array([0,2])

    x_1, y_1,x_2,y_2  = plotAngle(jointAngles)

    err = np.array([target[0] - x_2, target[1] - y_2])

    j = np.array([
        [-(l_1*ma.sin(jointAngles[0])) -(l_2*ma.sin(jointAngles[0] + jointAngles[1])), -(l_2*ma.sin(jointAngles[0]+ jointAngles[1]))],
        [(l_1 * ma.cos(jointAngles[0])) + (l_2 * ma.cos(jointAngles[0] + jointAngles[1])),(l_2*ma.cos(jointAngles[0]+ jointAngles[1]))]
    ])
    j_pinv = np.linalg.pinv(j)     
    jointAngles += j_pinv @ (a*err) 
    x_3, y_3, x_4, y_4 = plotAngle(jointAngles)
    plt.figure(figsize=(6,6))
    plt.plot(target[0], target[1], 'rx', markersize=8)
    plt.plot(fixed[0], fixed[1], 'ro') 

    plt.plot(x_1,y_1,'go')
    plt.plot(x_2,y_2,'go')

    plt.plot(x_3,y_3, 'gx')
    plt.plot(x_4,y_4, 'gx')

    angle = np.linspace(0,2*ma.pi, 100)

    x = 2 * np.cos(angle)
    y = 2 * np.sin(angle)
    plt.plot(x,y, 'm-', linewidth=1) 
    plt.grid(True)
    plt.xlim(-4, 4) 
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal') 
    plt.show()     


if __name__ == "__main__": 
    main() 
