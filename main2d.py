import numpy as np  
import matplotlib.pyplot as plt 
import math as ma 
# Each L is length 1 
l_1 = 1
l_2 = 1 
fixed = np.array([0,0])
def plotElbow(angles): 
    x = l_1 * ma.cos(angles[0]) 
    y = l_1 * ma.sin(angles[0])
    return x,y 

def plotEnd(angles): 
    x = l_1 * ma.cos(angles[0]) + l_2 * ma.cos(angles[0] + angles[1]) 
    y = l_1 * ma.sin(angles[0]) + l_2 * ma.sin(angles[0] + angles[1]) 
    return x,y 
def main():
    jointAngles = np.array([0,0])
    target = np.array([[3,5]])
    x_1, y_1 = plotElbow(jointAngles)
    x_2,y_2 = plotEnd(jointAngles)
    plt.figure(figsize=(6,6))
    plt.plot(fixed[0], fixed[1], 'ro') 
    plt.plot(x_1,y_1,'go')
    plt.plot(x_2,y_2,'go')
    plt.grid(True)
    plt.xlim(-4, 4) 
    plt.ylim(-4, 4) 
    plt.show()     


if __name__ == "__main__": 
    main() 
