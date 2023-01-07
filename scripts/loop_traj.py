import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

if __name__ == "__main__":
    path = "scripts/IMS_centerline.csv"
    waypoints = np.genfromtxt(path, dtype=float, delimiter=",")
    xCoords = waypoints[:,0]
    yCoords = waypoints[:,1]

    tmax = 120
    tvec = np.linspace(0,120,len(xCoords))
    xTraj = CubicSpline(tvec,xCoords)
    yTraj = CubicSpline(tvec,yCoords)

    fig, ax = plt.subplots()
    ax.plot(xCoords, yCoords, label="real" )
    ax.plot(xTraj(tvec), yTraj(tvec), label="spline", linestyle="dashed")
    ax.grid()
    ax.legend()
    plt.show()


