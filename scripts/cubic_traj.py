import numpy as np
from scipy.interpolate import CubicSpline, BSpline
import matplotlib as mpl
from matplotlib import pyplot as plt
import math

if __name__ == "__main__":
    x0 = np.array([0,0,0])
    xf = np.array([1.5,0.5,0])
    L = 1

    tau_vec = np.linspace(0, 1,1001)

    AmatY = np.array([[tau_vec[0]**5, tau_vec[0]**4, tau_vec[0]**3, tau_vec[0]**2, tau_vec[0]**1, tau_vec[0]**0],
                    [tau_vec[-1]**5, tau_vec[-1]**4, tau_vec[-1]**3, tau_vec[-1]**2, tau_vec[-1]**1, tau_vec[-1]**0],
                    [5*tau_vec[0]**4, 4*tau_vec[0]**3, 3*tau_vec[0]**2, 2*tau_vec[0]**1, tau_vec[0]**0, 0],
                    [5*tau_vec[-1]**4, 4*tau_vec[-1]**3, 3*tau_vec[-1]**2, 2*tau_vec[-1]**1, tau_vec[-1]**0, 0]])

    AmatX = np.array([[tau_vec[0]**5, tau_vec[0]**4, tau_vec[0]**3, tau_vec[0]**2, tau_vec[0]**1, tau_vec[0]**0],
                    [tau_vec[-1]**5, tau_vec[-1]**4, tau_vec[-1]**3, tau_vec[-1]**2, tau_vec[-1]**1, tau_vec[-1]**0]])

    xPar = np.matmul(np.linalg.pinv(AmatX),np.array([[x0[0]],[xf[0]]]))
    yPar = np.matmul(np.linalg.pinv(AmatY),np.array([[x0[1]],[xf[1]],[x0[2]],[xf[2]]]))

    xPar = np.poly1d(np.squeeze(xPar))
    yPar = np.poly1d(np.squeeze(yPar))

    xTraj_normalized = np.polyval(xPar, tau_vec)
    yTraj_normalized = np.polyval(yPar, tau_vec)

    tmax = 10
    tvec = np.linspace(0, 10, 1001)

    xTrajCS = CubicSpline(tvec, xTraj_normalized)
    yTrajCS = CubicSpline(tvec, yTraj_normalized)

    fig, ax = plt.subplots()
    ax.plot(xTrajCS(tvec), yTrajCS(tvec))
    ax.grid()
    plt.show()
