import numpy as np
from matplotlib import pyplot as plt
from math import atan2, cos, sin, tan, atan, sqrt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, CubicSpline
import sys

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_turns(arr):
    p1 = None
    p2 = None
    idx_list = []
    for idx in range(arr.shape[0]):
        if arr[idx,0] > 0.03 :
            if not p1:
                p1 = idx
                continue
            if p1:
                for jdx in range(idx+1, arr.shape[0]): 
                    if arr[jdx] <= 0.03:
                        p2 = jdx 
                        idx_list.append((p1,p2))
                        break
                 
        p1 = None
        p2 = None 
    
    return(idx_list)       

def get_curvature_radius(R, idx, vmax):
    c_r = []
    count = 0
    for i in range(R.shape[0]):
        # print(1/R[i])
        c_r.append(1/abs(R[i,0]))
    #     if i in idx:
    #         c_r.append(1/R[i,0])
    #         count += 1
    #     else: c_r.append(0)
    # print(count)
    c_r = np.array(c_r).reshape(-1,1)
    print(c_r.min(), c_r.max())
    return np.clip(c_r, 0, 50)

if __name__ == "__main__":
    track = sys.argv[1]
    vmax = int(sys.argv[2])
    path = "scripts/"+track+"_centerline.csv"
    waypoints = np.genfromtxt(path, dtype=float, delimiter=",")
    xCoords = waypoints[:,0]
    yCoords = waypoints[:,1]

    tmax = 100
    tvec = np.linspace(0,tmax,len(xCoords))

    correction_angle = -atan2(yCoords[1]-yCoords[0], xCoords[1]-xCoords[0])
    R_z = np.array(
                    [[cos(correction_angle), -sin(correction_angle)],
                    [sin(correction_angle), cos(correction_angle)]])
    coords = np.array([xCoords, yCoords]).T

    coords = np.dot(coords, R_z)

    xTraj = CubicSpline(tvec,coords[:,0])
    yTraj = CubicSpline(tvec,coords[:,1])

    x = xTraj(tvec).reshape(-1,1)
    y = yTraj(tvec).reshape(-1,1)
    
    xd = xTraj(tvec,1).reshape(-1,1)
    yd = yTraj(tvec,1).reshape(-1,1)



    xdd = xTraj(tvec,2).reshape(-1,1)
    ydd = yTraj(tvec,2).reshape(-1,1)

    
    D1 = xd*ydd
    D2 = yd*xdd
    N = (xd**2 + yd**2)**1.5
    D = D1-D2
    Kappa = D/N
    print(np.around(abs(Kappa),8).min())
    r_z = []
    turn_idx = get_turns(Kappa)
    for i in range(Kappa.shape[0]):
        if Kappa[i,0] > 0.0003:
            r_z.append(i)
    # r_z = np.where(Kappa > 0.03)
    print(len(r_z))
    c_r = get_curvature_radius(Kappa, r_z, vmax)
    
    # print(c_r)
    turns_X = []
    turns_Y = []
    for idx in turn_idx:
        turns_X.append(coords[idx[0]:idx[1],0][0])
        turns_Y.append(coords[idx[0]:idx[1],1][0])

    z = np.hstack((xd, yd))
    # print(z.shape)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x, y, c_r)
    # ax = plt.figure().add_subplot()
    # ax.plot(tvec, R, label="R" )
    # ax.plot(tvec, abs(D1-D2))
    # ax.scatter(turns_X, turns_Y)
    # ax.plot(coords[:,0], coords[:,1], label="track" )
    # ax.scatter(coords[:,0], diff )
    # ax.plot(tvec, xTrajdd, label="x_dd")
    # ax.plot(tvec, xTrajddd, label="x_ddd" )
    ax.grid()
    # ax.legend()
    plt.show()