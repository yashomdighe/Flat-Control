import numpy as np
import matplotlib as mpl 
from matplotlib import pyplot as plt
from math import atan2, cos, sin, tan, atan
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, CubicSpline

def model_ode(t, m0, xTrajCS, yTrajCS, gain, tmax):

    x = m0[0]
    y = m0[1]
    th = m0[2]
    v = m0[3]
    L = 0.324
    if v == 0:
        v = 0.0001


    x_ref = xTrajCS(t)
    x_refd = xTrajCS(t,1)
    x_refdd = xTrajCS(t,2)

    y_ref = yTrajCS(t)
    y_refd = yTrajCS(t,1)
    y_refdd = yTrajCS(t,2)

    e1 = x_ref-x
    e2 = y_ref-y
    e1_dot = x_refd-v*cos(th)
    e2_dot = y_refd-v*sin(th)
    
    k1 = gain[1]
    k2 = gain[0]

    k3 = gain[1] # kd 
    k4 = gain[0] # kp

    z1 = x_refdd+ k1*e1_dot+k2*e1
    z2 = y_refdd+ k3*e2_dot+k4*e2

    control1 = z1*cos(th) + z2*sin(th)
    control2 = atan((L/v**2)*(z2*cos(th)-z1*sin(th)))

    xdot = v*cos(th)
    ydot = v*sin(th)
    thdot = (v/L)*tan(control2)
    vdot = control1

    mdot = [xdot, ydot, thdot, vdot]

    return mdot

def get_Control(tvec, sol, xTrajCS, yTrajCS, gain, tmax):
    
    control1_vec = []
    control2_vec = []
    xsol = sol.y[0]
    ysol = sol.y[1]
    thsol = sol.y[2]
    vsol = sol.y[3] 
    L = 0.324
    for i in range(len(tvec)):
        x = xsol[i]
        y = ysol[i]
        th = thsol[i]
        v = vsol[i]
        t = tvec[i]
        if v == 0:
            v = 0.0001

        x_ref = xTrajCS(t)
        x_refd = xTrajCS(t,1)
        x_refdd = xTrajCS(t,2)

        y_ref = yTrajCS(t)
        y_refd = yTrajCS(t,1)
        y_refdd = yTrajCS(t,2)

        e1 = x_ref-x
        e2 = y_ref-y
        e1_dot = x_refd-v*cos(th)
        e2_dot = y_refd-v*sin(th)

        k1 = gain[1]
        k2 = gain[0]

        k3 = gain[1] # kd 
        k4 = gain[0] # kp

        z1 = x_refdd+ k1*e1_dot+k2*e1
        z2 = y_refdd+ k3*e2_dot+k4*e2

        control1 = z1*cos(th) + z2*sin(th)
        control2 = atan((L/v**2)*(z2*cos(th)-z1*sin(th)))

        control1_vec.append(control1)
        control2_vec.append(control2)
    
    return np.array([control1_vec, control2_vec])

def main():
    
    L = 0.324
    path = "scripts/IMS_centerline.csv"
    waypoints = np.genfromtxt(path, dtype=float, delimiter=",")
    xCoords = waypoints[:,0]
    yCoords = waypoints[:,1]

    tmax = 120
    tvec = np.linspace(0,120,len(xCoords))
    xTrajCS = CubicSpline(tvec,xCoords)
    yTrajCS = CubicSpline(tvec,yCoords)

    xTraj = xTrajCS(tvec)
    yTraj = yTrajCS(tvec)

    xdTraj = xTrajCS(tvec,1)
    ydTraj = yTrajCS(tvec, 1)
    
    xddTraj = xTrajCS(tvec,2)
    yddTraj = yTrajCS(tvec, 2)
    
    thTraj = np.arctan2(ydTraj,xdTraj)
    thdTraj = np.divide(np.multiply(yddTraj, xdTraj)-np.multiply(ydTraj, xddTraj), (np.power(xdTraj,2)+np.power(ydTraj,2)))

    vTraj = np.sqrt(np.power(xdTraj,2)+ np.power(ydTraj,2))
    phiTraj = np.arctan2(L*thdTraj, vTraj)

    ref = np.array([tvec, xTraj, xdTraj, xddTraj, yTraj, ydTraj, yddTraj])
    gain = [1,1]

    m0 = [xTraj[0], yTraj[0], thTraj[0], vTraj[0]]
    kp =1
    kd = 1
    gain = [kp, kd]

    sol = solve_ivp(model_ode, [0, tmax], m0, t_eval=tvec ,args=(xTrajCS, yTrajCS, gain, tmax))
    xsol = sol.y[0]
    ysol = sol.y[1]
    thsol = sol.y[2]
    vsol = sol.y[3]      

    # Post process the solution to get the control signals
    control = get_Control(tvec=tvec, sol= sol, xTrajCS=xTrajCS, yTrajCS=yTrajCS, gain=gain, tmax=tmax)  

    #  Plot the trajectory
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(xTraj, yTraj, label = 'reference trajectory')
    ax1.plot(xsol, ysol, label = 'simulated trajectory', linewidth = 1.3 ,linestyle = 'dashed')
    ax1.set_title("Trajectories")
    ax1.legend()
    ax1.grid()
    
    # Plot v and th vs time
    ax2.plot(sol.t, vTraj, label = 'reference v')
    ax2.plot(sol.t, vsol, label = 'simulated v', linewidth = 1.3 ,linestyle = 'dashed')
    ax2.plot(sol.t, thTraj, label = 'reference th')
    ax2.plot(sol.t, thsol, label = 'simulated th', linewidth = 1.3 ,linestyle = 'dashed')
    ax2.set_title("Velocity and Theta")
    ax2.legend()
    ax2.grid()

    # Plot a and phi vs time
    a_ref = np.divide(2*np.multiply(xdTraj,xddTraj)+2*np.multiply(ydTraj, yddTraj), 2*np.sqrt(np.power(xdTraj,2) + np.power(ydTraj,2)))
    ax3.plot(sol.t, phiTraj, label = 'phi reference')
    ax3.plot(sol.t, control[1], label = 'controller output (phi)', linewidth = 1.3 ,linestyle = 'dashed')
    ax3.plot(sol.t, a_ref, label = 'vdot reference')
    ax3.plot(sol.t, control[0], label = 'controller output (vdot)', linewidth = 1.3 ,linestyle = 'dashed')
    ax3.set_title("Control Signals")
    ax3.legend()
    ax3.grid()
    plt.show()

if __name__ == "__main__":
    main()