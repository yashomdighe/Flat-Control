import numpy as np
import matplotlib as mpl 
from matplotlib import pyplot as plt
from math import atan2, cos, sin, tan, atan
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def model_ode(t, m0, ref, gain, tmax):

    x = m0[0]
    y = m0[1]
    th = m0[2]
    v = m0[3]
    L = 1
    if v == 0:
        v = 0.0001

    f1 = interp1d(ref[0,:], ref[1,:])
    f2 = interp1d(ref[0,:], ref[2,:])
    f3 = interp1d(ref[0,:], ref[3,:])

    f4 = interp1d(ref[0,:], ref[4,:])
    f5 = interp1d(ref[0,:], ref[5,:])
    f6= interp1d(ref[0,:], ref[6,:])

    x_ref = f1(t/tmax)
    x_refd = f2(t/tmax)
    x_refdd = f3(t/tmax)

    y_ref = f4(t/tmax)
    y_refd = f5(t/tmax)
    y_refdd = f6(t/tmax)

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

    xdot = v*cos(th)/tmax
    ydot = v*sin(th)/tmax
    thdot = (v/L)*tan(control2)/tmax
    vdot = control1/tmax

    mdot = [xdot, ydot, thdot, vdot]

    return mdot

def get_Control(tvec, sol, ref, gain, tmax):
    
    control1_vec = []
    control2_vec = []
    xsol = sol.y[0]
    ysol = sol.y[1]
    thsol = sol.y[2]
    vsol = sol.y[3] 
    L = 1

    for i in range(len(tvec)):
        x = xsol[i]
        y = ysol[i]
        th = thsol[i]
        v = vsol[i]
        t = tvec[i]
        if v == 0:
            v = 0.0001
    
        f1 = interp1d(ref[0,:], ref[1,:])
        f2 = interp1d(ref[0,:], ref[2,:])
        f3 = interp1d(ref[0,:], ref[3,:])

        f4 = interp1d(ref[0,:], ref[4,:])
        f5 = interp1d(ref[0,:], ref[5,:])
        f6= interp1d(ref[0,:], ref[6,:])

        x_ref = f1(t/tmax)
        x_refd = f2(t/tmax)
        x_refdd = f3(t/tmax)

        y_ref = f4(t/tmax)
        y_refd = f5(t/tmax)
        y_refdd = f6(t/tmax)

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

        xdot = v*cos(th)/tmax
        ydot = v*sin(th)/tmax
        thdot = (v/L)*tan(control2)/tmax
        vdot = control1/tmax
    
    return np.array([control1_vec, control2_vec])


def main():
    # 5th order polynomial

    x0 = np.array([0,0,0])
    xf = np.array([1.5,0.5,0])
    L = 1

    tmax = 20
    tvec = np.linspace(0, tmax, tmax*1001)
    tau_vec = tvec/tmax

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

    xTraj = np.polyval(xPar, tau_vec)
    yTraj = np.polyval(yPar, tau_vec)

    xdTraj = np.polyval(np.polyder(xPar,1),tau_vec)
    ydTraj = np.polyval(np.polyder(yPar,1),tau_vec)
    
    xddTraj = np.polyval(np.polyder(xPar,2),tau_vec)
    yddTraj = np.polyval(np.polyder(yPar,2),tau_vec)
    
    thTraj = np.arctan2(ydTraj,xdTraj)
    thdTraj = np.divide(np.multiply(yddTraj, xdTraj)-np.multiply(ydTraj, xddTraj), (np.power(xdTraj,2)+np.power(ydTraj,2)))

    vTraj = np.sqrt(np.power(xdTraj,2)+ np.power(ydTraj,2))
    phiTraj = np.arctan2(L*thdTraj, vTraj)

    ref = np.array([tau_vec, xTraj, xdTraj, xddTraj, yTraj, ydTraj, yddTraj])
    gain = [1,1]

    m0 = [xTraj[0], yTraj[0], thTraj[0], vTraj[0]]
    kp =1
    kd = 1
    gain = [kp, kd]

    sol = solve_ivp(model_ode, [0, tmax], m0, t_eval=tau_vec*tmax ,args=(ref, gain, tmax))
    xsol = sol.y[0]
    ysol = sol.y[1]
    thsol = sol.y[2]
    vsol = sol.y[3]      

    # Post process the solution to get the control signals
    control = get_Control(tvec=tvec, sol= sol, ref=ref, gain=gain, tmax=tmax)  

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