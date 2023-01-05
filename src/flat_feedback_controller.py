#! /usr/bin/env python3
import rospy 
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDrive
import tf_conversions

import numpy as np
from math import atan2, cos, sin, tan, atan, sqrt
from scipy.interpolate import interp1d, CubicSpline
import matplotlib as mpl
from matplotlib import pyplot as plt

class Flat_Controller:
    
    def __init__(self, ref_traj, gain, tmax) -> None:
        rospy.loginfo("Initialized Flat Controller")
        rospy.sleep(2)
        self.t0 = rospy.get_time()
        self.vel_prev = 0 # 0 initial velocity
        self.t_prev = 0 # initial time
        
        self.L = 0.324
        self.tvec = []
        self.x_pose = []
        self.y_pose = []
        self.control1_vec = []
        self.control2_vec = []
        self.control3_vec = []
        #  Make a copy of the reference trajectory and construct interpolation functions
        #  to get the value of the function at any time instance t
        self.ref = ref_traj
        self.f1 = interp1d(self.ref[0,:], self.ref[1,:])
        self.f2 = interp1d(self.ref[0,:], self.ref[2,:])
        self.f3 = interp1d(self.ref[0,:], self.ref[3,:])
        self.f4 = interp1d(self.ref[0,:], self.ref[4,:])
        self.f5 = interp1d(self.ref[0,:], self.ref[5,:])
        self.f6 = interp1d(self.ref[0,:], self.ref[6,:])

        self.gain = gain
        self.tmax = tmax
        self.drive_msg = AckermannDrive()
        self.driver = rospy.Publisher(name="/car_1/command", data_class=AckermannDrive, queue_size=5)
        self.feedback = rospy.Subscriber(name="/car_1/base/odom", data_class=Odometry, queue_size=1, callback=self.traj_track)

    def traj_track(self, odom):
        self.ti = rospy.get_time() - self.t0
        # rospy.loginfo(self.t0)
        rospy.loginfo(self.ti)
        # rospy.loginfo(self.tmax - self.ti)
        if self.ti > self.tmax:
            self.drive_msg.speed = 0
            self.drive_msg.acceleration = 0
            self.drive_msg.steering_angle = 0

            self.driver.publish(self.drive_msg)
            x_ref = self.ref[1,:]
            y_ref = self.ref[4,:]

            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.plot(x_ref, y_ref, label = "reference")
            ax1.plot(self.x_pose, self.y_pose, label = "executed")
            ax1.set_title("executed trajectory")
            ax1.legend()
            ax1.grid()

            ax2.plot(self.tvec, self.control1_vec, label= "acceleration")
            ax2.plot(self.tvec, self.control2_vec, label= "phi")
            ax2.plot(self.tvec, self.control3_vec, label= "speed")
            ax2.set_title("control signals")
            ax2.legend()
            ax2.grid()

            plt.show()

            rospy.signal_shutdown("Reached Traj end time")
        
       
        v = odom.twist.twist.linear
        ang_vel = odom.twist.twist.angular
        pose = odom.pose.pose.position
        self.x_pose.append(pose.x)
        self.y_pose.append(pose.y)

        orientation_quat = odom.pose.pose.orientation
        (roll, pitch, yaw) = tf_conversions.transformations.euler_from_quaternion([orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w])
        
        x_ref = self.f1(self.ti)
        x_refd = self.f2(self.ti)
        x_refdd = self.f3(self.ti)

        y_ref = self.f4(self.ti)
        y_refd = self.f5(self.ti)
        y_refdd = self.f6(self.ti)

        e1 = x_ref-pose.x
        e2 = y_ref-pose.y
        e1_dot = x_refd-v.x
        e2_dot = y_refd-v.y
        
        k1 = self.gain[1]
        k2 = self.gain[0]

        k3 = self.gain[1] # kd 
        k4 = self.gain[0] # kp

        z1 = x_refdd+ k1*e1_dot+k2*e1
        z2 = y_refdd+ k3*e2_dot+k4*e2

        vel_fwd = sqrt(v.x**2 + v.y**2)
        dt = self.ti - self.t_prev
        # rospy.loginfo(dt)
        control1 = (z1*cos(yaw) + z2*sin(yaw)) # Acceleration
        control2 = atan((self.L/vel_fwd**2)*(z2*cos(yaw)-z1*sin(yaw))) # Ph
        control3 = self.vel_prev + control1*dt # numerical integration to calculate velocity

        self.drive_msg.acceleration = control1
        self.drive_msg.steering_angle = control2
        self.drive_msg.speed = control3

        self.control1_vec.append(control1)
        self.control2_vec.append(control2)
        self.control3_vec.append(control3)

        self.vel_prev = control3
        self.t_prev = self.ti
        self.tvec.append(self.ti)

        self.driver.publish(self.drive_msg)

        


if __name__ == "__main__":
    rospy.init_node("flat_controller")

    # 5th order polynomial

    x0 = np.array([0,0,0])
    xf = np.array([1.5,0.5,0])
    L = 0.324

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
    tvec = np.linspace(0, tmax, 1001)

    xTrajCS = CubicSpline(tvec, xTraj_normalized)
    yTrajCS = CubicSpline(tvec, yTraj_normalized)

    xTraj = xTrajCS(tvec)
    yTraj = yTrajCS(tvec)

    # xdTraj = np.polyval(np.polyder(xPar,1),tau_vec)
    # ydTraj = np.polyval(np.polyder(yPar,1),tau_vec)

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

    controller = Flat_Controller(ref, gain, tmax)
    rospy.spin()
