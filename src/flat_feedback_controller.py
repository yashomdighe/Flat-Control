#! /usr/bin/env python3
import rospy 
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDrive
import tf_conversions

import numpy as np
from math import atan2, cos, sin, tan, atan
from scipy.interpolate import interp1d

class Flat_Controller:
    
    def __init__(self, ref_traj) -> None:
        rospy.loginfo("Initialized Flat Controller")

        self.t0 = rospy.Time.now().to_sec()

        #  Make a copy of the reference trajectory and construct interpolation functions
        #  to get the value of the function at any time instance t
        self.ref = ref_traj
        self.f1 = interp1d(self.ref[0,:], self.ref[1,:])
        self.f2 = interp1d(self.ref[0,:], self.ref[2,:])
        self.f3 = interp1d(self.ref[0,:], self.ref[3,:])
        self.f4 = interp1d(self.ref[0,:], self.ref[4,:])
        self.f5 = interp1d(self.ref[0,:], self.ref[5,:])
        self.f6 = interp1d(self.ref[0,:], self.ref[6,:])

        self.drive_msg = AckermannDrive()
        self.driver = rospy.Publisher(name="/car1/command", data_class=AckermannDrive, queue_size=5)
        self.feedback = rospy.Subscriber(name="/car_1/base/odom", data_class=Odometry, queue_size=1, callback=self.traj_track)

    def traj_track(self, odom):
        self.ti = rospy.Time.now().to_sec() - self.t0
        rospy.loginfo(self.ti)


if __name__ == "__main__":
    rospy.init_node("flat_controller")

    #  Create a 5th order trajectory

    x0 = np.array([0,0,0])
    xf = np.array([1.5,0.5,0])
    L = 1

    tmax = 1
    tvec = np.linspace(0, tmax, tmax*1001)
    AmatY = np.array([[tvec[0]**5, tvec[0]**4, tvec[0]**3, tvec[0]**2, tvec[0]**1, tvec[0]**0],
                    [tvec[-1]**5, tvec[-1]**4, tvec[-1]**3, tvec[-1]**2, tvec[-1]**1, tvec[-1]**0],
                    [5*tvec[0]**4, 4*tvec[0]**3, 3*tvec[0]**2, 2*tvec[0]**1, tvec[0]**0, 0],
                    [5*tvec[-1]**4, 4*tvec[-1]**3, 3*tvec[-1]**2, 2*tvec[-1]**1, tvec[-1]**0, 0]])

    AmatX = np.array([[tvec[0]**5, tvec[0]**4, tvec[0]**3, tvec[0]**2, tvec[0]**1, tvec[0]**0],
                    [tvec[-1]**5, tvec[-1]**4, tvec[-1]**3, tvec[-1]**2, tvec[-1]**1, tvec[-1]**0]])

    xPar = np.matmul(np.linalg.pinv(AmatX),np.array([[x0[0]],[xf[0]]]))
    yPar = np.matmul(np.linalg.pinv(AmatY),np.array([[x0[1]],[xf[1]],[x0[2]],[xf[2]]]))

    xPar = np.poly1d(np.squeeze(xPar))
    yPar = np.poly1d(np.squeeze(yPar))

    xTraj = np.polyval(xPar, tvec)
    yTraj = np.polyval(yPar, tvec)

    xdTraj = np.polyval(np.polyder(xPar,1),tvec)
    ydTraj = np.polyval(np.polyder(yPar,1),tvec)
    
    xddTraj = np.polyval(np.polyder(xPar,2),tvec)
    yddTraj = np.polyval(np.polyder(yPar,2),tvec)
    a_ref = np.sqrt(np.power(xddTraj,2) + np.power(yddTraj,2))

    thTraj = np.arctan2(ydTraj,xdTraj)
    thdTraj = np.divide(np.multiply(yddTraj, xdTraj)-np.multiply(ydTraj, xddTraj), (np.power(xdTraj,2)+np.power(ydTraj,2)))

    vTraj = np.sqrt(np.power(xdTraj,2)+ np.power(ydTraj,2))
    phiTraj = np.arctan2(L*thdTraj, vTraj)

    ref = np.array([tvec, xTraj, xdTraj, xddTraj, yTraj, ydTraj, yddTraj])

    controller = Flat_Controller(ref)
    rospy.spin()
