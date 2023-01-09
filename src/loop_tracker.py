#! /usr/bin/env python3
import rospy 
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive
import tf_conversions

import numpy as np
from math import cos, sin, atan, sqrt
from scipy.interpolate import CubicSpline
import matplotlib as mpl
from matplotlib import pyplot as plt
from std_srvs.srv import SetBool, SetBoolRequest

class Flat_Controller:
    
    def __init__(self, gain, tmax, xTrajCS, yTrajCS) -> None:
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
        self.e1_vec = []
        self.e2_vec = []
        self.e1dot_vec = []
        self.e2dot_vec = []
        #  Make a copy of the reference trajectory and construct interpolation functions
        #  to get the value of the reference at any time instance t
        self.xTraj = xTrajCS
        self.yTraj = yTrajCS

        self.gain = gain
        self.tmax = tmax
        self.drive_msg = AckermannDrive()
        self.driver = rospy.Publisher(name="/car_1/command", data_class=AckermannDrive, queue_size=1)
        self.feedback = rospy.Subscriber(name="/car_1/base/odom", data_class=Odometry, queue_size=1, callback=self.traj_track)

    def traj_track(self, odom):
        self.ti = rospy.get_time() - self.t0
        # rospy.loginfo(self.t0)
        rospy.loginfo(self.ti)
        # rospy.loginfo(self.tmax - self.ti)
        if self.ti > self.tmax+0.01:
            self.drive_msg.speed = 0
            self.drive_msg.acceleration = 0
            self.drive_msg.steering_angle = 0

            self.driver.publish(self.drive_msg)
            x_ref = self.xTraj(tvec)
            y_ref = self.yTraj(tvec)

            fig, (ax1, ax2, ax3) = plt.subplots(1,3)
            ax1.plot(x_ref, y_ref, label = "reference")
            ax1.plot(self.x_pose, self.y_pose, label = "executed", linestyle="dashed")
            ax1.set_title("executed trajectory")
            ax1.legend()
            ax1.grid()

            ax2.plot(self.tvec, self.control1_vec, label= "acceleration")
            ax2.plot(self.tvec, self.control2_vec, label= "phi")
            ax2.plot(self.tvec, self.control3_vec, label= "speed", linestyle="dashed")
            ax2.set_title("control signals")
            ax2.legend()
            ax2.grid()

            ax3.plot(self.tvec, self.e1_vec, label = "x error")
            ax3.plot(self.tvec, self.e2_vec, label = "y error")
            ax3.plot(self.tvec, self.e1dot_vec, label = "x_dot error")
            ax3.plot(self.tvec, self.e2dot_vec, label = "y_dot error")
            ax3.set_title("error vs t")
            ax3.legend()
            ax3.grid()

            plt.show()

            rospy.signal_shutdown("Reached Traj end time")
        
       
        v = odom.twist.twist.linear
        ang_vel = odom.twist.twist.angular
        pose = odom.pose.pose.position
        self.x_pose.append(pose.x)
        self.y_pose.append(pose.y)

        orientation_quat = odom.pose.pose.orientation
        (roll, pitch, yaw) = tf_conversions.transformations.euler_from_quaternion([orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w], "rxyz")
        
        x_ref = self.xTraj(self.ti)
        x_refd = self.xTraj(self.ti,1)
        x_refdd = self.xTraj(self.ti,2)

        y_ref = self.yTraj(self.ti)
        y_refd = self.yTraj(self.ti,1)
        y_refdd = self.yTraj(self.ti,2)

        # rospy.loginfo(str(x_ref) + " " +str(y_ref))

        e1 = x_ref-pose.x
        e2 = y_ref-pose.y
        e1_dot = x_refd-v.x
        e2_dot = y_refd-v.y

        self.e1_vec.append(e1)
        self.e2_vec.append(e2)

        self.e1dot_vec.append(e1_dot)
        self.e2dot_vec.append(e2_dot)
        
        k1 = self.gain[1]
        k2 = self.gain[0]

        k3 = self.gain[1] # kd 
        k4 = self.gain[0] # kp

        z1 = x_refdd+k1*e1_dot+k2*e1
        z2 = y_refdd+k3*e2_dot+k4*e2

        vel_fwd = sqrt(v.x**2 + v.y**2)
        dt = self.ti - self.t_prev
        # rospy.loginfo(dt)
        control1 = (z1*cos(yaw) + z2*sin(yaw)) # Acceleration
        control2 = atan((self.L/vel_fwd**2)*(z2*cos(yaw)-z1*sin(yaw))) # Phi
        control3 = self.vel_prev + control1*dt # numerical integration to calculate velocity

        # self.drive_msg.acceleration = control1
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
    
    rospy.wait_for_service("/reset_car")
    reset = rospy.ServiceProxy("/reset_car", SetBool)
    try:
        reset_status = reset(True)
        rospy.sleep(1.2)
    except rospy.ServiceException as e:
        rospy.signal_shutdown("Could not reset world")

    L = 0.324
    path = "scripts/IMS_centerline.csv"
    waypoints = np.genfromtxt(path, dtype=float, delimiter=",")
    xCoords = waypoints[:,0]
    yCoords = waypoints[:,1]

    tmax = 35
    tvec = np.linspace(0,tmax,len(xCoords))
    xTrajCS = CubicSpline(tvec,-yCoords)
    yTrajCS = CubicSpline(tvec,xCoords)

    xTraj = xTrajCS(tvec)
    yTraj = yTrajCS(tvec)

    gain = [2, 3] # Golden
    # gain = [10, 12]
    # gain = [1,1]
    # gain = [50, 15]
    controller = Flat_Controller(gain, tmax, xTrajCS, yTrajCS)
    r = rospy.Rate(400)
    while not rospy.is_shutdown():
        r.sleep()
