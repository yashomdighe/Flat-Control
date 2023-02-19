#! /usr/bin/env python3
import rospy 
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from ackermann_msgs.msg import AckermannDrive
import tf_conversions

import numpy as np
from math import cos, sin, atan, sqrt, atan2
from scipy.interpolate import CubicSpline
from scipy.spatial import transform
import matplotlib as mpl
from matplotlib import pyplot as plt
from std_srvs.srv import SetBool, SetBoolRequest
import sys

class Flat_Controller:
    
    def __init__(self, gain, tmax, xTrajCS, yTrajCS, track, max_speed, trial) -> None:
        rospy.loginfo("Initialized Flat Controller")

        self.track = track
        self.max_speed = max_speed
        self.trial = trial
        rospy.sleep(2)
        self.t0 = rospy.get_time()
        self.vel_prev = 0 # 0 initial velocity
        self.t_prev = 0 # initial time

        self.L = 0.324
        self.tvec = []
        self.x_pose = []
        self.y_pose = []
        self.vel = []
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
            x_ref = self.xTraj(self.tvec)
            y_ref = self.yTraj(self.tvec)

            with open("trials/"+self.track+"_"+self.max_speed+"_trial_"+self.trial+".csv", "w+") as f:
                f.write("tvec,x_ref,y_ref,x_real,y_real,x_error,y_error,vel,x_dot_error,y_dot_error\n")
                for i in range(len(self.tvec)):
                    f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        self.tvec[i],
                        x_ref[i],
                        y_ref[i],
                        self.x_pose[i],
                        self.y_pose[i],
                        self.e1_vec[i],
                        self.e2_vec[i],
                        self.vel[i],
                        self.e1dot_vec[i],
                        self.e2dot_vec[i]
                    ))

            fig, (ax1, ax2, ax3) = plt.subplots(1,3)
            ax1.plot(x_ref, y_ref, label = "reference")
            ax1.plot(self.x_pose, self.y_pose, label = "executed", linestyle="dashed")
            ax1.set_title("executed trajectory")
            ax1.legend()
            ax1.grid()

            ax2.plot(self.tvec, self.vel, label= "speed actual")
            # ax2.plot(self.tvec, self.control2_vec, label= "phi")
            ax2.plot(self.tvec, self.control3_vec, label= "speed ref", linestyle="dashed")
            ax2.set_title("control signals")
            ax2.legend()
            ax2.grid()

            ax3.plot(self.tvec, self.e1_vec, label = "x error")
            ax3.plot(self.tvec, self.e2_vec, label = "y error")
            # ax3.plot(self.tvec, self.e1dot_vec, label = "x_dot error")
            # ax3.plot(self.tvec, self.e2dot_vec, label = "y_dot error")
            ax3.set_title("error vs t")
            ax3.legend()
            ax3.grid()

            n_data = np.ones(len(self.tvec))/self.tvec
            z = np.zeros((len(self.tvec),1))
    
            ref = np.hstack((np.array(x_ref).reshape(len(self.tvec),1), np.array(y_ref).reshape(len(self.tvec),1), z))
            exe = np.hstack((np.array(self.x_pose).reshape(len(self.tvec),1), np.array(self.y_pose).reshape(len(self.tvec),1), z))
            
            sum = 0
            for i in range(ref.shape[0]):
                sum += sqrt((ref[i,0]-exe[i,0])**2+(ref[i,1]-exe[i,1])**2)

            sum /= len(self.tvec)    
            R, rmsd = transform.Rotation.align_vectors(ref, exe, n_data.T)
            print("RMSD: ", rmsd)
            print("RMSE: ", sum)
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

        k3 = self.gain[3] # kd 
        k4 = self.gain[2] # kp

        z1 = x_refdd+k1*e1_dot+k2*e1
        z2 = y_refdd+k3*e2_dot+k4*e2

        vel_fwd = sqrt(v.x**2 + v.y**2)
        dt = self.ti - self.t_prev
        # rospy.loginfo(dt)
        control1 = (z1*cos(yaw) + z2*sin(yaw)) # Acceleration
        control2 = atan((self.L/vel_fwd**2)*(z2*cos(yaw)-z1*sin(yaw))) # Phi
        control3 = self.vel_prev + control1*dt # numerical integration to calculate velocity
        if control3 <= 0:
            control3 = 0.001

        self.drive_msg.acceleration = control1
        self.drive_msg.steering_angle = control2
        self.drive_msg.speed = control3

        self.control1_vec.append(control1)
        self.control2_vec.append(control2)
        self.control3_vec.append(control3)

        self.vel_prev = control3
        self.t_prev = self.ti
        self.tvec.append(self.ti)
        self.vel.append(vel_fwd)

        self.driver.publish(self.drive_msg)

def getTvec(trajectory, maxSpeeds, avgSpeed):
    length = 0
    minSpeed = 0.5
    clip = minSpeed
    clipDelta = 0.1
    clipSpeeds = np.clip(maxSpeeds, minSpeed, clip)
    while np.mean(clipSpeeds) < avgSpeed:
        clip += clipDelta
        clipSpeeds = np.clip(maxSpeeds, minSpeed, clip)
    Tvec = [0]
    lastPt = trajectory[0]
    for pt, speed in zip(trajectory, clipSpeeds):
        dist = ((lastPt[0] - pt[0]) ** 2 + (lastPt[1] - pt[1]) ** 2) ** 0.5
        Tvec.append(Tvec[-1] + dist / speed)
        lastPt = pt
        length+= dist
    Tvec = Tvec[1:]
    # Tvec = np.array(Tvec).reshape(len(Tvec), 1)
    print(f"Track Length -> {length}")
    print(
        f"Expected laptime -> {length/np.mean(clipSpeeds)} with avgSpeed -> {np.mean(clipSpeeds)}"
    )
    print(f"Cumulated laptime -> {Tvec[-1]}")
    # plt.figure()
    # ax = plt.axes(projection="3d")
    # # ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], label="Trajectory")
    # ax.plot3D(traj[:, 0], traj[:, 1], oldSpeeds, label="Speeds")
    # ax.plot3D(traj[:, 0], traj[:, 1], clipSpeeds, label="New Speeds")
    # plt.legend(loc="best")
    # plt.show()
    return Tvec    


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
    track = sys.argv[1]
    max_speed = sys.argv[2]
    trial = sys.argv[3]
    path = "scripts/"+track+"_centerline.csv"
    waypoints = np.genfromtxt(path, dtype=float, delimiter=",") 
    coords = waypoints[:,0:2]
    # perm_speed= waypoints[:,4]
    
    correction_angle = atan2(coords[1,1]-coords[0,1], coords[1,0]-coords[0,0])
    R_z = np.array(
                    [[cos(correction_angle), -sin(correction_angle)],
                    [sin(correction_angle), cos(correction_angle)]])

    corrected_coords = np.dot(coords, R_z)

    tmax = 59
    tvec = np.linspace(0,tmax,len(corrected_coords))
    # tvec = t_space * tmax / t_space[-1]
    # tvec = getTvec(coords, perm_speed, 5)
    # print(tvec)
    xTrajCS = CubicSpline(tvec,corrected_coords[:,0])
    yTrajCS = CubicSpline(tvec,corrected_coords[:,1])

    xTraj = xTrajCS(tvec)
    yTraj = yTrajCS(tvec)

    mapViz = rospy.Publisher("/ref", MarkerArray, queue_size=10, latch=True)


    mapArray = MarkerArray()
    for itr,pt in enumerate(zip(xTraj, yTraj)):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = itr
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = 0.2, 0.2, 0.2
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = 0,1,0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = pt[0]
        marker.pose.position.y = pt[1]
        mapArray.markers.append(marker)
    
    mapViz.publish(mapArray)
    
    tmax = tvec[-1]
    gain = [2.5,3.5, 3, 4]
    # gain = [2, 3] # Golden
    # gain = [10, 12]
    # gain = [1,1]
    # gain = [50, 15]
    controller = Flat_Controller(gain, tmax, xTrajCS, yTrajCS, track, max_speed, trial)
    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        r.sleep()
