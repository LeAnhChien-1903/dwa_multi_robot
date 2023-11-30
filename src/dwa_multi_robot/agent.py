#!/usr/bin/env python

import numpy as np
import os

import rospy
from geometry_msgs.msg import Twist, Point, PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from dwa_multi_robot.ultis import *
from dwa_multi_robot.astar import *

map_path = "/home/leanhchien/deep_rl_ws/src/dwa_multi_robot/slam"

class Agent:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.robot_id = int(robot_name.split('_')[1])
        self.params: Parameters = Parameters()
        self.astar: AStarPlanner = AStarPlanner(os.path.join(map_path, self.params.map_name + ".png"),
                                                self.params.origin_x, self.params.origin_y, 
                                                self.params.resolution, self.params.radius)
        self.robot_poses: np.ndarray = np.zeros((8, 3))
        self.robot_velocities: np.ndarray = np.zeros((8, 2))
        self.setGoalPoseAndColor()
        self.initializeSubscriberAndPublisher()
        self.set_path = False
        self.global_path_msg = Path()
    
    def timerCallback(self, event):
        self.setGlobalPath()
        self.path_pub.publish(self.global_path_msg)
        self.visualization()
    
    def setGlobalPath(self):
        # Global path planning
        global_condition = self.set_path == False
        global_condition = global_condition and self.robot_poses[self.robot_id, 0] != 0.0
        global_condition = global_condition and self.robot_poses[self.robot_id, 1] != 0.0
        # global_condition = global_condition and self.robot_poses[self.robot_id, 2] != 0.0
        if global_condition == True:
            self.global_path, _ = self.astar.planning(self.robot_poses[self.robot_id, :], self.goal_pose)
            self.set_path = True
            self.global_path_msg.header.frame_id = "map"
            self.global_path_msg.header.stamp = rospy.Time.now()
            for i in range(self.global_path.shape[0]):
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.header.seq = i
                pose.pose.position.x = self.global_path[i, 0]
                pose.pose.position.y = self.global_path[i, 1]
                self.global_path_msg.poses.append(pose)
    def getRobotPoses(self, odom: Odometry):
        id = int(odom.header.frame_id.split("/")[1].split("_")[1])
        # Get current robot pose
        self.robot_poses[id, 0] = odom.pose.pose.position.x
        self.robot_poses[id, 1] = odom.pose.pose.position.y
        
        rqy = euler_from_quaternion([odom.pose.pose.orientation.x, 
                                    odom.pose.pose.orientation.y, 
                                    odom.pose.pose.orientation.z, 
                                    odom.pose.pose.orientation.w])
        self.robot_poses[id, 2] = rqy[2]
        # Get current velocity
        self.robot_velocities[id, 0] = odom.twist.twist.linear.x
        self.robot_velocities[id, 1] = odom.twist.twist.angular.z

    def setGoalPoseAndColor(self):
        id = int(self.robot_name.split("_")[1])
        if id == 0:
            self.goal_pose = np.array([9.0, -4.8, 0.0])
            self.color = (1.0, 0.0, 0.0)
        elif id == 1:
            self.goal_pose = np.array([9.0, 4.8, 0.0])
            self.color = (0.0, 1.0, 0.0)
        elif id == 2:
            self.goal_pose = np.array([-9.0, -4.8, math.pi])
            self.color = (0.0, 0.0, 1.0)
        elif id == 3:
            self.goal_pose = np.array([-9.0, 4.8, math.pi])
            self.color = (1.0, 1.0, 0.0)
        elif id == 4:
            self.goal_pose = np.array([-4.8, -9.0, -math.pi/2])
            self.color = (1.0, 0.0, 1.0)
        elif id == 5:
            self.goal_pose = np.array([4.8, -9.0, -math.pi/2])
            self.color = (0.0, 1.0, 1.0)
        elif id == 6:
            self.goal_pose = np.array([-4.8, 9.0, math.pi/2])
            self.color = (0.0, 0.0, 0.0)
        elif id == 7:
            self.goal_pose = np.array([4.8, 9.0, math.pi/2])
            self.color = (0.0, 1.0, 0.0)
            
    def initializeSubscriberAndPublisher(self):
        self.odometry_subs = []
        for i in range(8):
            self.odometry_subs.append(rospy.Subscriber("/robot_{}/{}".format(i, self.params.odometry_topic), Odometry, self.getRobotPoses, queue_size=10))
        self.cmd_vel_pub = rospy.Publisher(self.robot_name + "/" + self.params.cmd_vel_topic, Twist, queue_size= 10)
        self.markers_pub = rospy.Publisher(self.robot_name + "/visualize", MarkerArray, queue_size= 10)
        self.path_pub = rospy.Publisher(self.robot_name + "/global_path", Path, queue_size= 10)
        self.timer = rospy.Timer(rospy.Duration(self.params.sample_time), self.timerCallback)
        
        self.path_marker = Marker()
        self.path_marker.header.stamp = rospy.Time.now()
        self.path_marker.header.frame_id = "map"
        self.path_marker.ns = "robot_path"
        self.path_marker.action = self.path_marker.ADD
        self.path_marker.type = self.path_marker.LINE_STRIP
        
        self.path_marker.pose.orientation.x = 0.0
        self.path_marker.pose.orientation.y = 0.0
        self.path_marker.pose.orientation.z = 0.0
        self.path_marker.pose.orientation.w = 1.0
    
        self.path_marker.scale.x = 0.05
        self.path_marker.scale.y = 0.05
        self.path_marker.scale.z = 0.05
        
        self.path_marker.color.r = self.color[0]
        self.path_marker.color.g = self.color[1]
        self.path_marker.color.b = self.color[2]
        self.path_marker.color.a = 1.0
    
    def visualization(self):
        visual_markers = MarkerArray()
        robot_marker = Marker()
        robot_marker.header.stamp = rospy.Time.now()
        robot_marker.header.frame_id = "map"
        robot_marker.ns = "robot_position"
        robot_marker.action = robot_marker.ADD
        robot_marker.type = robot_marker.CUBE
        
        robot_marker.pose.position.x = self.robot_poses[self.robot_id, 0]
        robot_marker.pose.position.y = self.robot_poses[self.robot_id, 1]
        robot_marker.pose.position.z = 0.2
        
        q = quaternion_from_euler(0, 0, self.robot_poses[self.robot_id, 2])
        robot_marker.pose.orientation.x = q[0]
        robot_marker.pose.orientation.y = q[1]
        robot_marker.pose.orientation.z = q[2]
        robot_marker.pose.orientation.w = q[3]
        
        robot_marker.scale.x = self.params.robot_length
        robot_marker.scale.y = self.params.robot_width
        robot_marker.scale.z = 0.4
        
        robot_marker.color.r = self.color[0]
        robot_marker.color.g = self.color[1]
        robot_marker.color.b = self.color[2]
        robot_marker.color.a = 1.0
        
        visual_markers.markers.append(robot_marker)
        
        goal_marker = Marker()
        goal_marker.header.stamp = rospy.Time.now()
        goal_marker.header.frame_id = "map"
        goal_marker.ns = "robot_goal"
        goal_marker.action = goal_marker.ADD
        goal_marker.type = goal_marker.CUBE
        
        goal_marker.pose.position.x = self.goal_pose[0]
        goal_marker.pose.position.y = self.goal_pose[1]
        goal_marker.pose.position.z = 0.2
        
        q = quaternion_from_euler(0, 0, self.goal_pose[2])
        goal_marker.pose.orientation.x = q[0]
        goal_marker.pose.orientation.y = q[1]
        goal_marker.pose.orientation.z = q[2]
        goal_marker.pose.orientation.w = q[3]
        
        goal_marker.scale.x = self.params.robot_length
        goal_marker.scale.y = self.params.robot_width
        goal_marker.scale.z = 0.4
        
        goal_marker.color.r = self.color[0]
        goal_marker.color.g = self.color[1]
        goal_marker.color.b = self.color[2]
        goal_marker.color.a = 1.0
        
        visual_markers.markers.append(goal_marker)

        p = Point()
        if self.robot_poses[self.robot_id, 0] != 0.0 and self.robot_poses[self.robot_id, 1] != 0.0:
            p.x = self.robot_poses[self.robot_id, 0]
            p.y = self.robot_poses[self.robot_id, 1]
            self.path_marker.points.append(p)
            visual_markers.markers.append(self.path_marker)
        
        self.markers_pub.publish(visual_markers)

