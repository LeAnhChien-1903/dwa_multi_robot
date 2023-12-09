#!/usr/bin/env python
import cv2
import rospy
import numpy as np
import math
class Parameters:
    def __init__(self):
        self.alpha = 0.8
        self.beta = 1.0
        self.gamma = 0.5
        self.delta = 0.5
        self.max_linear_acceleration = 5.0
        self.max_angular_acceleration = 5.0
        self.min_linear_velocity = -1.0
        self.max_linear_velocity = 1.0
        self.min_angular_velocity = -1.5
        self.max_angular_velocity = 1.5
        self.time_trajectory = 0.3
        self.trajectory_sample = 3
        self.velocity_sample = 7
        self.sample_time = 0.1
        self.goal_tolerance = 0.05
        self.map_name: str = "office"
        self.laser_topic: str = "base_scan"
        self.position_topic: str = "base_pose_ground_truth"
        self.velocity_topic: str = "odom"
        self.cmd_vel_topic: str = "cmd_vel"
        self.robot_width: float =  0.6
        self.robot_length: float = 0.9
        self.resolution: float = 0.025
        self.origin_x: float = -10.0
        self.origin_y: float = -10.0
        self.radius: float = 0.6
        self.initialize()
    
    def initialize(self):
        '''
            Initialize the parameters from ros param
        '''
        self.alpha = rospy.get_param("/alpha")
        self.beta = rospy.get_param("/beta")
        self.gamma = rospy.get_param("/gamma")
        self.delta = rospy.get_param("/delta")
        
        self.max_linear_acceleration = rospy.get_param("/max_linear_acceleration")
        self.max_angular_acceleration = rospy.get_param("/max_angular_acceleration")
        self.min_linear_velocity = rospy.get_param("/min_linear_velocity")
        self.max_linear_velocity = rospy.get_param("/max_linear_velocity")
        self.min_angular_velocity = rospy.get_param("/min_angular_velocity")
        self.max_angular_velocity = rospy.get_param("/max_angular_velocity")
        
        self.time_trajectory = rospy.get_param("/time_trajectory")
        self.sample_time = rospy.get_param("/sample_time")
        self.trajectory_sample = rospy.get_param("/trajectory_sample")
        self.velocity_sample = rospy.get_param("/velocity_sample")
        
        self.laser_topic = rospy.get_param("/laser_topic")
        self.position_topic = rospy.get_param("/position_topic")
        self.velocity_topic = rospy.get_param("/velocity_topic")
        self.cmd_vel_topic = rospy.get_param("/cmd_vel_topic")
        
        self.robot_width = rospy.get_param("/robot_width")
        self.robot_length = rospy.get_param("/robot_length")
        self.radius = round(math.hypot(self.robot_length/2,  self.robot_width/2), 1)
        
        self.map_name = rospy.get_param("/map_name")
        self.resolution = rospy.get_param("/resolution")
        origin = rospy.get_param("/origin")
        self.origin_x = origin[0]
        self.origin_y = origin[1]
        
        self.goal_tolerance = rospy.get_param("/goal_tolerance")