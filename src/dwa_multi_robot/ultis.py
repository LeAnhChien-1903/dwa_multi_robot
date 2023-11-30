import cv2
import rospy
import numpy as np


class Parameters:
    def __init__(self):
        self.alpha = 0.8
        self.beta = 1.0
        self.gamma = 0.5
        self.delta = 0.5
        self.max_linear_acceleration = 2.0
        self.max_angular_acceleration = 1.0
        self.min_linear_velocity = -1.0
        self.max_linear_velocity = 1.0
        self.min_angular_velocity = -1.5
        self.max_angular_velocity = 1.5
        self.time_trajectory = 0.3
        self.trajectory_sample = 2
        self.velocity_sample = 7
        self.sample_time = 0.1
    
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
        self.sample_time = rospy.get_param("/sample_time")