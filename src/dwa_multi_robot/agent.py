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
        self.cost_map = 1 - cv2.GaussianBlur(self.astar.cost_map.copy(), (5, 5), 2)/255
        self.robot_states: np.ndarray = np.zeros((8, 5))
        self.setGoalPoseAndColor()
        self.initializeSubscriberAndPublisher()
        self.calculateNonHolonomicAcceleration()
        
        self.set_path = False
        self.global_path_msg = Path()
        self.local_goal_index: int = 0
        self.global_index: int = 0
    
    def timerCallback(self, event):
        self.setGlobalPath()
        if self.set_path == True:
            self.path_pub.publish(self.global_path_msg)
            if calculatedDistance(self.robot_states[self.robot_id, 0:2], self.goal_pose[0:-1]) < self.params.goal_tolerance:
                cmd_vel = Twist()
                self.cmd_vel_pub.publish(cmd_vel)
            else:
                local_goal = self.calculateLocalGoal()
                cost = np.zeros(self.params.velocity_sample**2)
                new_vel = self.calculateNewVelocity(self.robot_states[self.robot_id, 3:5], self.params.sample_time)
                new_trajectories = self.calculateAllNewTrajectories(self.robot_states[self.robot_id], new_vel)
                progress_cost = self.calculateProgressCost(new_trajectories,local_goal)
                
                for i in range(cost.shape[0]):
                    grid_cost = self.calculateGridClearance(new_trajectories[i])
                    time_collision = self.calculateTimeCollision(new_trajectories[i])
                    progress = progress_cost[i]
                    cost[i] = self.params.alpha * grid_cost  + self.params.gamma * progress + self.params.beta * time_collision
                # print("Cost of {}:{}".format(self.robot_name, cost.max()))
                max_index = cost.argmax()
                # print(cost)
                cmd_vel = Twist()
                cmd_vel.linear.x = new_vel[max_index, 0]
                cmd_vel.angular.z = new_vel[max_index, 1]
                self.cmd_vel_pub.publish(cmd_vel)
                self.visualization() 
        
    def calculateLocalGoal(self):
        self.updateGlobalIndex(20)
        sum_distance = 0.0
        i = self.global_index
        while sum_distance < 2:
            if i == self.global_path.shape[0] - 1:
                return self.goal_pose[0:-1]
            sum_distance += calculatedDistance(self.global_path[i], self.global_path[i+1])
            i = i + 1
        self.local_goal_index = i
        return self.global_path[self.local_goal_index]
    
    def updateGlobalIndex(self, num_of_check: int = 20):
        current_position = self.robot_states[self.robot_id, 0:2]
        min_distance = float('inf')
        
        for i in range(num_of_check):
            index = self.global_index + i
            if index > self.global_path.shape[0] - 1:
                break
            distance = calculatedDistance(current_position, self.global_path[index])
            if distance < min_distance:
                self.global_index = index
                min_distance = distance
    
    def calculateTimeCollision(self, robot_trajectory: np.ndarray):
        time_collision_cost = []
        for id in range(self.robot_states.shape[0]):
            if id == self.robot_id:
                continue
            else:
                distance = calculatedDistance(self.robot_states[self.robot_id, 0:2], self.robot_states[id, 0:2])
                if distance > 3 or (self.robot_states[id, 0] == 0.0 and self.robot_states[id, 1] == 0.0 and self.robot_states[id, 2] == 0.0) :
                    continue
                else:
                    other_trajectory = self.calculateTrajectory(self.robot_states[id])
                    # print(other_trajectory)
                    time_collision_cost.append(self.calculateTimeCollisionWithRobots(robot_trajectory, other_trajectory))
        
        if len(time_collision_cost) > 0:
            return min(time_collision_cost)
        else:
            return self.params.time_trajectory

    def calculateTimeCollisionWithRobots(self, robot_trajectory: np.ndarray, other_trajectory: np.ndarray):
        current_state = self.robot_states[self.robot_id, 0:2].copy()
        for i in range(robot_trajectory.shape[0]):
            bounding_box = self.robotBoundingBox(other_trajectory[i, 0:3], 
                                                self.params.robot_width + 2 * self.params.radius + 0.05,
                                                self.params.robot_length + 2 * self.params.radius + 0.05)
            if calculatedDistance(robot_trajectory[i, 0:2], other_trajectory[i, 0: 2]) < self.params.radius * 2 + 0.05:
                return self.params.sample_time * i
            
            flag, intersection = self.checkLinePolygonIntersection(current_state, robot_trajectory[i, 0:2], bounding_box)
            if flag == True:
                time_collision = self.params.sample_time * i + self.params.sample_time * calculatedDistance(intersection, current_state) / calculatedDistance(current_state, robot_trajectory[i, 0:2])
                return time_collision
            current_state = robot_trajectory[i, 0:2].copy()
        
        return self.params.time_trajectory
    
    def calculateProgressCost(self, new_trajectories: np.ndarray, goal: np.ndarray):
        distance_to_goal = np.zeros(new_trajectories.shape[0])
        
        for i in range(new_trajectories.shape[0]):
            distance_to_goal[i] = calculatedDistance(new_trajectories[i, -1, 0:2], goal)
        
        max_distance = distance_to_goal.max()
        
        return 1 - distance_to_goal/max_distance

    def calculateGridClearance(self, trajectory: np.ndarray):
        max_grid = []
        for i in range(trajectory.shape[0]):
            point_pixel = self.astar.convertMeterToPixel(trajectory[i, 0:2])
            max_grid.append(self.getMaxGridInPoint(point_pixel, 2))
    
        return -max(max_grid)
    
    def getMaxGridInPoint(self, point: np.ndarray, num_of_pixel_check: int = 2):
        grid = []
        for x_add in range(num_of_pixel_check):
            for y_add in range(num_of_pixel_check):
                # collision check
                grid.append(self.cost_map[point[1] + y_add, point[0] + x_add])

        return max(grid)
    
    def calculateAllNewTrajectories(self, current_state: np.ndarray, new_vel: np.ndarray):
        new_trajectories = np.zeros((new_vel.shape[0], self.params.trajectory_sample, current_state.shape[0]))
        for i in range(new_vel.shape[0]):
            new_trajectories[i] = self.calculateNewTrajectory(current_state, new_vel[i])
        return new_trajectories
    
    def calculateNewTrajectory(self, current_state: np.ndarray, new_vel: np.ndarray):
        new_trajectory = np.zeros((self.params.trajectory_sample, current_state.shape[0]))
        new_trajectory[0] = self.calculateNewStateWithNewVelocity(current_state, new_vel, self.params.sample_time)
        for i in range(1, self.params.trajectory_sample):
            new_trajectory[i] = self.calculateNewState(new_trajectory[i-1], self.params.sample_time)
            
        return new_trajectory
    
    def calculateTrajectory(self, current_state: np.ndarray):
        new_trajectory = np.zeros((self.params.trajectory_sample, current_state.shape[0]))
        new_trajectory[0] = self.calculateNewState(current_state, self.params.sample_time)
        for i in range(1, self.params.trajectory_sample):
            new_trajectory[i] = self.calculateNewState(new_trajectory[i-1], self.params.sample_time)
            
        return new_trajectory
    
    def calculateNewState(self, current_state: np.ndarray, dt: float):
        new_state = current_state.copy()
        new_state[2] = new_state[2] + new_state[4] * dt
        new_state[0] = new_state[0] + new_state[3] * math.cos(new_state[2]) * dt
        new_state[1] = new_state[1] + new_state[3] * math.sin(new_state[2]) * dt
        
        return new_state
    
    def calculateNewStateWithNewVelocity(self, current_state: np.ndarray, new_vel: np.ndarray, dt: float):
        new_state = current_state.copy()
        
        new_state[2] = new_state[2] + new_vel[1] * dt
        new_state[0] = new_state[0] + new_vel[0] * math.cos(new_state[2]) * dt
        new_state[1] = new_state[1] + new_vel[0] * math.sin(new_state[2]) * dt
        new_state[3] = new_vel[0]
        new_state[4] = new_vel[1]
        
        return new_state
    
    def calculateNewVelocity(self, current_vel: np.ndarray, t: float):
        new_vel = current_vel + self.acceleration * t * self.params.delta
        for i in range(new_vel.shape[0]):
            if new_vel[i, 0] > self.params.max_linear_velocity:
                new_vel[i, 0] = self.params.max_linear_velocity
            if new_vel[i, 0] < self.params.min_linear_velocity:
                new_vel[i, 0] = self.params.min_linear_velocity
            if new_vel[i, 1] > self.params.max_angular_velocity:
                new_vel[i, 1] = self.params.max_angular_velocity
            if new_vel[i, 1] < self.params.min_angular_velocity:
                new_vel[i, 1] = self.params.min_angular_velocity
        
        return np.round(new_vel, 2)
    
    def calculateHolonomicAcceleration(self):
        self.acceleration = np.zeros((self.params.velocity_sample ** 2, 2))
        counter = 0
        for i in range(self.params.velocity_sample):
            for j in range(self.params.velocity_sample):
                self.acceleration[counter, 0] = - self.params.max_linear_acceleration + i * (2*self.params.max_linear_acceleration / (self.params.velocity_sample-1))
                self.acceleration[counter, 1] = - self.params.max_linear_acceleration + j * (2*self.params.max_linear_acceleration / (self.params.velocity_sample-1))
                counter += 1
    
    def calculateNonHolonomicAcceleration(self):
        self.acceleration = np.zeros((self.params.velocity_sample ** 2, 2))
        counter = 0
        for i in range(self.params.velocity_sample):
            for j in range(self.params.velocity_sample):
                self.acceleration[counter, 0] = - self.params.max_linear_acceleration + i * (2*self.params.max_linear_acceleration / (self.params.velocity_sample-1))
                self.acceleration[counter, 1] = - self.params.max_angular_acceleration + j * (2*self.params.max_angular_acceleration / (self.params.velocity_sample-1))
                counter += 1
    
    @staticmethod
    def checkLineIntersection(line1_start: np.ndarray, line1_end: np.ndarray, line2_start: np.ndarray, line2_end: np.ndarray):
        line1 = line1_end - line1_start
        line2 = line2_end - line2_start
        denom = line1[0] * line2[1] - line2[0] * line1[1]
        
        if (denom == 0): 
            return False, None
        denomPositive = denom > 0
        
        aux = line1_start - line2_start
        
        s_numer = line1[0] * aux[1] - line1[1] * aux[0]
        
        if (s_numer < 0) == denomPositive: 
            return False, None
        
        t_numer = line2[0] * aux[1] - line2[1] * aux[0]
        
        if (t_numer < 0) == denomPositive: 
            return False, None
        
        if ((s_numer > denom) == denomPositive) or ((t_numer > denom) == denomPositive): 
            return False, None
        t = t_numer / denom
        intersection = line1_start + t * line1
        
        return True, intersection
    
    def checkLinePolygonIntersection(self, line_start: np.array, line_end: np.array, polygon: np.array):
        collision_points = []
        for i in range(polygon.shape[0]):
            if i == polygon.shape[0] - 1:
                flag_, point = self.checkLineIntersection(line_start, line_end, polygon[i], polygon[0])
                if flag_ == True:
                    collision_points.append(point)
            else:
                flag_, point = self.checkLineIntersection(line_start, line_end, polygon[i], polygon[i+1])
                if flag_ == True:
                    collision_points.append(point)
        
        if len(collision_points) > 0 :
            min_point = collision_points[0]
            min_distance = calculatedDistance(line_start, min_point)
            for i in range(1, len(collision_points)):
                distance = calculatedDistance(line_start, collision_points[i])
                if distance < min_distance:
                    min_distance = distance
                    min_point = collision_points[i]
            return True, min_point
        else:
            return False, None
        
    @staticmethod
    def robotBoundingBox(robot_state: np.ndarray, width: float, length: float):
        bounding_box = np.zeros((4, 2))
        x = robot_state[0]
        y = robot_state[1]
        half_length = width / 2
        half_width = length / 2
        sin_angle = math.sin(robot_state[2])
        cos_angle = math.cos(robot_state[2])

        # Bottom left
        bounding_box[0, 0] = x + (cos_angle * -half_length) - (sin_angle * half_width)
        bounding_box[0, 1] = y + (sin_angle * -half_length) + (cos_angle * half_width)

        # Top left corner
        bounding_box[1, 0] = x + (cos_angle * -half_length) - (sin_angle * -half_width)
        bounding_box[1, 1] = y + (sin_angle * -half_length) + (cos_angle * -half_width)

        # Top right
        bounding_box[2, 0] = x + (cos_angle * half_length) - (sin_angle * -half_width)
        bounding_box[2, 1] = y + (sin_angle * half_length) + (cos_angle * -half_width)

        # Bottom right
        bounding_box[3, 0] = x + (cos_angle * half_length) - (sin_angle * half_width)
        bounding_box[3, 1] = y + (sin_angle * half_length) + (cos_angle * half_width)

        return bounding_box
    
    def setGlobalPath(self):
        # Global path planning
        global_condition = self.set_path == False
        global_condition = global_condition and self.robot_states[self.robot_id, 0] != 0.0
        global_condition = global_condition and self.robot_states[self.robot_id, 1] != 0.0
        # global_condition = global_condition and self.robot_states[self.robot_id, 2] != 0.0
        if global_condition == True:
            self.global_path, _ = self.astar.planning(self.robot_states[self.robot_id, :], self.goal_pose)
            self.global_path_msg.header.frame_id = "map"
            self.global_path_msg.header.stamp = rospy.Time.now()
            for i in range(self.global_path.shape[0]):
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.header.seq = i
                pose.pose.position.x = self.global_path[i, 0]
                pose.pose.position.y = self.global_path[i, 1]
                self.global_path_msg.poses.append(pose)
            self.set_path = True
        
        

    def getRobotPoses(self, odom: Odometry):
        id = int(odom.header.frame_id.split("/")[1].split("_")[1])
        # Get current robot pose
        self.robot_states[id, 0] = odom.pose.pose.position.x
        self.robot_states[id, 1] = odom.pose.pose.position.y
        
        rqy = euler_from_quaternion([odom.pose.pose.orientation.x, 
                                    odom.pose.pose.orientation.y, 
                                    odom.pose.pose.orientation.z, 
                                    odom.pose.pose.orientation.w])
        self.robot_states[id, 2] = rqy[2]
    
    def getRobotVelocities(self, odom: Odometry):
        id = int(odom.header.frame_id.split("/")[1].split("_")[1])
        self.robot_states[id, 3] = odom.twist.twist.linear.x
        self.robot_states[id, 4] = odom.twist.twist.angular.z

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
            self.color = (0.5, 1.0, 0.0)
            
    def initializeSubscriberAndPublisher(self):
        self.position_subs = []
        self.velocity_subs = []
        for i in range(8):
            self.position_subs.append(rospy.Subscriber("/robot_{}/{}".format(i, self.params.position_topic), Odometry, self.getRobotPoses, queue_size=10))
            self.velocity_subs.append(rospy.Subscriber("/robot_{}/{}".format(i, self.params.velocity_topic), Odometry, self.getRobotVelocities, queue_size=10))
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
        
        robot_marker.pose.position.x = self.robot_states[self.robot_id, 0]
        robot_marker.pose.position.y = self.robot_states[self.robot_id, 1]
        robot_marker.pose.position.z = 0.2
        
        q = quaternion_from_euler(0, 0, self.robot_states[self.robot_id, 2])
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
        if self.robot_states[self.robot_id, 0] != 0.0 and self.robot_states[self.robot_id, 1] != 0.0:
            p.x = self.robot_states[self.robot_id, 0]
            p.y = self.robot_states[self.robot_id, 1]
            self.path_marker.points.append(p)
            visual_markers.markers.append(self.path_marker)
        
        self.markers_pub.publish(visual_markers)

