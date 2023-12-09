import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from dwa_multi_robot.agent import Agent
image = cv2.imread("/home/leanhchien/deep_rl_ws/src/dwa_multi_robot/slam/office.png", cv2.IMREAD_GRAYSCALE)

cv2.imwrite("/home/leanhchien/deep_rl_ws/src/dwa_multi_robot/slam/office.png", cv2.resize(image, (800, 800)))
# def calculatedDistance(point1: np.ndarray, point2: np.ndarray):
#     '''
#         Calculates the Euclidean distance between two points
#         ### Parameters
#         - point1: the coordinate of first point
#         - point2: the coordinate of second point
#     '''
#     return math.sqrt(np.sum(np.square(point1 - point2)))

# def checkLineIntersection(line1_start: np.ndarray, line1_end: np.ndarray, line2_start: np.ndarray, line2_end: np.ndarray):
#     line1 = line1_end - line1_start
#     line2 = line2_end - line2_start
#     denom = line1[0] * line2[1] - line2[0] * line1[1]
    
#     if (denom == 0): 
#         return False, None
#     denomPositive = denom > 0
    
#     aux = line1_start - line2_start
    
#     s_numer = line1[0] * aux[1] - line1[1] * aux[0]
    
#     if (s_numer < 0) == denomPositive: 
#         return False, None
    
#     t_numer = line2[0] * aux[1] - line2[1] * aux[0]
    
#     if (t_numer < 0) == denomPositive: 
#         return False, None
    
#     if ((s_numer > denom) == denomPositive) or ((t_numer > denom) == denomPositive): 
#         return False, None
#     t = t_numer / denom
#     intersection = line1_start + t * line1
    
#     return True, intersection
# def robotBoundingBox(robot_state: np.ndarray, width: float, length: float):
#     bounding_box = np.zeros((4, 2))
#     x = robot_state[0]
#     y = robot_state[1]
#     half_length = width / 2
#     half_width = length / 2
#     sin_angle = math.sin(robot_state[2])
#     cos_angle = math.cos(robot_state[2])

#     # Bottom left
#     bounding_box[0, 0] = x + (cos_angle * -half_length) - (sin_angle * half_width)
#     bounding_box[0, 1] = y + (sin_angle * -half_length) + (cos_angle * half_width)

#     # Top left corner
#     bounding_box[1, 0] = x + (cos_angle * -half_length) - (sin_angle * -half_width)
#     bounding_box[1, 1] = y + (sin_angle * -half_length) + (cos_angle * -half_width)

#     # Top right
#     bounding_box[2, 0] = x + (cos_angle * half_length) - (sin_angle * -half_width)
#     bounding_box[2, 1] = y + (sin_angle * half_length) + (cos_angle * -half_width)

#     # Bottom right
#     bounding_box[3, 0] = x + (cos_angle * half_length) - (sin_angle * half_width)
#     bounding_box[3, 1] = y + (sin_angle * half_length) + (cos_angle * half_width)

#     return bounding_box

# def checkLinePolygonIntersection(line_start: np.array, line_end: np.array, polygon: np.array):
#     collision_points = []
#     for i in range(polygon.shape[0]):
#         if i == polygon.shape[0] - 1:
#             flag_, point = checkLineIntersection(line_start, line_end, polygon[i], polygon[0])
#             if flag_ == True:
#                 collision_points.append(point)
#         else:
#             flag_, point = checkLineIntersection(line_start, line_end, polygon[i], polygon[i+1])
#             if flag_ == True:
#                 collision_points.append(point)
    
#     if len(collision_points) > 0:
#         min_point = collision_points[0]
#         min_distance = calculatedDistance(line_start, min_point)
#         for i in range(1, len(collision_points)):
#             distance = calculatedDistance(line_start, collision_points[i])
#             if distance < min_distance:
#                 min_distance = distance
#                 min_point = collision_points[i]
#         return True, min_point
#     else:
#         return False, None

# def calculateTimeCollisionWithRobots(current: np.array, robot_trajectory: np.ndarray, other_trajectory: np.ndarray):
#     current_state = current.copy()
#     for i in range(robot_trajectory.shape[0]):
#         bounding_box = robotBoundingBox(other_trajectory[i, 0:3], 
#                                         0.6 + 0.6,
#                                         0.9 + 0.6)
        
#         flag, intersection = checkLinePolygonIntersection(current_state, robot_trajectory[i, 0:2], bounding_box)
#         if flag == True:
#             time_collision = 0.1 * i + 0.1 * calculatedDistance(intersection, current_state) / calculatedDistance(current_state, robot_trajectory[i, 0:2])
#             return time_collision
#         current_state = robot_trajectory[i, 0:2].copy()
    
#     return 0.3

# def calculateNewStateWithNewVelocity(current_state: np.ndarray, new_vel: np.ndarray, dt: float):
#         new_state = current_state.copy()
        
#         new_state[2] = new_state[2] + new_vel[1] * dt
#         new_state[0] = new_state[0] + new_vel[0] * math.cos(new_state[2]) * dt
#         new_state[1] = new_state[1] + new_vel[0] * math.sin(new_state[2]) * dt
#         new_state[3] = new_vel[0]
#         new_state[4] = new_vel[1]
        
#         return new_state
    
# def calculateNewState(current_state: np.ndarray, dt: float):
#     new_state = current_state.copy()
#     new_state[2] = new_state[2] + new_state[4] * dt
#     new_state[0] = new_state[0] + new_state[3] * math.cos(new_state[2]) * dt
#     new_state[1] = new_state[1] + new_state[3] * math.sin(new_state[2]) * dt
    
#     return new_state

# def calculateNewTrajectory(current_state: np.ndarray, new_vel: np.ndarray):
#     new_trajectory = np.zeros((3, current_state.shape[0]))
#     new_trajectory[0] = calculateNewStateWithNewVelocity(current_state, new_vel, 0.1)
#     for i in range(1, 3):
#         new_trajectory[i] = calculateNewState(new_trajectory[i-1], 0.1)
        
#     return new_trajectory

# def calculateNonHolonomicAcceleration():
#     acceleration = np.zeros((7 ** 2, 2))
#     counter = 0
#     for i in range(7):
#         for j in range(7):
#             acceleration[counter, 0] = - 5.0 + i * (2*5.0 / (7-1))
#             acceleration[counter, 1] = - 5.0 + j * (2*5.0 / (7-1))
#             counter += 1
            
#     return acceleration
# def calculateNewVelocity(current_vel: np.ndarray, acceleration: np.ndarray, t: float):
#     new_vel = current_vel + acceleration * t * 0.5
#     for i in range(new_vel.shape[0]):
#         if new_vel[i, 0] > 1.0:
#             new_vel[i, 0] = 1.0
#         if new_vel[i, 0] < -1.0:
#             new_vel[i, 0] = -1.0
#         if new_vel[i, 1] > 1.5:
#             new_vel[i, 1] = 1.5
#         if new_vel[i, 1] < -1.5:
#             new_vel[i, 1] = -1.5
    
#     return np.round(new_vel, 2)

# def calculateAllNewTrajectories(current_state: np.ndarray, new_vel: np.ndarray):
#     new_trajectories = np.zeros((new_vel.shape[0], 3, current_state.shape[0]))
#     for i in range(new_vel.shape[0]):
#         new_trajectories[i] = calculateNewTrajectory(current_state, new_vel[i])
#     return new_trajectories

# acceleration = calculateNonHolonomicAcceleration()
# new_vel = calculateNewVelocity(np.zeros(2), acceleration, 0.1)

# current_state = np.zeros(5)
# current_state[2] = math.pi/6
# new_trajectories = calculateAllNewTrajectories(current_state, new_vel)
# for i in range(new_trajectories.shape[0]):
#     x = [new_trajectories[i, 0, 0], new_trajectories[i, 1, 0], new_trajectories[i, 2, 0]]
#     y = [new_trajectories[i, 0, 1], new_trajectories[i, 1, 1], new_trajectories[i, 2, 1]]
#     plt.plot(x, y, 'r-')
# plt.show()
