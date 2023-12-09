"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""
#!/usr/bin/env python
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dwa_multi_robot.cubic_spline import *
show_animation = True

def calculatedDistance(point1: np.ndarray, point2: np.ndarray):
    '''
        Calculates the Euclidean distance between two points
        ### Parameters
        - point1: the coordinate of first point
        - point2: the coordinate of second point
    '''
    return math.sqrt(np.sum(np.square(point1 - point2)))
class AStarPlanner:

    def __init__(self, map_path:str, origin_x: float, origin_y: float, resolution: float, robot_radius:float):
        """
            Initialize grid map for a star planning
            resolution: grid resolution [m]
            rr: robot radius[m]
        """
        self.map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        self.origin_x: float = origin_x
        self.origin_y: float = origin_y
        self.resolution: float = resolution
        self.robot_radius: float = robot_radius
        self.cost_map: np.ndarray = self.getCostmap()
        self.motion = self.get_motion_model()
    def getCostmap(self):
        radius = int(self.robot_radius/self.resolution)
        cost_map: np.ndarray = self.map.copy()
        for x in range(self.map.shape[1]):
            for y in range(self.map.shape[0]):
                if self.map[y, x] == 0:
                    cv2.circle(cost_map, (x, y), radius, 100, -1)
        for x in range(self.map.shape[1]):
            for y in range(self.map.shape[0]):
                if self.map[y, x] == 0:
                    cost_map[y, x] = 0
                    
        return cost_map
    def convertMeterToPixel(self, meter_point: np.ndarray):
        pixel_point = np.zeros_like(meter_point)
        pixel_point[0] = (meter_point[0] - self.origin_x)/self.resolution
        pixel_point[1] = self.cost_map.shape[0] - (meter_point[1] - self.origin_y)/self.resolution
        
        return pixel_point.astype(np.int16)
    def convertPixelToMeter(self, pixel_point: np.ndarray):
        meter_point = np.zeros_like(pixel_point).astype(np.float32)
        meter_point[0] = (pixel_point[0] * self.resolution) + self.origin_x
        meter_point[1] = float(((self.cost_map.shape[1] - pixel_point[1]) * self.resolution) + self.origin_y)
        
        return meter_point

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, start_point:np.ndarray, goal_point:np.ndarray):
        """
        A star path search

        input:
            start_point: current position of the robot [m, m]
            goal_point: goal position of the robot [m, m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_pixel = self.convertMeterToPixel(start_point)
        goal_pixel = self.convertMeterToPixel(goal_point)

        start_node = self.Node(x = start_pixel[0], y = start_pixel[1], cost = 0.0, parent_index = -1)
        goal_node = self.Node(x = goal_pixel[0], y = goal_pixel[1], cost = 0.0, parent_index= -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break
                
            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                current.y + self.motion[i][1],
                                current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node
                        
        path_meter, path_pixel = self.calc_final_path(goal_node, closed_set)
        smooth_path = self.smoothPath(path_meter)
        return smooth_path, path_pixel

    def calc_final_path(self, goal_node: Node, closed_set):
        # generate final course
        # path = self.convertPixelToMeter(np.array([goal_node.x, goal_node.y])).reshape(1, 2)
        path_meter = self.convertPixelToMeter(np.array([goal_node.x, goal_node.y])).reshape(1, 2)
        path_pixel = np.array([goal_node.x, goal_node.y]).reshape(1, 2)
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            # path = np.concatenate((path, self.convertPixelToMeter(np.array([n.x, n.y])).reshape(1, 2)), axis=0)
            path_pixel = np.concatenate((path_pixel, np.array([n.x, n.y]).reshape(1, 2)), axis=0)
            path_meter = np.concatenate((path_meter, self.convertPixelToMeter(np.array([n.x, n.y])).reshape(1, 2)), axis=0)
            
            parent_index = n.parent_index

        return np.flip(path_meter, axis = 0), np.flip(path_pixel.astype(np.int32), axis= 0)
    
    def checkCollision(self, start: np.ndarray, end: np.ndarray):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        dist = np.arange(0, distance, 0.1)
        # print(dist)
        for i in dist:
            point = np.array([start[0] + i * math.cos(angle), start[1] + i * math.sin(angle)])
            point_pixel = self.convertMeterToPixel(point)
            for extra_x in range(0, 2):
                for extra_y in range(0,2):
                    if self.cost_map[point_pixel[1] + extra_y, point_pixel[0] + extra_x] <= 100:
                        return True
        return False
    def smoothPath(self, path_meter: np.ndarray):
        smooth_path: np.ndarray = path_meter[0].reshape((1, 2))
        for i in range(1, path_meter.shape[0], 5):
            if i < path_meter.shape[0] - 1:
                condition =  self.checkCollision(smooth_path[-1], path_meter[i]) == False
                condition = condition and self.checkCollision(smooth_path[-1], path_meter[i+1]) == True
                condition = condition or calculatedDistance(smooth_path[-1], path_meter[i]) > 0.5
                if condition == True:
                    smooth_path = np.append(smooth_path, path_meter[i].reshape((1, 2)), axis= 0)
            else:
                if self.checkCollision(smooth_path[-1], path_meter[i]) == False:
                    smooth_path = np.append(smooth_path, path_meter[i].reshape((1, 2)), axis= 0)
        smooth_path = np.append(smooth_path, path_meter[-1].reshape((1, 2)), axis= 0)
        
        x = smooth_path[:, 0]
        y = smooth_path[:, 1]
        ds = 0.1  # [m] distance of each interpolated points
        
        sp = CubicSpline2D(x, y)
        s = np.arange(0, sp.s[-1], ds)
        path_result = np.zeros((s.shape[0], 2))
        for i in range(s.shape[0]):
            ix, iy = sp.calc_position(s[i])
            path_result[i, 0] = ix
            path_result[i, 1] = iy
            
        return path_result
    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_index(self, node:Node):
        return node.y * self.map.shape[0] + node.x

    def verify_node(self, node:Node):
        if node.x < 0:
            return False
        elif node.y < 0:
            return False
        elif node.x >= self.map.shape[1]:
            return False
        elif node.y >= self.map.shape[0]:
            return False

        # collision check
        if self.cost_map[node.y, node.x] <= 100:
            return False

        return True
    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [  [3, 0, 3],
                    [0, 3, 3],
                    [-3, 0, 3],
                    [0, -3, 3],
                    [-3, -3, math.sqrt(18)],
                    [-3, 3, math.sqrt(18)],
                    [3, -3, math.sqrt(18)],
                    [3, 3, math.sqrt(18)]]

        return motion


def main():
    a_star = AStarPlanner("/home/leanhchien/deep_rl_ws/src/dwa_multi_robot/slam/office.png", -10.0, -10.0, 0.025, 0.6)
    cost_map = a_star.cost_map.copy()
    start = np.array([-9.0, 4.8])
    goal = np.array([9.0,-4.8])
    start_pixel = a_star.convertMeterToPixel(start)
    goal_pixel = a_star.convertMeterToPixel(goal)
    path_meter, path = a_star.planning(start, goal)
    
    plt.subplots(1)
    plt.plot(path_meter[:, 0], path_meter[:, 1], "-g", label="Origin path")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()
    path.reshape((-1, 1, 2))
    cv2.circle(cost_map, start_pixel, 10, 0, -1)
    cv2.circle(cost_map, goal_pixel, 10, 0, -1)
    cv2.polylines(cost_map, [path], 0, 0, 4)
    # plt.imshow(cost_map, cmap='gray')
    plt.show()


# if __name__ == '__main__':
#     main()
