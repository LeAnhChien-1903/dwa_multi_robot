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

show_animation = True


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

        return path_meter, path_pixel

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
        if self.cost_map[node.y, node.x] == 0 or self.cost_map[node.y, node.x] == 100:
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
    print(path_meter)
    path.reshape((-1, 1, 2))
    cv2.circle(cost_map, start_pixel, 10, 0, -1)
    cv2.circle(cost_map, goal_pixel, 10, 0, -1)
    cv2.polylines(cost_map, [path], 0, 0, 4)
    plt.imshow(cost_map, cmap='gray')
    plt.show()


# if __name__ == '__main__':
#     main()
