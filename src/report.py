from dwa_multi_robot.ultis import *
from dwa_multi_robot.astar import *

import matplotlib.pyplot as plt
astar = AStarPlanner("/home/leanhchien/deep_rl_ws/src/dwa_multi_robot/slam/office.png",
                    -10, -10, 0.025, 0.6)

origin_map = astar.map.copy()
cost_map = astar.cost_map.copy()
blur_map =  cv2.GaussianBlur(cost_map, (5, 5), 2)
normal_map = 255 - blur_map

cv2.imwrite("/home/leanhchien/deep_rl_ws/src/dwa_multi_robot/report/origin_map.png", origin_map)
cv2.imwrite("/home/leanhchien/deep_rl_ws/src/dwa_multi_robot/report/cost_map.png", cost_map)
cv2.imwrite("/home/leanhchien/deep_rl_ws/src/dwa_multi_robot/report/blur_map.png", blur_map)
cv2.imwrite("/home/leanhchien/deep_rl_ws/src/dwa_multi_robot/report/normal_map.png", normal_map)