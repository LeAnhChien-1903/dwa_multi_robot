import cv2
import numpy as np
# image = cv2.imread("/home/leanhchien/deep_rl_ws/src/dwa_multi_robot/slam/office.png", cv2.IMREAD_GRAYSCALE)

# cv2.imwrite("/home/leanhchien/deep_rl_ws/src/dwa_multi_robot/slam/office.png", cv2.resize(image, (800, 800)))

x = "/robot_0/odom"

print("/robot_{}/{}".format(2, "base_pose_ground_truth"))