import cv2
import numpy as np
image = cv2.imread("/home/leanhchien/deep_rl_ws/src/dwa_multi_robot/slam/office.png", cv2.IMREAD_GRAYSCALE)

x = np.array([[0, 0]])

y = np.array([[0, 0]])

x = np.concatenate((x, y), axis= 0)
print(x)