#!/usr/bin/env python

import rospy
from dwa_multi_robot.agent import Agent

rospy.init_node("dwa_multi_robot", anonymous=False)

controller = Agent(rospy.get_name())
rospy.spin()

