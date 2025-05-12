"""
Goal is to generate a plot showing the relationship between the angle of a UAV
along its circular path and the achievable rate for a user in the dead zone 
(average/integral of all possible points in the dead zone?)

Parameters
theta : position of UAV along the circle
h : height of UAV above the ground
r_uav : radius of uav flight circle
r_user : radius of dead zone / places user could be (same as r_uav?)
d : distance from BS to center of UAV flight circle
(x, y) : position of user (vary over all in dead zone?)

Output
rate : achievable rate in bits/s/hz
"""
import math
import numpy as np
import matplotlib.pyplot as plt

UAV_HEIGHT = 30   # UAV circles 30m above the ground
UAV_RADIUS = 30   # UAV circles with a radius of 30m
USER_RADIUS = 30  # Dead zone is a circle with radius 30m
BS_DISTANCE = 100 # BS is 100m from the center of the UAV circle

BS_TRANSMIT_POWER = 100
UAV_TRANSMIT_POWER = 100
NOISE_POWER = 0.01

def uav_distance(theta, x, y):
    """returns the distance from a point to the UAV

    Arguments:
    theta -- the UAV's position along its flight path
    x -- x-coord of the point
    y -- y-coord of the point
    """
    
    uav_x = UAV_RADIUS * math.cos(theta)
    uav_y = UAV_RADIUS * math.sin(theta)

    plane_distance = math.sqrt((uav_x - x)**2 + (uav_y - y)**2)
    
    return math.sqrt(UAV_HEIGHT**2 + plane_distance**2)

def uav_downlink_rate(theta):
    """returns the downlink rate achievable by the UAV

    Arguments
    theta -- the UAV's position along its flight path
    """

    d = uav_distance(theta, -BS_DISTANCE, 0)
    power = BS_TRANSMIT_POWER / (d**2)
    
    return math.log2(1 + power/NOISE_POWER) 

def user_downlink_rate(theta, x, y):
    """returns the downlink rate achievable by the user

    Arguments
    theta -- the UAV's position along its flight path
    x -- user x-coord
    y -- user y-coord
    """

    d = uav_distance(theta, x, y)
    power = UAV_TRANSMIT_POWER / (d**2)

    return math.log2(1 + power/NOISE_POWER)

theta = np.linspace(0, 2*math.pi, 100)
uav_downlink = np.array([uav_downlink_rate(t) for t in theta])
user_downlink = np.array([user_downlink_rate(t, UAV_RADIUS, 0) for t in theta])

fig, ax = plt.subplots()

ax.plot(theta, uav_downlink, color='red', label='uav downlink')
ax.plot(theta, user_downlink, color='blue', label='user downlink')

ax.set_xticks(np.arange(0, 9*math.pi/4, math.pi/4))
ax.set_xlabel('theta (rad)')
ax.set_ylabel('achievable rate')
ax.legend(loc='upper left')

plt.show()
