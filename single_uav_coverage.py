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

UAV_HEIGHT = 30   # UAV circles 30m above the ground
UAV_RADIUS = 30   # UAV circles with a radius of 30m
USER_RADIUS = 30  # Dead zone is a circle with radius 30
BS_DISTANCE = 100 # BS is 100m from the center of the UAV circle



def uav_distance(theta):
    """returns the distance from the BS to the UAV

    Arguments:
    theta -- the UAV's position along its flight path
    """

    return 0;

def achievable_downlink_fixed(theta, x, y):
    """returns the maximum achievable downlink (BS to user) rate in bits/s/hz 
    assuming a fixed timeshare rate 

    Arguments:
    theta -- the UAV's position along its flight path
    x -- the user's x coordinate
    y -- the user's y coordinate
    """

    return 0;
