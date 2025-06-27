"""
                                 (UAV flight path)
                                 _..----------..
                              .-~                -.
                              |.        x         .| (radius = R)
                               "-..___________..-"
                                        |
                                        |
                                        |  
                                        |   H
                                        |
                                        |
                                        |          
    BS ---------------------------------.-------------------- Dead zone
(0, 0, 0)                           (A, 0, 0)                 (B, 0, 0)
"""

import math
import scipy

UAV_HEIGHT = 100 # UAV circles 100m above the ground 
UAV_RADIUS = 30  # UAV flight path has a radius of 30m
A = 2000         # UAV circles 2km away from the base station
B = 2500         # The dead zone is centered 2.5km away from the base station

SIGNAL_FREQUENCY = 2*10**9 # 2GHz signal
SIGNAL_WAVELENGTH = scipy.constants.c / SIGNAL_FREQUENCY

# TODO: Tx powers for BS and UAV + noise power (in dB)

def bs_uav_distance(theta):
    """returns distance between the base station and UAV

    Arguments:
    theta -- the UAV's position along its flight path
    """

    return math.sqrt( (A + UAV_RADIUS * math.cos(theta)**2
                    + UAV_HEIGHT**2
                    + (UAV_RADIUS * math.sin(theta))**2) )

def uav_gu_distance(theta, x, z):
    """returns the distance between the UAV and a ground user

    Arguments:
    theta -- the UAV's position along its flight path
    x -- the x coordinate of the ground user
    z -- the z coordinate of the ground user
    """

    return math.sqrt( (A + UAV_RADIUS * math.cos(theta) - x)**2
                    + UAV_HEIGHT**2
                    + (UAV_RADIUS * math.sin(theta) - z)**2 )
                     



