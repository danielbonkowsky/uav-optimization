"""
                                  UAV flight path
                                 _..----------..
                              .-~                -.
                              |.        x         .|
                               "-..___________..-"
                                        |  
                                        |    
                                        | H                         Dead zone
   /\ Base station                      |                        _..----------..
  /__\                                  |                     .-~                _.
 /____\ --------------------------------.-------------------- |.        x         .|
(0, 0, 0)                           (A, 0, 0)                  "-..___________..-" 
                                                                    (B, 0, 0)
"""

import math
import scipy
import random

RADIUS_UAV = 30       # Radius of UAV flight path
H = 100               # Height at which UAV circles
A = 2000              # Distance from BS to center of UAV flight path
B = 2500              # Distance from BS to center of dead zone

SIGNAL_FREQUENCY = 2*10**9
SIGNAL_WAVELENGTH = scipy.constants.c / SIGNAL_FREQUENCY

BANDWIDTH = 1*10**6 # UAV/BS/GU bandwidth (from energy-efficiency paper)

# Powers in dBW (see https://en.wikipedia.org/wiki/Decibel_watt)
# power in dBW = 10 * log10(power / 1W)
# power in W = 10^(power in dBW / 10)
POWER_NOISE = 7.5 - 174 + 10 * math.log10(BANDWIDTH) # Noise power in dBW
POWER_BASE_STATION = 47                              # BS power in dBW (from 3GPP paper)
POWER_UAV = 10                                       # UAV power in dBW (from energy-efficiency paper)

RADIUS_DEADZONE = 100         # Radius of dead zone
NUM_USERS = 50                # Number of users in the dead zone
USER_DISTRIBUTION = 'uniform' # Distribution of users in the dead zone

def bs_uav_distance(theta):
    """returns distance between the base station and UAV

    Arguments:
    theta -- the UAV's position along its flight path
    """

    return math.sqrt( (A + RADIUS_UAV * math.cos(theta)**2
                    + H**2
                    + (RADIUS_UAV * math.sin(theta))**2) )

def uav_gu_distance(theta, x, z):
    """returns the distance between the UAV and a ground user

    Arguments:
    theta -- the UAV's position along its flight path
    x -- the x coordinate of the ground user
    z -- the z coordinate of the ground user
    """

    return math.sqrt( (A + RADIUS_UAV * math.cos(theta) - x)**2
                    + H**2
                    + (RADIUS_UAV * math.sin(theta) - z)**2 )
                     
def uav_receive_power(theta):
    # TODO: Calculate P_rx of the BS -> UAV channel for a given theta
    pass

def user_receive_power(theta, x, y):
    # TODO: Calculate the P_rx of the UAV -> GU channel for a given theta, x, y
    pass

def generate_users():
    """returns a list of user coordinates generated based on how the users are 
    distributed in the dead zone
    """

    user_coordinates = []
    if (USER_DISTRIBUTION == 'uniform'):
        # Maybe there's a more efficient way to do it vectorized? NumPy?
        for i in range(NUM_USERS):
            x = B + random.uniform(-RADIUS_DEADZONE, RADIUS_DEADZONE)
            z = random.uniform(-RADIUS_DEADZONE, RADIUS_DEADZONE)
            user_coordinates.append((x, 0, z))

        return user_coordinates
        
