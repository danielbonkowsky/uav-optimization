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
import time
import random
import matplotlib.pyplot as plt
import pandas as pd

RADIUS_MIN_UAV = 30  # Minimum radius that the UAV can circle in meters
RADIUS_MAX_UAV = 60   # Maximum radius that the UAV can circle in meters
STEP_SIZE = 0.1       # Step size that the program will go through radii
H = 100               # Height at which UAV circles in meters
A = 2000              # Distance from BS to center of UAV flight path in meters
B = 2500              # Distance from BS to center of dead zone in meters

SIGNAL_FREQUENCY = 2*10**9
SIGNAL_WAVELENGTH = scipy.constants.c / SIGNAL_FREQUENCY

BANDWIDTH = 1*10**6 # 1 MHz UAV/BS/GU bandwidth (from energy-efficiency paper)


# power in dBW = 10 * log10(power / 1W)
# power in W = 10^(power in dBW / 10)
POWER_NOISE = 10 ** ( 
    (7.5 - 174 + 10 * math.log10(BANDWIDTH)) / 10 ) # Noise power in watts
POWER_GROUND_USER = 1                               # User power in watts
POWER_UAV = 1                                       # UAV power in watts

RADIUS_DEADZONE = 100         # Radius of dead zone in meters
NUM_USERS = 1                # Number of users in the dead zone
USER_DISTRIBUTION = 'uniform' # Distribution of users in the dead zone

ANTENNA_GAIN = 1 # GT, GR in power equations

VELOCITY_UAV = 21 # m/s, see energy-efficiency paper

def gu_uav_distance(theta, radius, x, z):
    """returns the distance between a ground user and the UAV

    Arguments:
    theta -- the UAV's position along its flight path
    radius -- the radius of the UAV's flight path in meters
    x -- the x coordinate of the ground user
    z -- the z coordinate of the ground user
    """

    return math.sqrt( (A + radius * math.cos(theta) - x)**2
                    + H**2
                    + (radius * math.sin(theta) - z)**2 )

def uav_bs_distance(theta, radius):
    """returns distance between the UAV and base station

    Arguments:
    theta -- the UAV's position along its flight path
    radius -- the radius of the UAV's flight path in meters
    """

    return math.sqrt( (A + radius * math.cos(theta)**2
                    + H**2
                    + (radius * math.sin(theta))**2) )

def uav_receive_power(radius, time, x, z):
    """returns the received power at the UAV from the user in watts

    Arguments:
    radius -- the radius of the UAV's flight path in meters
    time -- the elapsed time since the beginning of simulation in seconds
    x -- the x coordinate of the ground user in meters
    z -- the z coordinate of the ground user in meters
    """
    theta = get_uav_theta(radius, time)

    return ( (POWER_GROUND_USER * ANTENNA_GAIN**2 * SIGNAL_WAVELENGTH**2)
           / (4 * math.pi * gu_uav_distance(theta, radius, x, z))**2 )

def bs_receive_power(radius, time):
    """returns the received power at the BS from the UAV in watts
    
    Arguments:
    radius -- the radius of the UAV's flight path in meters
    time -- the elapsed time since the beginning of simulation in seconds
    """
    theta = get_uav_theta(radius, time)

    return ( (POWER_UAV * ANTENNA_GAIN**2 * SIGNAL_WAVELENGTH**2)
           / (4 * math.pi * uav_bs_distance(theta, radius))**2 )

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
            user_coordinates.append((x, z))

        return user_coordinates

def get_uav_theta(radius, time):
    """returns the UAV's position along its flight path given a time
    
    Arguments:
        radius -- the radius of the UAV's flight path in meters
        time -- the elapsed time since the beginning of simulation in seconds
    """
    period = (2 * math.pi * radius) / VELOCITY_UAV

    return (2 * math.pi / period) * time

if (__name__ == '__main__'):
    start_time = time.time()

    longest_period = (2 * math.pi * RADIUS_MAX_UAV) / VELOCITY_UAV
    
    data = pd.DataFrame()
    optimal_radii = []

    for it in range(1000):

        user = generate_users()[0]
        radius = RADIUS_MIN_UAV

        max_bits = -1
        optimal_radius = -1

        radii = []
        bits = []

        while (radius <= RADIUS_MAX_UAV):
            uav_received_bits = 0
            bs_received_bits = 0

            for i in range(int(longest_period * 1000)):

                if (i % 2 == 0):
                    bps = BANDWIDTH * math.log2(1 + uav_receive_power(radius,
                                                                    i / 1000.0,
                                                                    user[0],
                                                                    user[1]) 
                                                    / POWER_NOISE)
                    uav_received_bits += bps / 1000
                else:
                    bps = BANDWIDTH * math.log2(1 + bs_receive_power(radius,
                                                                    i / 1000.0)
                                                    / POWER_NOISE)
                    bs_received_bits += bps / 1000
            
            if (min(bs_received_bits, uav_received_bits) > max_bits):
                max_bits = min(bs_received_bits, uav_received_bits)
                optimal_radius = radius

            radii.append(radius)
            bits.append(min(uav_received_bits, bs_received_bits))
            
            radius += STEP_SIZE

        optimal_radii.append(optimal_radius)
        data[f'radii{it}'] = radii
        data[f'bits{it}'] = bits
    
    for i in range(1000):
        plt.plot(data[f'radii{i}'], data[f'bits{i}'])
        plt.axvline(x = optimal_radii[i])
    
    plt.xlabel('Radius (m)')
    plt.ylabel('Bits transmitted')

    print(f'Runtime: {time.time() - start_time} seconds')

    plt.show()

