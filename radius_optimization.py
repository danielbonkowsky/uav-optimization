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

RADIUS_MIN_UAV = 30 # Minimum radius that the UAV can circle
RADIUS_MAX_UAV = 50 # Maximum radius that the UAV can circle
STEP_SIZE = 0.1     # Step size that the program will go through radii
H = 100             # Height at which UAV circles
A = 2000            # Distance from BS to center of UAV flight path
B = 2500            # Distance from BS to center of dead zone

SIGNAL_FREQUENCY = 2*10**9
SIGNAL_WAVELENGTH = scipy.constants.c / SIGNAL_FREQUENCY

BANDWIDTH = 1*10**6 # UAV/BS/GU bandwidth (from energy-efficiency paper)

# Powers in dBW (see https://en.wikipedia.org/wiki/Decibel_watt)
# power in dBW = 10 * log10(power / 1W)
# power in W = 10^(power in dBW / 10)
POWER_NOISE = 7.5 - 174 + 10 * math.log10(BANDWIDTH) # Noise power in dBW
POWER_GROUND_USER = 10                               # User power in dBW
POWER_UAV = 10                                       # UAV power in dBW (from energy-efficiency paper)

RADIUS_DEADZONE = 100         # Radius of dead zone
NUM_USERS = 50                # Number of users in the dead zone
USER_DISTRIBUTION = 'uniform' # Distribution of users in the dead zone

ANTENNA_GAIN = 1 # GT, GR in power equations

VELOCITY_UAV = 21 # see energy-efficiency paper

def gu_uav_distance(theta, radius, x, z):
    """returns the distance between a ground user and the UAV

    Arguments:
    theta -- the UAV's position along its flight path
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
    """

    return math.sqrt( (A + radius * math.cos(theta)**2
                    + H**2
                    + (radius * math.sin(theta))**2) )

def uav_receive_power(theta, radius, x, z):
    """returns the received power at the UAV from the user

    Arguments:
    theta -- the UAV's position along its flight path
    x -- the x coordinate of the ground user
    z -- the z coordinate of the ground user
    """

    return ( (POWER_GROUND_USER * ANTENNA_GAIN**2 * SIGNAL_WAVELENGTH**2)
           / (4 * math.pi * gu_uav_distance(theta, radius, x, z))**2 )

def bs_receive_power(theta, radius):
    """returns the received power at the BS from the UAV
    
    Arguments:
    theta -- the UAV's position along its flight path
    """

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

if (__name__ == '__main__'):

    uav_thetas = [(2 * user + 1) * math.pi / NUM_USERS for user in range(NUM_USERS)]
    user_positions = generate_users()

    optimal_radius = RADIUS_MIN_UAV
    optimal_bits_transmitted = 0

    for i in range(int( (RADIUS_MAX_UAV - RADIUS_MIN_UAV) / STEP_SIZE )):

        radius = RADIUS_MIN_UAV + i * STEP_SIZE

        print(f'radius: {radius}')

        period = (2 * math.pi * radius) / VELOCITY_UAV
        timeslot_length = period / NUM_USERS

        print(f'period: {period}')
        print(f'timeslot_length: {timeslot_length}')

        bits_transmitted = 0
        for j in range(NUM_USERS):

            uav_receive_power_W = 10 ** (uav_receive_power(uav_thetas[j], radius, user_positions[j][0], user_positions[j][1]) / 10)
            noise_power_W = 10 ** (POWER_NOISE / 10)

            uav_received_bits = ( 0.5 
                                 * timeslot_length 
                                 * BANDWIDTH 
                                 * math.log2(1 
                                             + uav_receive_power_W / noise_power_W ) )
            
            bs_receive_power_W = 10 ** (bs_receive_power(uav_thetas[j], radius) / 10)

            bs_received_bits = ( 0.5 
                                * timeslot_length 
                                * BANDWIDTH 
                                * math.log2(1 
                                            + bs_receive_power_W / noise_power_W ) )

            bits_transmitted += min(uav_received_bits, bs_received_bits)
        
        if (bits_transmitted > optimal_bits_transmitted):
            optimal_bits_transmitted = bits_transmitted
            optimal_radius = radius
        
        print()
    
    print(f'Optimal radius {optimal_radius} m')
    print(f'{optimal_bits_transmitted} bits transmitted')