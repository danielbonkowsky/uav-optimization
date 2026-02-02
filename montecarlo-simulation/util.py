import scipy
import random
import numpy as np

def thermal_noise_watts(bandwidth):
    """ calculates the thermal noise in watts based on bandwidth """
    return 10**((7.5-174+10*np.log10(bandwidth))/10)/1000

def freq_to_wavelength(freq):
    """ returns a wavelength given frequency """
    return scipy.constants.c / freq

def normalDistribution(num_users, cx, cy, std_x, std_y, seed=324):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    users = []
    while (len(users) < num_users):
        x = rng.normal(loc=cx, scale=std_x)
        y = rng.normal(loc=cy, scale=std_y)

        if (x**2 + y**2 > 4000000):
            users.append((x, y))
    
    return np.asarray(users)