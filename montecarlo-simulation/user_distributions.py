import numpy as np
import random

def oneFarOneClose(num_users, mean, std, seed=324):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    users = []

    weights = [0.5, 0.5]
    means = [(mean, -2*std), (mean, 2*std)]
    stds = [std/4, std/4]

    tmp_users = []
    while (len(tmp_users) < weights[0]*num_users):
        x1 = rng.normal(loc=means[0][0], scale=stds[0])
        y1 = rng.normal(loc=means[0][1], scale=stds[0])

        if ((x1 - mean)**2 + y1**2 < 4*std**2):
            tmp_users += [ (x1, y1) ]
    users += tmp_users

    tmp_users = []
    tmp_users = []
    while (len(tmp_users) < weights[1]*num_users):
        x2 = rng.normal(loc=means[1][0], scale=stds[1])
        y2 = rng.normal(loc=means[1][1], scale=stds[1])

        if ((x2 - mean)**2 + y2**2 < 4*std**2):
            tmp_users += [ (x2, y2) ]
    users += tmp_users
    
    return users

def skewedNormal(num_users, mean, std, seed=324):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    users = []

    x_mean = mean + 3000
    y_mean = 0
    while (len(users) < num_users):
        x = rng.normal(loc=x_mean, scale=std)
        y = rng.normal(loc=y_mean, scale=std)

        if ((x < mean and abs(y) < std) or (x - mean)**2 + y**2 < std**2):
            if (x**2 + y**2 > 4000000):
                users.append((x, y))
    
    return users