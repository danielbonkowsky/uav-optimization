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
    
    return np.asarray(users)

def leftSkewedNormal(num_users, mean, std, seed=324):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    users = []
    while (len(users) < num_users):
        x = rng.normal(loc=mean+3000, scale=std)
        y = rng.normal(loc=0, scale=std)

        if ((x < mean and abs(y) < std) or (x - mean)**2 + y**2 < std**2):
            if (x**2 + y**2 > 4000000):
                users.append((x, y))
    
    return np.asarray(users)

def uniform(num_users, center, radius, seed=324):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    users = []

    for _ in range(num_users):
            r = radius * np.sqrt(random.random())
            theta = random.uniform(0, 2*np.pi)
            x = center + r * np.cos(theta)
            y = r * np.sin(theta)
    
            users.append((x, y))
    
    return np.asarray(users)

def leftUpSkewedNormal(num_users, mean1, std1, mean2, std2, seed=324):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    users = []

    # left skewed
    while (len(users) < 0.5*num_users):
        x = rng.normal(loc=mean1, scale=std1)
        y = rng.normal(loc=0, scale=std1)

        if ((x < mean1 and abs(y) < std1) or (x - mean1)**2 + y**2 < std1**2):
            if (x**2 + y**2 > 4000000):
                users.append((x, y))
    
    # up skewed
    while (len(users) < num_users):
        x = rng.normal(loc=mean2, scale=std2)
        y = rng.normal(loc=0, scale=std2)

        if ((y > 0 and abs(x - mean2) < std2) or (x - mean2)**2 + y**2 < std2**2):
            if (x**2 + y**2 > 4000000):
                users.append((x, y))
    
    return np.asarray(users)

def leftUpRectangles(num_users, rect_length, rect_width, cx, cy, seed=324):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    users = []

    while (len(users) < num_users):
        x = rng.uniform(0, rect_length)
        y = rng.uniform(0, rect_length)

        translated_x = x + cx - rect_length
        translated_y = y + cy

        if (translated_x**2 + translated_y**2 > 4000000):
            if x < (rect_length - rect_width) and y < rect_width:
                users.append((translated_x, translated_y))
            elif x > (rect_length - rect_width):
                users.append((translated_x, translated_y))
    
    return np.asarray(users)
        