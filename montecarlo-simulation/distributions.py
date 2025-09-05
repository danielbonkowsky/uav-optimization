import numpy as np
import random
from math import ceil

def southernUtah(num_users, xbox, ybox, num_cities=4, seed=325):
    """
    Generate a bunch of spread out cities in the box
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    users = []

    weights = np.random.dirichlet(np.ones(num_cities)).tolist()
    std_devs = [random.uniform(500, 1000) for _ in range(num_cities)]

    means = []

    while len(means) < num_cities:
        x_mean = random.uniform(xbox[0], xbox[1])
        y_mean = random.uniform(ybox[0], ybox[1])

        if (abs(x_mean) > 2000 and abs(y_mean) > 2000):
            means.append((x_mean, y_mean))


    means = [(random.uniform(xbox[0], xbox[1]), random.uniform(ybox[0], ybox[1])) for _ in range(num_cities)]
    
    for (i, (x_mean, y_mean)) in enumerate(means):
        tmp_users = []

        while len(tmp_users) < weights[i]*num_users:
            x = rng.normal(loc=x_mean, scale=std_devs[i])
            y = rng.normal(loc=y_mean, scale=std_devs[i])

            if (x > xbox[0] and 
                x < xbox[1] and 
                y > ybox[0] and 
                y < ybox[1] and
                x**2 + y**2 > 4000000):
                tmp_users.append((x, y))
        
        users.extend(tmp_users)

    return users


def saltLakeCity(total_users, seed=324):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    users = []

    # main st. east
    density = 0.3
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(2000, 10000)
        y = rng.normal(0, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # main st. west
    density = 0.3
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(-10000, -2000)
        y = rng.normal(0, 200)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # state st.
    density = 0.3
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(-10000, 10000)
        y = rng.normal(-5000, 200)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # ibrahim st.
    density = 0.05
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(-5000, 100)
        y = rng.uniform(-10000, 10000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # robert st. north
    density = 0.025
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(0, 100)
        y = rng.uniform(2000, 10000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # robert st. south
    density = 0.025
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(0, 100)
        y = rng.uniform(-10000, -2000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    return users

def sanDiego(total_users, seed=324):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    users = []

    # highway cool
    density = 0.3
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(-9000, 100)
        y = rng.uniform(-10000, 10000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # daniel st
    density = 0.1
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(-10000, -4000)
        y = rng.normal(2000, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # ibrahim st
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(-10000, -7000)
        y = rng.normal(5000, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # robert st
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(-7000, 100)
        y = rng.uniform(5000, 10000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # jonas st
    density = 0.06
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(-7000, -2000)
        y = rng.normal(7500, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # isaac st
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(-2000, 100)
        y = rng.uniform(7500, 10000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # mom st
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(-3000, 100)
        y = rng.uniform(5000, 7500)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # jack st
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(-4000, 100)
        y = rng.uniform(2000, 5000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # liam st
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(-4000, 1000)
        y = rng.normal(5000, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # jon st
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(1000, 100)
        y = rng.uniform(5000, 10000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # sharayu st
    density = 0.1
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(5000, 100)
        y = rng.uniform(-2000, 10000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # florida st
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(7000, 10000)
        y = rng.normal(4000, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # josh st
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(7000, 100)
        y = rng.uniform(0, 10000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # marianne st
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(5000, 7000)
        y = rng.normal(0, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # anne st
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(5000, 10000)
        y = rng.normal(-2000, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # fred st
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(7500, 10000)
        y = rng.normal(-4000, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # terri st
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(7500, 100)
        y = rng.uniform(-6000, -2000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # shauna st
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(7500, 10000)
        y = rng.normal(-6000, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # ida st
    density = 0.3
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(9000, 100)
        y = rng.uniform(-10000, -6000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    return users

def twinCities(total_users, seed=324):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    users = []

    # 1st
    density = 0.05
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(8000, 100)
        y = rng.uniform(-10000, -8000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # 2nd
    density = 0.05
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(9000, 100)
        y = rng.uniform(-10000, -9000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # 3rd
    density = 0.3
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(8000, 10000)
        y = rng.normal(-9000, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # 4th
    density = 0.3
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(8000, 10000)
        y = rng.normal(-8000, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # 5th
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(7000, 8000)
        y = rng.normal(6000, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # 6th
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(8000, 100)
        y = rng.uniform(6000, 10000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # 7th
    density = 0.03
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(7000, 100)
        y = rng.uniform(6000, 9000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # 8th
    density = 0.1
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(4000, 10000)
        y = rng.normal(8000, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # 9th
    density = 0.1
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(6000, 10000)
        y = rng.normal(9000, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # 10th
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(6000, 100)
        y = rng.uniform(8000, 9000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # 11th
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(4000, 100)
        y = rng.uniform(8000, 10000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # 12th
    density = 0.02
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.uniform(3000, 4000)
        y = rng.normal(8500, 100)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    # 13th
    density = 0.01
    num_users = ceil(total_users * density)
    tmp_users = []

    while len(tmp_users) < num_users:
        x = rng.normal(3000, 100)
        y = rng.uniform(8500, 10000)

        if (x**2 + y**2 > 2000**2):
            tmp_users.append((x, y))
    
    users.extend(tmp_users)

    return users

def oneFarOneClose(num_users, seed=324):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    users = []

    weights = [0.3, 0.7]
    means = [(3000, 3000), (10000, 10000)]
    std_devs = [500, 500]

    for (i, (x_mean, y_mean)) in enumerate(means):
        tmp_users = []

        while len(tmp_users) < weights[i]*num_users:
            x = rng.normal(loc=x_mean, scale=std_devs[i])
            y = rng.normal(loc=y_mean, scale=std_devs[i])

            if (abs(x) < 10000 and abs(y) < 10000 and
                x**2 + y**2 > 4000000):
                tmp_users.append((x, y))
        
        users.extend(tmp_users)
    
    return users