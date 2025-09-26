import random
import scipy
import numpy as np
import cvxpy as cp
import gc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize
from joblib import Parallel, delayed

plt.rcParams["font.size"] = 12

# Seed random number generators
seed = 324
random.seed(seed)
np.random.seed(seed)
rng = np.random.default_rng(seed)

# Parameters used for optimization
r_min = 500 # meters
r_max = 15000 # meters
D = 15000 # meters
rD = 2000 # meters
cx_min = -20000 # meters
cx_max = 40000 # meters
cy_min = -20000
cy_max = 20000
v = 50 # meters/second
N = 200 # timeslots in one rotation
M = 5 # users
K = 20 # total number of users
H = 1000 # meters
PAtx = 10 # watts
PUtx = 0.01 # watts
GT = 1
GR = 1
B = 1e6 # Hz
N0 = 10**((7.5-174+10*np.log10(B))/10)/1000 # watts
F = 2e9 # Hz
WAVELENGTH = scipy.constants.c / F # meters

# More complex constants for convenience
thetas = 2*np.pi*np.arange(N)/N
cos_th = np.cos(thetas)
sin_th = np.sin(thetas)
CU = PUtx*GT*GR*(WAVELENGTH/(4*np.pi))**2
CA = PAtx*GT*GR*(WAVELENGTH/(4*np.pi))**2
AU = (CU * M) / N0
AB = CA / N0

def normalDistribution(num_users, cx, cy, std):
    """
    Generate users in a normal distribution with centerpoint (cx, cy) and
    standard deviation std
    """

    users = []
    while (len(users) < num_users):
        x = rng.normal(loc=cx, scale=std)
        y = rng.normal(loc=cy, scale=std)

        if (x**2 + y**2 > 4000000):
            users.append((x, y))
    
    return np.asarray(users)

users = normalDistribution(K, D, 0, 2000)

def meanSE(alpha, 
           r, 
           cx, 
           cy, 
           a,
           users=users,
           AU=AU,
           AB=AB,
           H=H,
           M=M):

    # Compute UAV position at each timeslot
    ax = cx + r*cos_th
    ay = cy + r*sin_th

    # Compute squared distances
    dx = ax[:, None] - users[:, 0]
    dy = ay[:, None] - users[:, 1]
    snk0 = dx*dx + dy*dy + H*H

    sbn0 = (ax*ax + ay*ay + H*H)

    # Calculate SEUA
    log_terms = np.log2(1.0 + AU/snk0)
    seua = (alpha/M) * np.sum(a * log_terms, axis=1)

    # Calculate SEAB
    seab = (1-alpha) * np.log2(1.0 + AB/sbn0)

    # Calculate the mins
    min_se = np.minimum(seua, seab)

    # Return the average
    return np.mean(min_se)

def optimize_alpha(r, 
                   cx, 
                   cy, 
                   a, 
                   users=users, 
                   AU=AU,
                   AB=AB,
                   H=H,
                   M=M,
                   verbose=False):
    
    # Compute UAV position at each timeslot
    ax = cx + r*cos_th
    ay = cy + r*sin_th

    # Compute squared distances
    dx = ax[:, None] - users[:, 0]
    dy = ay[:, None] - users[:, 1]
    snk0 = dx*dx + dy*dy + H*H
    sbn0 = (ax*ax + ay*ay + H*H)

    # Vectorized computation of an and bn
    # an[n] = sum over users of a[n] * log2(1 + AU/snk0[n])
    log_terms = np.log2(1.0 + AU/snk0)  # Shape: (N, num_users)
    an = np.sum(a * log_terms, axis=1)   # Shape: (N,) - sum over users for each timeslot
    bn = np.log2(1.0 + AB/sbn0)          # Shape: (N,)

    alpha_var = cp.Variable()
    tn_var = cp.Variable(N)

    # Vectorized constraints
    cons = []
    cons += [tn_var <= (alpha_var/M) * an]      # Vectorized constraint
    cons += [tn_var <= (1 - alpha_var) * bn]    # Vectorized constraint
    cons += [alpha_var >= 0, alpha_var <= 1]

    obj = cp.Maximize( cp.mean(tn_var) )

    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.MOSEK, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SCA subproblem infeasible/failed: status {prob.status}")
    
    return float(alpha_var.value)

def optimize_a(alpha, 
               r, 
               cx, 
               cy, 
               users=users,
               AU=AU,
               AB=AB,
               H=H,
               M=M,
               verbose=False):
    
    # Compute UAV position at each timeslot
    ax = cx + r*cos_th
    ay = cy + r*sin_th

    # Compute squared distances
    dx = ax[:, None] - users[:, 0]
    dy = ay[:, None] - users[:, 1]
    snk0 = dx*dx + dy*dy + H*H
    sbn0 = (ax*ax + ay*ay + H*H)

    # Calculate seua and seab
    seua = (alpha/M) * np.log2(1.0 + AU/snk0)  # Shape: (N, num_users)
    seab = (1-alpha) * np.log2(1.0 + AB/sbn0)   # Shape: (N,)

    # define CVXPY variables
    a_vars = cp.Variable((N, K))
    t_vars = cp.Variable(N)

    # Constraints
    cons = []

    # Add tn constraints - vectorized
    cons += [t_vars[n] <= cp.sum(cp.multiply(a_vars[n], seua[n])) for n in range(N)]
    cons += [t_vars <= seab]  # Vectorized constraint
    
    # Box constraints on a_vars - vectorized
    cons += [a_vars >= 0, a_vars <= 1]
    
     # Column sum constraint: each user at most once - vectorized
    cons += [cp.sum(a_vars, axis=0) <= (N*M)/K]
    
    # Row sum constraint: at most M users per timeslot - vectorized
    cons += [cp.sum(a_vars, axis=1) <= M]

    # Want to maximize tn
    obj = cp.Maximize( cp.mean(t_vars) )

    # Solve problem and check convergence
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.MOSEK, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SCA subproblem infeasible/failed: status {prob.status}")
    
    return a_vars.value

def objective(params, alpha, a, users, AU, AB, H, M):
    r, cx, cy = params
    return -meanSE(alpha, 
                   r, 
                   cx, 
                   cy, 
                   a, 
                   users=users,
                   AU=AU,
                   AB=AB,
                   H=H,
                   M=M)

def powells_optimizer(
                    alpha0,
                    r0, 
                    cx0, 
                    cy0, 
                    rbounds=(r_min, r_max),
                    cxbounds=(cx_min, cx_max),
                    cybounds=(cy_min, cy_max),
                    users=users,
                    AU=AU,
                    AB=AB,
                    H=H,
                    M=M,
                    tolerance=1e-3,
                    verbose=True
):

    alpha = alpha0
    r = r0
    cx = cx0
    cy = cy0
    bounds = [rbounds, cxbounds, cybounds]

    traj_history = []
    obj_history = []

    it = 0
    while True:
        if verbose:
            print(f'Powell\'s Optimizer Iteration {it}')
            print(f'    Values at iteration {it}: alpha = {alpha}, r={r}, c=({cx}, {cy})')

        a = optimize_a(alpha, 
                       r, 
                       cx, 
                       cy, 
                       users=users,
                       AU=AU,
                       AB=AB,
                       H=H,
                       M=M)
        if verbose:
            print('    User scheduling optimized')

        alpha = optimize_alpha(r,
                               cx,
                               cy,
                               a,
                               users=users,
                               AU=AU,
                               AB=AB,
                               H=H,
                               M=M)
        if verbose:
            print('    Timeshare optimized')
        
        result = minimize(
                    objective,
                    [r,cx,cy],
                    args=(alpha, a, users, AU, AB, H, M),
                    method='Powell',
                    bounds=bounds,
                    options={
                        'maxiter':1000,
                        'xtol':1e-3,
                        'ftol':1e-3
                    }
        )
        if verbose:
            print(f'    Trajectory optimized, result.message: {result.message}')
            print()

        r, cx, cy = result.x

        obj_history.append(meanSE(alpha, 
                                  r, 
                                  cx, 
                                  cy, 
                                  a, 
                                  users=users,
                                  AU=AU,
                                  AB=AB,
                                  H=H,
                                  M=M))
        traj_history.append((r, cx, cy))

        if it > 0:
            if ( (obj_history[-1] - obj_history[-2])/obj_history[-2] < tolerance):
                break
        it += 1
    
    return alpha, a, traj_history

def random_schedule(M=M, maxiters=1000, tol=1e-9):
    mat = np.random.rand(N, K)

    for _ in range(maxiters):
        # Scale rows
        row_sums = mat.sum(axis=1, keepdims=True)
        mat *= (M / row_sums)

        # Scale columns
        col_sums = mat.sum(axis=0, keepdims=True)
        mat *= ( (N*M/K) / col_sums)

        # Check convergence
        if (np.allclose(mat.sum(axis=1), M, atol=tol) and
            np.allclose(mat.sum(axis=0), N, atol=tol)):
            break

    return mat

# Starting conditions
alpha0 = 0.5
r0 = r_min
cx0 = D/2
cy0 = 0

def run_once(PAtx, users):
    CA = PAtx*GT*GR*(WAVELENGTH/(4*np.pi))**2
    AB = CA / N0
    alpha_opt, a_opt, traj_hist = powells_optimizer(alpha0, r0, cx0, cy0, users=users, AB=AB, verbose=False)
    r_opt, cx_opt, cy_opt = traj_hist[-1]
    return r_opt

def patx_plot(iterations=1000):
    PAtx_arr = np.arange(0.1, 10, 0.1)

    sm_results = []
    md_results = []
    lg_results = []

    for it in range(iterations):
        print(f'{np.round(it/iterations * 100, decimals=2)}% complete')
        
         # Generate users once per iteration
        sm_users = normalDistribution(K, D, 0, 1000)
        md_users = normalDistribution(K, D, 0, 2000)
        lg_users = normalDistribution(K, D, 0, 3000)

        # Calculate r values
        sm_batch = Parallel(n_jobs=-1, backend='threading')(
            delayed(run_once)(PAtx, sm_users) for PAtx in PAtx_arr
        )

        md_batch = Parallel(n_jobs=-1, backend='threading')(
            delayed(run_once)(PAtx, md_users) for PAtx in PAtx_arr
        )

        lg_batch = Parallel(n_jobs=-1, backend='threading')(
            delayed(run_once)(PAtx, lg_users) for PAtx in PAtx_arr
        )

        sm_results.append(sm_batch)
        md_results.append(md_batch)
        lg_results.append(lg_batch)

        # Garbage collection
        gc.collect()

    sm_results = np.array(sm_results)
    md_results = np.array(md_results)
    lg_results = np.array(lg_results)

    # Calculate means and plot
    sm_radii = sm_results.mean(axis=0)
    md_radii = md_results.mean(axis=0)
    lg_radii = lg_results.mean(axis=0)
    
    plt.plot(PAtx_arr, sm_radii, c='r', label='Case A')
    plt.fill_between(PAtx_arr, np.add(sm_radii, sm_results.std(axis=0)), np.subtract(sm_radii, sm_results.std(axis=0)), color='r', alpha=0.3)

    plt.plot(PAtx_arr, md_radii, c='g', label='Case B')
    plt.fill_between(PAtx_arr, np.add(md_radii, md_results.std(axis=0)), np.subtract(md_radii, md_results.std(axis=0)), color='g', alpha=0.3)

    plt.plot(PAtx_arr, lg_radii, c='b', label='Case C')
    plt.fill_between(PAtx_arr, np.add(lg_radii, lg_results.std(axis=0)), np.subtract(lg_radii, lg_results.std(axis=0)), color='b', alpha=0.3)

    plt.grid(True)
    plt.xlabel("UAV transmission power (W)")
    plt.ylabel("Optimal radius (m)")
    plt.legend()
    plt.savefig("patx-vs-r-m5.svg")

    np.savez('patx-vs-r-m5-data.npz',
             PAtx_arr=PAtx_arr,
             sm_radii=sm_radii,
             md_radii=md_radii,
             lg_radii=lg_radii)

patx_plot()