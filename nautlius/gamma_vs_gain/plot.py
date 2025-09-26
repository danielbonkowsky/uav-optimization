import random
import scipy
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize
from joblib import Parallel, delayed

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
M = 2 # users
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
    for i in range(num_users):
        x = rng.normal(loc=cx, scale=std)
        y = rng.normal(loc=cy, scale=std)
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

def objective(params, alpha, a, users, AU, AB, H):
    r, cx, cy = params
    return -meanSE(alpha, 
                   r, 
                   cx, 
                   cy, 
                   a, 
                   users=users,
                   AU=AU,
                   AB=AB,
                   H=H)

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
a0 = random_schedule()

def run_once(D, users):
    shifted_users = []

    for user in users:
        xk, yk = user
        shifted_users.append((xk + D, yk))
    
    shifted_users = np.asarray(shifted_users)

    alpha_opt, a_opt, traj_hist = powells_optimizer(
                            alpha0,
                            r0,
                            D,
                            cy0,
                            users=shifted_users,
                            verbose=False)
    r_opt, cx_opt, cy_opt = traj_hist[-1]
    opt_se = meanSE(alpha_opt, r_opt, cx_opt, cy_opt, a_opt, users=users)
    centered_se = meanSE(alpha0, r0, D, cy0, a0, users=users)
    return opt_se - centered_se

def gamma_vs_gain_plot(iterations=100):
    # gamma_arr = np.arange(0.1, 3, 0.01)
    dist_arr = np.arange(500, 15000, 50)

    sm_results = []
    md_results = []
    lg_results = []
    for it in range(iterations):
        print(f'{np.round(it/iterations * 100, decimals=2)}% complete')

        sm_users = normalDistribution(K, 0, 0, 1000)
        md_users = normalDistribution(K, 0, 0, 2000)
        lg_users = normalDistribution(K, 0, 0, 3000)

        # gamma = H/D
        # D = H/gamma

        # Run all D values in parallel
        results = Parallel(n_jobs=-1)(
            # delayed(run_once)(H/gamma, sm_users) for gamma in gamma_arr
            delayed(run_once)(D, sm_users) for D in dist_arr
        )
        sm_results.append(results)

        results = Parallel(n_jobs=-1)(
            # delayed(run_once)(H/gamma, md_users) for gamma in gamma_arr
            delayed(run_once)(D, md_users) for D in dist_arr
        )
        md_results.append(results)

        results = Parallel(n_jobs=-1)(
            # delayed(run_once)(H/gamma, lg_users) for gamma in gamma_arr
            delayed(run_once)(D, lg_users) for D in dist_arr
        )
        lg_results.append(results) 
    
    # Convert to array shape (iterations, len(D_arr))
    sm_results = np.array(sm_results)
    md_results = np.array(md_results)
    lg_results = np.array(lg_results)

    # Get average over iterations
    sm_means = sm_results.mean(axis=0)
    md_means = md_results.mean(axis=0)
    lg_means = lg_results.mean(axis=0)

    # plt.plot(gamma_arr, sm_means, c='r', label='variance = 1000m')
    # plt.plot(gamma_arr, md_means, c='g', label='variance = 2000m')
    # plt.plot(gamma_arr, lg_means, c='b', label='variance = 3000m')

    plt.plot(dist_arr, sm_means, c='r', label='Case A')
    plt.plot(dist_arr, md_means, c='g', label='Case B')
    plt.plot(dist_arr, lg_means, c='b', label='Case C')

    plt.grid(True)
    plt.xlabel("$\\mu_\\text{x}$ (m)")
    plt.ylabel("Spectral efficiency gain (bps/Hz)")
    plt.legend()
    plt.savefig("dist-vs-gain.svg")

    # np.savez('gamma-vs-gain-data.npz',
    #          gamma_arr=gamma_arr,
    #          sm_mean=sm_means,
    #          md_means=md_means,
    #          lg_means=lg_means)

    np.savez('dist-vs-gain-data.npz',
             dist_arr=dist_arr,
             sm_mean=sm_means,
             md_means=md_means,
             lg_means=lg_means)

gamma_vs_gain_plot()