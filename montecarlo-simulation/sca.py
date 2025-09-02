import numpy as np
import cvxpy as cp

def sca_optimize_uav_circle(
    alpha,
    user_xy,                 # shape (K,2)
    H,
    N,                       # number of timeslots
    Un_list,                 # list of length N with arrays of user indices served in slot n, each of length M
    M,                       # number served per slot
    # link budget params
    P_tx_user, P_tx_uav, G_T, G_R, lambd, N0,
    # SCA controls
    r_init=None, cx_init=None, cy_init=None,
    r_bounds=None,           # tuple (r_min, r_max) or None
    cx_bounds=None,
    cy_bounds=None,
    max_sca_iters=30,
    sca_tol=1e-3,
    trust_radius=None,       # e.g. 200.0 meters, or None
    solver="ECOS",           # or "SCS"
    verbose=False
):
    """
    Maximizes average spectral efficiency over (r, cx, cy) using SCA.

    Returns:
        result: dict with keys:
            'r', 'cx', 'cy'             (optimized values)
            'obj_hist'                  (list of objective values per SCA iter)
            'last_t'                    ((N,) array of t_n at last subproblem)
            's_nk'                      (list of per-slot arrays of s_{n,k} at last subproblem)
            's_bn'                      ((N,) array of b^2 at last subproblem)
    """
    K = user_xy.shape[0]
    user_xy = np.asarray(user_xy, dtype=float)

    # Angles per slot
    thetas = 2*np.pi*np.arange(N)/N
    cos_th = np.cos(thetas)
    sin_th = np.sin(thetas)

    # Collect physical constants
    C_U = P_tx_user * G_T * G_R * (lambd/(4*np.pi))**2
    C_A = P_tx_uav  * G_T * G_R * (lambd/(4*np.pi))**2
    A_U = (C_U * M) / N0
    A_B = C_A / N0

    def f_and_grad(s, A):
        """Return f(s)=log2(1+A/s) and f'(s) = -A/(ln2*s*(s+A)) elementwise."""
        s = np.asarray(s, dtype=float)
        f = np.log2(1.0 + A/np.maximum(s, 1e-18))
        fp = -(A) / (np.log(2.0) * np.maximum(s,1e-18) * (np.maximum(s,1e-18) + A))
        return f, fp

    # Initialization (simple, robust): center midway between BS and user centroid; radius ~ distance to centroid
    centroid = user_xy.mean(axis=0)
    if cx_init is None or cy_init is None:
        cx = 0.5*centroid[0]
        cy = 0.5*centroid[1]
    else:
        cx, cy = float(cx_init), float(cy_init)

    if r_init is None:
        r = max(10.0, 0.5*np.linalg.norm(centroid - np.array([cx, cy])))
    else:
        r = float(r_init)

    obj_hist = []
    last_t = None
    last_snk = None
    last_sbn = None

    for it in range(max_sca_iters):
        # Compute current squared distances (to define linearizations)
        # UAV planar coords per slot
        ax = cx + r*cos_th   # (N,)
        ay = cy + r*sin_th   # (N,)

        # s_{n,k} current (only for k in U_n)
        s_nk0 = []
        for n in range(N):
            idxs = np.asarray(Un_list[n], dtype=int)
            dx = ax[n] - user_xy[idxs,0]
            dy = ay[n] - user_xy[idxs,1]
            s_nk0.append(dx*dx + dy*dy + H*H)   # shape (M,)
        # b^2 current
        s_b0 = (ax*ax + ay*ay + H*H)            # shape (N,)

        # Evaluate f and gradients at current s
        fU0 = []
        gU0 = []
        for n in range(N):
            f_u, g_u = f_and_grad(s_nk0[n], A_U)
            fU0.append(f_u)
            gU0.append(g_u)
        fB0, gB0 = f_and_grad(s_b0, A_B)

        # CVXPY variables
        r_var  = cp.Variable()
        cx_var = cp.Variable()
        cy_var = cp.Variable()
        t = cp.Variable(N)  # timeslot min-SE terms

        # Auxiliary upper-bounds for squared distances
        s_vars = [cp.Variable(len(Un_list[n])) for n in range(N)]  # s_{n,k}
        sb_var = cp.Variable(N)  # b^2_n

        cons = []

        # Geometric "upper-bound" constraints: s >= true squared distance
        for n in range(N):
            # planar expressions are affine in (r,cx,cy)
            ax_n = cx_var + r_var*cos_th[n]
            ay_n = cy_var + r_var*sin_th[n]

            # user distances
            idxs = np.asarray(Un_list[n], dtype=int)
            xk = user_xy[idxs,0]
            yk = user_xy[idxs,1]

            # ||(ax_n - xk, ay_n - yk)||^2 + H^2  <=  s_vars[n]  (as lower-bound on SE we want s_vars >= true)
            # Implement via rotated quadratic cone: s >= (ax_n - x)^2 + (ay_n - y)^2 + H^2
            # In cvxpy, simple convex way is: s >= (ax_n - x)^2 + (ay_n - y)^2 + H^2
            cons += [ s_vars[n] >= (ax_n - xk)**2 + (ay_n - yk)**2 + (H**2) ]

            # AB link
            cons += [ sb_var[n] >= (ax_n)**2 + (ay_n)**2 + (H**2) ]

        # Linearized (conservative) SE constraints per slot:
        # t_n <= (alpha/M) sum_k [ fU0 + gU0 * (s - s0) ]
        # t_n <= (1-alpha)      [ fB0 + gB0 * (sb - sb0) ]
        for n in range(N):
            # UA side
            f0 = fU0[n]         # shape (M,)
            g0 = gU0[n]         # shape (M,)
            s0 = s_nk0[n]       # shape (M,)
            cons += [
                t[n] <= (alpha/M) * ( cp.sum(f0) + g0 @ (s_vars[n] - s0) )
            ]

            # AB side
            cons += [
                t[n] <= (1.0 - alpha) * ( fB0[n] + gB0[n] * (sb_var[n] - s_b0[n]) )
            ]

        # Optional radius bounds
        if r_bounds is not None:
            r_min, r_max = r_bounds
            cons += [ r_var >= r_min, r_var <= r_max ]
        
        if cx_bounds is not None:
            cx_min, cx_max = cx_bounds
            cons += [ cx_var >= cx_min, cx_var <= cx_max ]
        
        if cy_bounds is not None:
            cy_min, cy_max = cy_bounds
            cons += [ cy_var >= cy_min, cy_var <= cy_max ]

        # Enforce positive radius (even if no bounds)
        cons += [ r_var >= 0.0 ]

        # Optional trust region to stabilize SCA
        if trust_radius is not None and it > 0:
            cons += [
                cp.abs(r_var  - r ) <= trust_radius,
                cp.abs(cx_var - cx) <= trust_radius,
                cp.abs(cy_var - cy) <= trust_radius,
            ]

        # Objective (maximize average SE lower bound)
        obj = cp.Maximize( cp.sum(t) / N )

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.MOSEK, verbose=verbose)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"SCA subproblem infeasible/failed at iter {it}: status {prob.status}")

        # Update iterate
        r_new  = float(r_var.value)
        cx_new = float(cx_var.value)
        cy_new = float(cy_var.value)

        obj_hist.append(float(np.sum(t.value)/N))

        # Convergence check (relative improvement in objective)
        if it > 0:
            rel_impr = (obj_hist[-1] - obj_hist[-2]) / max(1e-9, abs(obj_hist[-2]))
            if rel_impr < sca_tol:
                r, cx, cy = r_new, cx_new, cy_new
                last_t   = np.array(t.value).copy()
                last_snk = [np.array(si.value).copy() for si in s_vars]
                last_sbn = np.array(sb_var.value).copy()
                break

        r, cx, cy = r_new, cx_new, cy_new
        last_t   = np.array(t.value).copy()
        last_snk = [np.array(si.value).copy() for si in s_vars]
        last_sbn = np.array(sb_var.value).copy()

    return {
        "r": r, "cx": cx, "cy": cy,
        "obj_hist": obj_hist,
        "last_t": last_t,
        "s_nk": last_snk,
        "s_bn": last_sbn,
    }
