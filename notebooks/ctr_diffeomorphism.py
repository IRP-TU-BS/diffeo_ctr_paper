import random

import numpy as np
from scipy.optimize import least_squares


def kernel(x, y, p, mu=0.9):
    """
    Compute the Gaussian kernel between points x and y.

    Parameters:
    x (np.ndarray): First set of points.
    y (np.ndarray): Second set of points.
    p (float): Kernel parameter.
    mu (float): Scaling factor for the kernel parameter.

    Returns:
    np.ndarray: Kernel values.
    """
    diff = x - y
    norm_sq = np.sum(diff**2, axis=1)
    return np.exp(-((mu * p) ** 2) * norm_sq)


def apply_gauss_kernel(p, x, c, v):
    """
    Apply Gaussian transformation to points x.

    Parameters:
    p (float): Kernel parameter.
    x (np.ndarray): Points to transform.
    c (np.ndarray): Center points for the kernel.
    v (np.ndarray): Transformation vectors.

    Returns:
    np.ndarray: Transformed points.
    """
    return x + kernel(x, c, p)[:, np.newaxis] * v


def inverse_apply_gauss_kernel(p, y, c, v):
    """
    Apply inverse Gaussian transformation to points y.

    Parameters:
    p (float): Kernel parameter.
    y (np.ndarray): Points to transform.
    c (np.ndarray): Center points for the kernel.
    v (np.ndarray): Transformation vectors.

    Returns:
    np.ndarray: Transformed points.
    """
    return y - kernel(y, c, p)[:, np.newaxis] * v


def apply_kernel_addition(p, x, c, v):
    """
    Apply Gaussian kernel addition to points x.

    Parameters:
    p (list): List of kernel parameters.
    x (np.ndarray): Points to transform.
    c (list): List of center points for the kernel.
    v (list): List of transformation vectors.

    Returns:
    np.ndarray: Transformed points.
    """
    kernels = np.array([kernel(x, c[i], p[i]) for i in range(len(p))])
    zah = np.sum(kernels[:, :, np.newaxis] * v, axis=0)
    nenn = np.sum(kernels, axis=0)
    return zah / nenn[:, np.newaxis]


def application_kernel_addition(p, x, c, v):
    """
    Apply Gaussian addition to points x.

    Parameters:
    p (list): List of kernel parameters.
    x (np.ndarray): Points to transform.
    c (list): List of center points for the kernel.
    v (list): List of transformation vectors.

    Returns:
    np.ndarray: Transformed points.
    """
    return x + apply_kernel_addition(p, x, c, v)


def inverse_application_kernel_addition(p, x, c, v):
    """
    Apply inverse Gaussian addition to points x.

    Parameters:
    p (list): List of kernel parameters.
    x (np.ndarray): Points to transform.
    c (list): List of center points for the kernel.
    v (list): List of transformation vectors.

    Returns:
    np.ndarray: Transformed points.
    """
    return x - apply_kernel_addition(p, x, c, v)


def pmax(v):
    """
    Compute the maximum value for the kernel parameter.

    Parameters:
    v (np.ndarray): Transformation vectors.

    Returns:
    float: Maximum kernel parameter value.
    """
    return 1 / (np.sqrt(2) * np.linalg.norm(v)) * np.exp(1 / 2)


def sklearn_loss(x, c, v, target):
    """
    Compute the loss function for optimization.

    Parameters:
    x (np.ndarray): Points to transform.
    c (np.ndarray): Center points for the kernel.
    v (np.ndarray): Transformation vectors.
    target (np.ndarray): Target points.

    Returns:
    function: Loss function.
    """

    def loss(p):
        diffeo_result = apply_gauss_kernel(p, x, c, v)
        return np.sum(np.linalg.norm(diffeo_result - target, axis=1), axis=0)

    return loss


def sklearn_optimize(p, x, c, v, target, mu=0.9):
    """
    Optimize the kernel parameters using least squares.

    Parameters:
    p (float): Initial kernel parameter.
    x (np.ndarray): Points to transform.
    c (np.ndarray): Center points for the kernel.
    v (np.ndarray): Transformation vectors.
    target (np.ndarray): Target points.
    mu (float): Scaling factor for the kernel parameter.

    Returns:
    np.ndarray: Optimized kernel parameters.
    """
    ps = random.uniform(0, mu * pmax(v[0, :]))
    loss = sklearn_loss(x, c, v, target)
    ret = least_squares(loss, ps, bounds=[0, mu * pmax(v[0, :])])
    return ret.x


def inv_diffeo_map_with_indices(X, param, ma, v):
    """
    Compute the inverse diffeomorphism map.

    Parameters:
    X (np.ndarray): Points to transform.
    param (list): List of kernel parameters.
    ma (list): List of center points for the kernel.
    v (list): List of transformation vectors.

    Returns:
    np.ndarray: Transformed points.
    """
    Z = np.asarray(X)
    for i in range(len(ma) - 1, -1, -1):
        c = Z[ma[i], :] * np.ones(Z.shape)
        v_tmp = v[i] * np.ones(Z.shape)
        Z = inverse_apply_gauss_kernel(param[i], Z, c, v_tmp)
    return Z


def inv_diffeo_map_with_c(X, param, cs, v):
    """
    Compute the inverse diffeomorphism map with given center points.

    Parameters:
    X (np.ndarray): Points to transform.
    param (list): List of kernel parameters.
    cs (list): List of center points for the kernel.
    v (list): List of transformation vectors.

    Returns:
    np.ndarray: Transformed points.
    """
    Z = np.asarray(X)
    for i in range(len(cs) - 1, -1, -1):
        c = cs[i] * np.ones(Z.shape)
        v_tmp = v[i] * np.ones(Z.shape)
        Z = inverse_apply_gauss_kernel(param[i], Z, c, v_tmp)
    return Z


def inv_diffeo_map_addition(X, param, ma, vs):
    """
    Compute the inverse diffeomorphism map with addition.

    Parameters:
    X (np.ndarray): Points to transform.
    param (list): List of kernel parameters.
    ma (list): List of center points for the kernel.
    vs (list): List of transformation vectors.

    Returns:
    np.ndarray: Transformed points.
    """
    Z = np.asarray(X)
    for i in range(len(ma) - 1, -1, -1):
        c = [Z[m, :] for m in ma[:, i]]
        v_tmp = [vs[j : j + 3, i] * np.ones(Z.shape) for j in range(0, vs.shape[0], 3)]
        params_tmp = [params for params in param[:, i]]
        Z = inverse_application_kernel_addition(params_tmp, Z, c, v_tmp)
    return Z


def diffeo_map_with_indices(X, param, ma, v):
    """
    Compute the diffeomorphism map.

    Parameters:
    X (np.ndarray): Points to transform.
    param (list): List of kernel parameters.
    ma (list): List of center points for the kernel.
    v (list): List of transformation vectors.

    Returns:
    np.ndarray: Transformed points.
    """
    Z = np.asarray(X)
    for i in range(len(ma)):
        c = Z[int(ma[i]), :] * np.ones(Z.shape)
        v_tmp = v[i] * np.ones(Z.shape)
        Z = apply_gauss_kernel(param[i], Z, c, v_tmp)
    return Z


def diffeo_map_with_centerpoints(X, param, cs, v):
    """
    Compute the diffeomorphism map with given center points.

    Parameters:
    X (np.ndarray): Points to transform.
    param (list): List of kernel parameters.
    cs (list): List of center points for the kernel.
    v (list): List of transformation vectors.

    Returns:
    np.ndarray: Transformed points.
    """
    Z = np.asarray(X)
    for i in range(len(cs)):
        c = cs[i] * np.ones(Z.shape)
        v_tmp = v[i] * np.ones(Z.shape)
        Z = apply_gauss_kernel(param[i], Z, c, v_tmp)
    return Z


def diffeo_map_gen(X, Y, K=10, beta=0.5):
    """
    Compute the diffeomorphism map between points X and Y.

    Parameters:
    X (np.ndarray): Source points.
    Y (np.ndarray): Target points.
    K (int): Number of iterations.
    step_size (float): Step size for the transformation.
    beta (float): Scaling factor for the transformation vectors.

    all other
    Returns:
    tuple: kernel parameters, center points, indices, transformation vectors.
    """
    Z = X
    pj = []
    ms = []
    v = []
    parameters = []

    Y = np.asarray(Y)
    Z = np.asarray(Z)

    for j in range(K):
        m = np.argmax(np.linalg.norm(Z - Y, axis=1), axis=0)
        ms.append(m)
        pj.append(Z[m, :])
        q = Y[m, :]
        v.append(beta * (q - pj[j]))
        c = pj[j] * np.ones(Z.shape)
        v_tmp = v[j] * np.ones(Z.shape)
        c = np.asarray(c)
        v_tmp = np.asarray(v_tmp)
        Z_tmp = Z
        param = np.asarray([1.0])
        param = sklearn_optimize(param, Z, c, v_tmp, Y)
        parameters.append(param)

        Z = apply_gauss_kernel(param, Z_tmp, c, v_tmp)

    return parameters, pj, ms, v


def find_diffeo_map(X, Y, K=10, beta=0.5):
    """
    Find the diffeomorphism map between points X and Y.

    Parameters:
    X (np.ndarray): Source points.
    Y (np.ndarray): Target points.
    K (int): Number of iterations.
    beta (float): Scaling factor for the transformation vectors.
    L (float): Scaling factor for the indices.
    step_size (float): Step size for the transformation.

    Returns:
    tuple: Kernel parameters, indices, transformation vectors.
    """
    params, _, ms, vs = diffeo_map_gen(X, Y, K, beta)
    return np.vstack(params), ms, np.vstack(vs)


def create_extended_copy_of_tube(points, orientations, max_points, step_len=0.01):
    """
    Create an extended copy of a tube by adding points along the orientation vector.

    Parameters:
    points (np.ndarray): Array of points representing the tube.
    orientations (np.ndarray): Array of orientation vectors.
    max_points (int): Maximum number of points in the extended tube.
    step_len (float): Step length for extending the tube.

    Returns:
    np.ndarray: Extended array of points.
    """
    ez = np.array([[0, 0, 1]]).T
    missing_parts = max_points - points.shape[0]
    pc2_add = np.zeros((points.shape[0] + missing_parts, points.shape[1]))
    pc2_add[: points.shape[0], :] = points
    dv = step_len * (orientations[-1].reshape(3, 3) @ ez).flatten()
    for i in range(points.shape[0], max_points):
        pc2_add[i, :] = pc2_add[i - 1, :] + dv
    return pc2_add


def GaussKernAddtion(p, x, c, v, EIs, n_tub):
    """
    Apply Gaussian kernel addition to points x.

    Parameters:
    p (list): List of kernel parameters.
    x (np.ndarray): Points to transform.
    c (list): List of center points for the kernel.
    v (list): List of transformation vectors.
    EIs (list): List of energy integrals.
    n_tub (int): Index of the current tube.

    Returns:
    np.ndarray: Transformed points.
    """
    zah = EIs[n_tub] * (np.asarray([kernel(x, c[n_tub], p[n_tub])]).T * v[n_tub])
    return zah / np.sum(np.asarray(EIs))


def GaussKern(p, x, c, v):
    """
    Apply Gaussian kernel to points x.

    Parameters:
    p (list): List of kernel parameters.
    x (np.ndarray): Points to transform.
    c (np.ndarray): Center points for the kernel.
    v (np.ndarray): Transformation vectors.

    Returns:
    np.ndarray: Transformed points.
    """
    return np.asarray([kernel(x, c, p)]).T * v


def gauss_step(p, x, c, v, EIs, n_tub):
    """
    Perform a single step of Gaussian kernel addition.

    Parameters:
    p (list): List of kernel parameters.
    x (np.ndarray): Points to transform.
    c (list): List of center points for the kernel.
    v (list): List of transformation vectors.
    EIs (list): List of energy integrals.
    n_tub (int): Index of the current tube.

    Returns:
    np.ndarray: Transformed points.
    """
    return x + GaussKernAddtion(p, x, c, v, EIs, n_tub)


def inv_gauss_step(p, x, c, v, EIs, n_tub):
    """
    Perform a single step of inverse Gaussian kernel addition.

    Parameters:
    p (list): List of kernel parameters.
    x (np.ndarray): Points to transform.
    c (list): List of center points for the kernel.
    v (list): List of transformation vectors.
    EIs (list): List of energy integrals.
    n_tub (int): Index of the current tube.

    Returns:
    np.ndarray: Transformed points.
    """
    return x - GaussKernAddtion(p, x, c, v, EIs, n_tub)


def diffeoRz(alpha):
    """
    Create a rotation matrix for a given angle alpha around the z-axis.

    Parameters:
    alpha (float): Rotation angle in radians.

    Returns:
    np.ndarray: Rotation matrix.
    """
    return np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]
    )


def create_shifts(max_n, Ls, betas):
    """
    Create a list of shifts based on the lengths of segments.
    
    Parameters:
    max_n (int): Maximum number of points.
    Ls (list): List of segment lengths where Ls[0] is the reference/total length.
    betas (list): List of scaling factors.
    
    Returns:
    list: List of shifts as discrete indices.
    """
    # Use the first length as the reference length
    reference_length = Ls[0]
    
    # Calculate shifts
    shifts = [0]  # First shift is always 0
    
    # For each segment (in reverse, as in the original)
    for i in range(len(Ls) - 1, 0, -1):
        # Calculate the normalized segment length relative to reference
        normalized_length = Ls[i] / reference_length
        
        # Apply beta scaling
        scaled_length = normalized_length #* betas[i]
        
        # Convert to discrete index
        shift = int(max_n * scaled_length)
        shifts.append(shift)
    return shifts

def old_create_shifts(max_n, Ls, betas):
    return [0] + [int((max_n-2)*(Ls[i]/Ls[0])) for i in range(len(Ls)-1,0,-1)]

def extend_input(X):
    """
    Extend the input array by adding points along the direction of the last segment.

    Parameters:
    X (np.ndarray): Input array of points.

    Returns:
    np.ndarray: Extended array of points.
    """
    Z = X.copy()
    extended = Z[-1, :] + np.vstack(
        [i * (Z[-1, :] - Z[-2, :]) for i in range(1, Z.shape[0] * 2)]
    )
    return np.vstack([Z, extended])

def do_forward_diffeo(X, alphas, betas, params, ms, vs, EIs, Ls):
    """
    Perform the forward diffeomorphism transformation.

    Parameters:
    X (np.ndarray): Input array of points.
    alphas (list): List of rotation angles.
    betas (list): List of scaling factors.
    params (np.ndarray): Array of kernel parameters.
    ms (np.ndarray): Array of indices.
    vs (np.ndarray): Array of transformation vectors.
    EIs (list): List of energy integrals.
    Ls (list): List of segment lengths.

    Returns:
    np.ndarray: Transformed points.
    """
    # Calculate shifts
    shifts = create_shifts(X.shape[0], Ls, betas)
 
    # Extend input points
    Z = extend_input(X)

    for n in range(len(Ls),0,-1):
        for i in range(ms.shape[0]):
            c_tmp = [Z[int(m+1)+shifts[len(Ls)-1-(n-1)],:] * np.ones(Z.shape) for m in ms[i,:]]
            v_tmp = [diffeoRz(alphas[j//3])@vs[i,j:j+3].T * np.ones(Z.shape) for j in range(0,vs.shape[1],3)]
            params_tmp = [param for param in params[i,:]] 

            for j in range(n):
                if n > 0:
                    Z = gauss_step(params_tmp, Z, c_tmp, v_tmp, EIs[:n],j)

    # Calculate the total path length of the input points X based on beta
    path_length_X = (1-betas[0]) * Ls[0]

    # Create a mask to identify valid points in Z based on the cumulative path length and beta
    valid_points_mask = np.cumsum(np.linalg.norm(np.diff(Z, axis=0), axis=1)) <= path_length_X

    # Ensure the first point is always included in the mask
    valid_points_mask = np.insert(valid_points_mask, 0, True)

    # Return only the valid points from Z
    return Z[valid_points_mask]


def do_inverse_diffeo(X, alphas, betas, params, ms, vs, EIs, Ls):
    """
    Perform the inverse diffeomorphism transformation.

    Parameters:
    X (np.ndarray): Input array of points.
    alphas (list): List of rotation angles.
    betas (list): List of scaling factors.
    params (np.ndarray): Array of kernel parameters.
    ms (np.ndarray): Array of indices.
    vs (np.ndarray): Array of transformation vectors.
    EIs (list): List of energy integrals.
    Ls (list): List of segment lengths.

    Returns:
    np.ndarray: Transformed points.
    """
    # Calculate shifts
    shifts = create_shifts(X.shape[0], Ls, betas)

    # Extend input points
    Z = extend_input(X)

    for n in range(len(Ls),0,-1):
        for i in range(ms.shape[0]):
            c_tmp = [Z[int(m)+shifts[len(Ls)-1-(n-1)],:] * np.ones(Z.shape) for m in ms[i,:]]
            v_tmp = [diffeoRz(alphas[j//3])@vs[i,j:j+3].T * np.ones(Z.shape) for j in range(0,vs.shape[1],3)]
            params_tmp = [param for param in params[i,:]] 

            for j in range(n):
                if n > 0:
                    Z = inv_gauss_step(params_tmp, Z, c_tmp, v_tmp, EIs[:n],j)

    # Calculate the total path length of the input points X based on beta
    path_length_X = (1-betas[0]) * Ls[0]

    # Create a mask to identify valid points in Z based on the cumulative path length and beta
    valid_points_mask = np.cumsum(np.linalg.norm(np.diff(Z, axis=0), axis=1)) <= path_length_X

    # Ensure the first point is always included in the mask
    valid_points_mask = np.insert(valid_points_mask, 0, True)

    # Return only the valid points from Z
    return Z[valid_points_mask]
