import numpy as np
import matplotlib.pyplot as plt


def camera_center(M):
    """
    Extract the camera center in world coordinates from a 3x4 projection
    matrix.  Recall M = [A | m4], so:
        C = -A^{-1} m4

    :param M: 3x4 projection matrix
    :return: length-3 numpy array, the camera center in world coordinates
    """
    C = None
    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    M_left = M[:, :3]
    m4 = M[:, 3]
    C = -np.linalg.inv(M_left) @ m4
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################
    return C


def project(M, points3d):
    """
    Forward-project 3D world points to 2D image coordinates using M.
        [su, sv, s]^T = M @ [X, Y, Z, 1]^T
        (u, v) = (su / s, sv / s)

    :param M: 3x4 projection matrix
    :param points3d: N x 3 array of 3D world coordinates
    :return: N x 2 array of 2D image coordinates (u, v)
    """
    points2d = None
    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    ones = np.ones((points3d.shape[0], 1))
    points3d_hom = np.hstack([points3d, ones])
    projected = M @ points3d_hom.T
    points2d = (projected[:2, :] / projected[2, :]).T
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################
    return points2d


def reprojection_error(M, points3d, points2d):
    """
    Per-point L2 reprojection error: how far each projected 3D point
    lands from its observed 2D position.
        error_i = || project(M, P_i) - p_i ||_2

    :param M: 3x4 projection matrix
    :param points3d: N x 3 array of 3D world coordinates
    :param points2d: N x 2 array of observed 2D image coordinates
    :return: length-N array of L2 reprojection errors
    """
    errors = None
    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    projected = project(M, points3d)
    errors = np.linalg.norm(projected - points2d, axis=1)

    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################
    return errors


def estimate_camera_matrix(points2d, points3d):
    """
    Estimate the 3x4 camera matrix M from 2D-3D correspondences via
    the Direct Linear Transform (DLT).

    Build the 2N x 12 matrix A from the homogeneous system Am = 0.  Each
    correspondence (u,v) <-> (X,Y,Z) gives two rows:
        [X Y Z 1  0 0 0 0  -uX -uY -uZ -u]
        [0 0 0 0  X Y Z 1  -vX -vY -vZ -v]

    Solve via SVD: the solution m is the last column of V (from A = U S V^T).
    Reshape m into the 3x4 matrix M.

    **Important**: After solving, normalize M so that the norm of the first
    three elements of the third row equals 1 (i.e., ||M[2,:3]|| = 1).
    Also ensure M[2,3] > 0 (points in front of the camera have positive depth).
    This normalization is necessary because DLT determines M only up to scale,
    and downstream functions (like `projection_2d_to_3d`) require properly
    scaled depth values.

    After normalizing, compute and return the residual as the sum of squared
    reprojection errors using our reprojection_error() function.

    :param points2d: N x 2 array of 2D image coordinates
    :param points3d: N x 3 array of corresponding 3D world coordinates
    :return: M, the 3x4 camera matrix (normalized)
             residual, the sum of squared reprojection error (scalar)
    """
    M = None
    residual = None
    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    N = points2d.shape[0]
    A = np.zeros((2 * N, 12))
    
    for i in range(N):
        X, Y, Z = points3d[i]
        u, v = points2d[i]
        
        A[2*i] = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
    
    U, S, Vt = np.linalg.svd(A)
    m = Vt[-1, :]
    M = m.reshape(3, 4)
    
    norm_m3 = np.linalg.norm(M[2, :3])
    M = M / norm_m3
    
    if M[2, 3] < 0:
        M = -M
    
    residual = np.sum(reprojection_error(M, points3d, points2d) ** 2)
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################
    return M, residual


def plot_box_and_camera(points_3d, cc, R):
    """
    Visualize the actual 3D points and the estimated 3D camera center.

    :param points_3d: N x 3 array of 3D world points
    :param cc: length-3 array, camera center in world coordinates
    :param R: 3x3 rotation matrix (columns are camera axes in world frame)
    """
    print("The camera center is at: \n", cc)

    v1 = R[:, 0]
    v2 = R[:, 1]
    v3 = R[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue',
               marker='o', s=10, depthshade=0)
    cc = np.array(cc).squeeze()
    ax.scatter(cc[0], cc[1], cc[2], c='red',
               marker='x', s=20, depthshade=0)

    cc0, cc1, cc2 = cc

    point0 = points_3d[0]
    ax.plot3D([point0[0], point0[0]+2], [point0[1], point0[1]], [point0[2], point0[2]], c='r')
    ax.plot3D([point0[0], point0[0]], [point0[1], point0[1]+2], [point0[2], point0[2]], c='g')
    ax.plot3D([point0[0], point0[0]], [point0[1], point0[1]], [point0[2], point0[2]+2], c='b')

    ax.plot3D([cc0, cc0+v1[0]], [cc1, cc1+v1[1]], [cc2, cc2+v1[2]], c='r')
    ax.plot3D([cc0, cc0+v2[0]], [cc1, cc1+v2[1]], [cc2, cc2+v2[2]], c='g')
    ax.plot3D([cc0, cc0+v3[0]], [cc1, cc1+v3[1]], [cc2, cc2+v3[2]], c='b')

    min_z = min(points_3d[:, 2])
    min_x = min(points_3d[:, 0])
    min_y = min(points_3d[:, 1])
    for p in points_3d:
        x, y, z = p
        ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c='black', linewidth=1)
        ax.plot3D(xs=[x, min_x], ys=[y, y], zs=[z, z], c='black', linewidth=1)
        ax.plot3D(xs=[x, x], ys=[y, min_y], zs=[z, z], c='black', linewidth=1)
    x, y, z = cc
    ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c='black', linewidth=1)
