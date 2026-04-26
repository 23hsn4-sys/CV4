import numpy as np

from scripts.pnp import camera_center, project, reprojection_error, estimate_camera_matrix


def test_estimate_camera_matrix():
    """
    Test DLT functions: camera_center, project, reprojection_error,
    estimate_camera_matrix.
    """
    # Known projection matrix (constructed from K, R, t)
    K = np.array([[480,   0, 240],
                  [  0, 640, 320],
                  [  0,   0,   1]], dtype=float)

    # Simple rotation and translation
    R = np.eye(3)
    t = np.array([[0], [0], [3]], dtype=float)

    # M = K @ [R | t]
    M_true = K @ np.hstack([R, t])

    # 3D box points
    box_3d = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.5, 0.5, 1.0],
    ])

    # Generate 2D points from the known M
    pts_h = np.hstack([box_3d, np.ones((8, 1))])
    proj = (M_true @ pts_h.T).T
    box_2d = proj[:, :2] / proj[:, 2:3]

    # --- Test camera_center ---
    cc = camera_center(M_true)
    expected_cc = np.array([0, 0, -3])  # C = -R^T @ t
    assert cc is not None, "camera_center returned None"
    assert np.allclose(cc, expected_cc, atol=0.01), \
        f"camera_center: expected {expected_cc}, got {cc}"

    # --- Test project ---
    proj_2d = project(M_true, box_3d)
    assert proj_2d is not None, "project returned None"
    assert np.allclose(proj_2d, box_2d, atol=0.01), \
        f"project: max error {np.max(np.abs(proj_2d - box_2d))}"

    # --- Test reprojection_error ---
    errors = reprojection_error(M_true, box_3d, box_2d)
    assert errors is not None, "reprojection_error returned None"
    assert np.allclose(errors, 0, atol=0.01), \
        f"reprojection_error: expected ~0, got {errors}"

    # --- Test estimate_camera_matrix (DLT) ---
    M_est, residual = estimate_camera_matrix(box_2d, box_3d)
    assert M_est is not None, "estimate_camera_matrix returned None"

    # M is determined up to scale; normalize both
    M_est_n = M_est / np.linalg.norm(M_est)
    M_true_n = M_true / np.linalg.norm(M_true)
    # Could be negated
    if M_est_n.flat[0] * M_true_n.flat[0] < 0:
        M_est_n = -M_est_n

    assert np.allclose(M_est_n, M_true_n, atol=0.05), \
        f"estimate_camera_matrix: M doesn't match (max diff {np.max(np.abs(M_est_n - M_true_n))})"

    # Residual should be small
    assert residual < 1.0, \
        f"estimate_camera_matrix: residual too large ({residual})"

    # Test that camera_center works on the estimated M
    cc_est = camera_center(M_est)
    assert np.allclose(cc_est, expected_cc, atol=0.5), \
        f"camera_center on estimated M: expected ~{expected_cc}, got {cc_est}"
