"""Ordered-contour vector-Huber fit shared by pseudo targets and inference."""
import cv2
import numpy as np


def sampling_matrix(count=32):
    if count < 8 or count % 4:
        raise ValueError('Four ordered edges must have equal sample counts.')
    per_edge = count // 4
    matrix = np.zeros((count, 4), np.float64)
    for index in range(count):
        edge, offset = divmod(index, per_edge)
        t = offset / per_edge
        matrix[index, edge] = 1. - t
        matrix[index, (edge + 1) % 4] = t
    return matrix


def fit_quadrilateral(contour, robust=True, delta_fraction=.1, iterations=50):
    """Joint vector-Huber IRLS; equal weights give ordinary least squares.

    Assumes the existing GT encoder's edge ordering and uniform per-edge t.
    Huber cutoff is fixed at 0.1 * max(y IQR, 1 grid pixel), not tuned to GT.
    Returns centered samples, matching the existing projection interface.
    """
    points = np.asarray(contour, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        return None, {}
    matrix = sampling_matrix(len(points))
    center = points.mean(axis=0)
    centered = points - center
    corners = np.linalg.lstsq(matrix, centered, rcond=None)[0]
    delta = delta_fraction * max(float(np.subtract(*np.percentile(points[:, 1], [75, 25]))), 1.)
    if delta <= 0:
        raise ValueError('Huber cutoff must be positive.')
    steps = 0
    if robust:
        for steps in range(1, iterations + 1):
            residual = np.linalg.norm(matrix @ corners - centered, axis=1)
            weights = np.minimum(1., delta / np.maximum(residual, 1e-12))
            root_weight = np.sqrt(weights)[:, None]
            fitted = np.linalg.lstsq(root_weight * matrix, root_weight * centered, rcond=None)[0]
            change = np.max(np.abs(fitted - corners))
            corners = fitted
            if change < 1e-8:
                break
    residual = np.linalg.norm(matrix @ corners - centered, axis=1)
    weights = np.minimum(1., delta / np.maximum(residual, 1e-12)) if robust else np.ones(len(points))
    polygon = corners.astype(np.float32)
    info = dict(cutoff_grid=delta, weights=weights.tolist(), iterations=steps,
                mean_fit_residual_grid=float(residual.mean()))
    if cv2.contourArea(polygon) <= 1e-6 or not cv2.isContourConvex(polygon):
        return None, info
    projected = matrix @ corners
    return (projected - projected.mean(axis=0)).astype(np.float32), info
