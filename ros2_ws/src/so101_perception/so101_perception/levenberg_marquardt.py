from collections.abc import Callable

import numpy as np

JACOBIAN_STEP = 1e-7
INITIAL_DAMPING = 1e-3
DAMPING_FACTOR = 4.0
RIDGE = 1e-12
CONVERGED_STEP = 1e-10


def huber_weights(residuals: np.ndarray, scale: float) -> np.ndarray:
    magnitude = np.abs(residuals)
    return np.where(magnitude <= scale, 1.0, scale / np.maximum(magnitude, 1e-12))


def numeric_jacobian(residuals: Callable[[np.ndarray], np.ndarray], x: np.ndarray, at: np.ndarray) -> np.ndarray:
    columns = []
    for index in range(len(x)):
        shifted = x.copy()
        shifted[index] += JACOBIAN_STEP
        columns.append((residuals(shifted) - at) / JACOBIAN_STEP)
    return np.column_stack(columns)


def levenberg_marquardt(
    residuals: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    robust_scale: float | None = None,
    iterations: int = 80,
) -> np.ndarray:
    x = np.asarray(start, dtype=np.float64).copy()
    damping = INITIAL_DAMPING
    for _ in range(iterations):
        current = residuals(x)
        weights = np.ones_like(current) if robust_scale is None else huber_weights(current, robust_scale)
        jacobian = numeric_jacobian(residuals, x, current)
        normal = jacobian.T @ (weights[:, None] * jacobian)
        gradient = jacobian.T @ (weights * current)
        step = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + RIDGE), -gradient)
        if np.linalg.norm(step) < CONVERGED_STEP:
            break
        candidate = residuals(x + step)
        if np.sum(weights * candidate**2) < np.sum(weights * current**2):
            x = x + step
            damping /= DAMPING_FACTOR
        else:
            damping *= DAMPING_FACTOR
    return x
