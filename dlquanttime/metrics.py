"""Evaluation metrics used to validate metabolite quantification accuracy.

Implements Pearson correlation and the two-way random-effects, single
measure intraclass correlation coefficient ICC(2,1) (McGraw & Wong, 1996),
as used to compare model predictions against ground-truth (synthetic) or
reference (LCModel) metabolite amplitudes.
"""

import numpy as np


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson product-moment correlation coefficient between two 1-D arrays."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size < 2:
        raise ValueError("At least two samples are required")

    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = np.sqrt(np.sum(yt**2) * np.sum(yp**2))
    if denom == 0:
        return 0.0
    return float(np.sum(yt * yp) / denom)


def icc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Two-way random-effects, single-measure ICC, i.e. ICC(2,1).

    Both inputs are treated as two "raters" (e.g. ground truth and model
    prediction) measuring the same set of "subjects" (samples).

    Parameters
    ----------
    y_true, y_pred:
        1-D arrays of equal length ``n`` (number of subjects).

    Returns
    -------
    float
        The ICC(2,1) agreement coefficient.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    n = y_true.size
    if n < 2:
        raise ValueError("At least two samples are required")

    data = np.stack([y_true, y_pred], axis=1)  # shape (n, k=2)
    k = 2

    subject_means = data.mean(axis=1)
    rater_means = data.mean(axis=0)
    grand_mean = data.mean()

    ss_total = np.sum((data - grand_mean) ** 2)
    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    ss_raters = n * np.sum((rater_means - grand_mean) ** 2)
    ss_error = ss_total - ss_subjects - ss_raters

    df_subjects = n - 1
    df_raters = k - 1
    df_error = (n - 1) * (k - 1)

    ms_subjects = ss_subjects / df_subjects
    ms_raters = ss_raters / df_raters if df_raters > 0 else 0.0
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    denom = ms_subjects + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
    if denom == 0:
        return 0.0
    return float((ms_subjects - ms_error) / denom)
