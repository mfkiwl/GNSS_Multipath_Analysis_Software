"""
This moduling is for detecting  cycle slips in the GNSS phase observations.

Made by: Per Helge Aarnes
E-mail: per.helge.aarnes@gmail.com
"""

import numpy as np


def detectCycleSlips(estimates, missing_obs_overview, epoch_first_obs, epoch_last_obs, tInterval, crit_slip_rate):
    """
    Detect epochs with cycle slips based on a critical rate of change
    and missing-observation flags.

    Parameters
    ----------
    estimates : ndarray, shape (nepochs, nPRN+1)
        Linear-combination estimates per epoch and PRN (column 0 unused).
    missing_obs_overview : ndarray, same shape as *estimates*
        1 where a satellite has no estimate at that epoch, 0 otherwise.
    epoch_first_obs : ndarray, shape (nPRN+1,)
        Epoch index of first observation per PRN (NaN if none).
    epoch_last_obs : ndarray, shape (nPRN+1,)
        Epoch index of last observation per PRN (NaN if none).
    tInterval : float
        Observation interval in seconds.
    crit_slip_rate : float
        Critical rate of change to flag a cycle slip [m/s].

    Returns
    -------
    dict[str, list[int]]
        Keys "1" … "nPRN", values are sorted epoch indices with detected slips.
    """
    n_prn = estimates.shape[1] - 1
    rate_of_change = np.diff(estimates, axis=0) / tInterval
    exceeds_threshold = np.abs(rate_of_change) > crit_slip_rate

    slip_epochs = {str(prn): [] for prn in range(1, n_prn + 1)}

    # Collect epochs where rate of change exceeds the critical value
    rows, cols = np.where(exceeds_threshold)
    for col, row in zip(cols, rows):
        slip_epochs[str(col)].append(row)

    # Merge in epochs flagged as missing observations
    if epoch_first_obs.size > 0:
        for prn in range(len(epoch_first_obs)):
            if np.isnan(epoch_first_obs[prn]) or np.isnan(epoch_last_obs[prn]):
                continue
            first = int(epoch_first_obs[prn])
            last = int(epoch_last_obs[prn])
            missing_epochs = np.where(missing_obs_overview[first:last, prn] == 1)[0] + first
            if missing_epochs.size > 0:
                slip_epochs[str(prn)].extend(missing_epochs.tolist())

    # Remove duplicates and sort
    for key in slip_epochs:
        slip_epochs[key] = sorted(set(slip_epochs[key]))

    return slip_epochs


def orgSlipEpochs(slip_epochs):
    """
    Group individual slip epochs into contiguous slip periods.

    Parameters
    ----------
    slip_epochs : ndarray
        Sorted epoch indices where cycle slips were detected.

    Returns
    -------
    slip_periods : ndarray, shape (N, 2) or list
        Each row is ``[start_epoch, end_epoch]``. Empty list if no slips.
    n_slip_periods : int
        Number of slip periods.
    """
    if len(slip_epochs) == 0:
        return [], 0

    gaps = np.diff(slip_epochs) != 1
    starts = np.concatenate(([slip_epochs[0]], slip_epochs[1:][gaps]))
    ends = np.concatenate((slip_epochs[:-1][gaps], [slip_epochs[-1]]))

    slip_periods = np.column_stack((starts, ends)).astype(float)
    return slip_periods, len(starts)


