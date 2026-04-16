import numpy as np

_LLI_SLIP_CODES = [1, 2, 3, 5, 6, 7]


def getLLISlipPeriods(LLI_current_phase):
    """
    Group LLI-indicated ambiguity slips from a RINEX observation file into
    contiguous slip periods per satellite.

    Parameters
    ----------
    LLI_current_phase : ndarray, shape (n_epochs, n_sat + 1)
        LLI indicators for every epoch and satellite.
        Column 0 is unused; columns 1 … n_sat hold per-satellite values.

    Returns
    -------
    dict[int, ndarray | list]
        Keys are 0-based satellite indices.  Values are (N, 2) float arrays
        where each row is ``[start_epoch, end_epoch]``, or an empty list
        when no slips were detected.
    """
    _, n_cols = LLI_current_phase.shape
    n_sat = n_cols - 1
    slip_periods = {}

    for sat in range(n_sat):
        slip_epochs = np.where(np.isin(LLI_current_phase[:, sat + 1], _LLI_SLIP_CODES))[0]

        if slip_epochs.size == 0:
            slip_periods[sat] = []
            continue

        # Identify boundaries: gaps > 1 between consecutive slip epochs
        gaps = np.diff(slip_epochs) != 1
        starts = np.concatenate(([slip_epochs[0]], slip_epochs[1:][gaps]))
        ends = np.concatenate((slip_epochs[:-1][gaps], [slip_epochs[-1]]))

        slip_periods[sat] = np.column_stack((starts, ends)).astype(float)

    return slip_periods