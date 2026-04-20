"""
Multipath / ionospheric / cycle-slip analysis for one signal combination.

Public API:
    SignalAnalyzer  - object-oriented analyzer (preferred entry point)
    SignalStats     - dataclass with per-combination results (with .to_dict())
    SlipPeriods     - dataclass holding range1/ambiguity/LLI slip data

Made by: Per Helge Aarnes
E-mail: per.helge.aarnes@gmail.com
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from gnssmultipath.estimateSignalDelays import estimateSignalDelays
from gnssmultipath.getLLISlipPeriods import getLLISlipPeriods
from gnssmultipath.computeDelayStats import computeDelayStats


SlipDict = Dict[int, np.ndarray]


@dataclass(slots=True)
class SlipPeriods:
    """Cycle-slip periods detected for a single signal combination.

    Attributes
    ----------
    range1: Dict[int, np.ndarray]
        Slip periods detected from the range1 / phase1 ionosphere-free
        combination, keyed by 1-based PRN. Each value is an ``(n, 2)``
        integer array of ``[start_epoch, end_epoch]`` pairs (0-based,
        inclusive). Empty entry ``[]`` means no slip detected for that PRN.
    ambiguity : Dict[int, np.ndarray]
        Slip periods detected from the geometry-free / ionospheric
        combination (rate of change of L1 - L2). Same layout as ``range1``.
    lli : np.ndarray
        Slip periods derived from the RINEX Loss-of-Lock Indicator on the
        phase1 observation. Shape ``(nepochs, n_sat)``; non-zero entries
        flag epochs where the receiver reported a loss of lock.
    """

    range1: SlipDict
    ambiguity: SlipDict
    lli: np.ndarray

    def filter_by_elevation(self, valid_elevation: np.ndarray) -> None:
        """Drop slip periods whose start or end epoch is below the cutoff.

        Mutates ``range1`` and ``ambiguity`` in place. The LLI slip array is
        already epoch-aligned and not touched.
        """
        for slip_dict in (self.range1, self.ambiguity):
            for sat in range(len(slip_dict)):
                periods = np.asarray(slip_dict[sat + 1], dtype=np.intp)
                if periods.size == 0:
                    continue
                col = sat + 1
                keep = (
                    valid_elevation[periods[:, 0], col]
                    & valid_elevation[periods[:, 1], col]
                )
                slip_dict[sat + 1] = periods[keep]


@dataclass(slots=True)
class SignalStats:
    """Statistics from a single signal-combination analysis.

    See :func:`gnssmultipath.computeDelayStats.computeDelayStats` for the
    full numerical definition of every aggregate.

    Multipath
    ---------
    mean_multipath_range1_satellitewise : np.ndarray
        Per-satellite mean multipath delay on the range1 code (meters).
    mean_multipath_range1_overall : float
        Mean multipath delay across all satellites and epochs (meters).
    rms_multipath_range1_satellitewise : np.ndarray
        Per-satellite RMS of the range1 multipath delay (meters).
    rms_multipath_range1_averaged : float
        Mean of the per-satellite RMS values (meters).
    elevation_weighted_rms_multipath_range1_satellitewise : np.ndarray
        Per-satellite elevation-weighted RMS of the range1 multipath delay,
        using ``w(E) = min(4 sin^2(E), 1)`` (meters).
    elevation_weighted_average_rms_multipath_range1 : float
        Mean of the elevation-weighted per-satellite RMS values (meters).

    Ionospheric delay
    -----------------
    mean_ion_delay_phase1_satellitewise : np.ndarray
        Per-satellite mean ionospheric delay on phase1 (meters).
    mean_ion_delay_phase1_overall : float
        Mean ionospheric delay across all satellites and epochs (meters).

    Geometry / counts
    -----------------
    mean_sat_elevation_angles : np.ndarray
        Per-satellite mean elevation angle of the satellites that produced
        an estimate (degrees).
    nEstimates : int
        Total number of multipath estimates across all satellites.
    nEstimates_per_sat : np.ndarray
        Number of multipath estimates per satellite.
    n_range1_obs_per_sat : np.ndarray
        Number of range1 observations per satellite (above the cutoff).
    nRange1Obs : int
        Total number of range1 observations.

    Slip distributions
    ------------------
    range1_slip_distribution_per_sat : np.ndarray
        Per-satellite count of cycle slips detected from the
        range1 / phase1 combination.
    range1_slip_distribution : int
        Total count of range1 / phase1 cycle slips.
    cycle_slip_distribution_per_sat : np.ndarray
        Per-satellite count of cycle slips detected from the
        ionospheric (geometry-free) combination.
    cycle_slip_distribution : int
        Total count of ionospheric-combination cycle slips.
    LLI_slip_distribution_per_sat : np.ndarray
        Per-satellite count of cycle slips reported by the RINEX LLI.
    LLI_slip_distribution : int
        Total count of LLI-reported cycle slips.
    slip_distribution_per_sat_LLI_fusion : np.ndarray
        Per-satellite count of cycle slips after fusing the algorithmic
        and LLI-reported slips (deduplicated).
    slip_distribution_LLI_fusion : int
        Total count of cycle slips after the LLI fusion.

    Raw arrays (for plotting / audit)
    ---------------------------------
    range1_observations : np.ndarray
        Range1 code observations after the elevation-cutoff mask, shape
        ``(nepochs, max_sat + 1)``.
    phase1_observations : np.ndarray
        Phase1 carrier-phase observations after the elevation-cutoff mask,
        shape ``(nepochs, max_sat + 1)``.
    ion_delay_phase1 : np.ndarray
        Estimated ionospheric delay on phase1 after the elevation-cutoff
        mask, shape ``(nepochs, max_sat + 1)``.
    multipath_range1 : np.ndarray
        Estimated range1 multipath delay after the elevation-cutoff mask,
        shape ``(nepochs, max_sat + 1)``.
    sat_elevation_angles : np.ndarray
        Satellite elevation angles passed to the analyzer, shape
        ``(nepochs, max_sat + 1)``.

    Codes / metadata
    ----------------
    range1_Code, range2_Code : str
        RINEX observation codes for the range1 and range2 inputs.
    phase1_Code, phase2_Code : str
        Derived phase observation codes (``'L' + range_code[1:]``).

    Slip periods (per satellite)
    ----------------------------
    range1_slip_periods : Dict[int, np.ndarray]
        Range1 / phase1 slip periods after the elevation-cutoff filter.
    cycle_slip_periods : Dict[int, np.ndarray]
        Ionospheric-combination slip periods after the elevation-cutoff
        filter.
    """

    mean_multipath_range1_satellitewise: np.ndarray
    mean_multipath_range1_overall: float
    rms_multipath_range1_satellitewise: np.ndarray
    rms_multipath_range1_averaged: float
    mean_ion_delay_phase1_satellitewise: np.ndarray
    mean_ion_delay_phase1_overall: float
    mean_sat_elevation_angles: np.ndarray
    nEstimates: int
    nEstimates_per_sat: np.ndarray
    n_range1_obs_per_sat: np.ndarray
    nRange1Obs: int
    range1_slip_distribution_per_sat: np.ndarray
    range1_slip_distribution: int
    cycle_slip_distribution_per_sat: np.ndarray
    cycle_slip_distribution: int
    LLI_slip_distribution_per_sat: np.ndarray
    LLI_slip_distribution: int
    slip_distribution_per_sat_LLI_fusion: np.ndarray
    slip_distribution_LLI_fusion: int
    elevation_weighted_rms_multipath_range1_satellitewise: np.ndarray
    elevation_weighted_average_rms_multipath_range1: float
    range1_observations: np.ndarray
    phase1_observations: np.ndarray
    ion_delay_phase1: np.ndarray
    multipath_range1: np.ndarray
    sat_elevation_angles: np.ndarray
    range1_Code: str
    range2_Code: str
    phase1_Code: str
    phase2_Code: str
    range1_slip_periods: SlipDict
    cycle_slip_periods: SlipDict

    def to_dict(self) -> Dict[str, object]:
        """Return a flat dict view of the stats.

        The downstream pipeline (plotting, output writers, pickled
        ``analysisResults``) consumes the stats by string keys. ``__slots__``
        avoids the per-instance ``__dict__`` overhead, and we build the dict
        directly (no ``dataclasses.asdict`` deepcopy).
        """
        return {name: getattr(self, name) for name in self.__slots__}


class SignalAnalyzer:
    """Run a multipath / ionospheric / cycle-slip analysis for one signal pair.

    Inputs are organized as object attributes; carrier frequencies and the
    GNSS-system index are resolved once in ``__init__`` so ``run()`` does no
    redundant lookups. Call :meth:`run` to execute the pipeline.

    Parameters
    ----------
    gnss_system : str
        Code identifying the current GNSS system. Example: ``"G"`` (GPS) or
        ``"E"`` (Galileo).
    range1_code : str
        RINEX observation code for the first code pseudorange observation
        (e.g. ``"C1C"``).
    range2_code : str
        RINEX observation code for the second code pseudorange observation
        (e.g. ``"C2W"``).
    gnss_systems : Mapping[int, str]
        Mapping from system index to GNSS system letter (e.g.
        ``{1: "G", 2: "R", ...}``).
    frequency_overview : Mapping[int, np.ndarray]
        Per-system carrier frequencies. Each entry holds the carrier-band
        frequencies for one GNSS system, in the order given by
        ``gnss_systems``. For GLONASS the entry is a matrix where row ``i``
        contains the carrier-band-``i`` frequencies for every GLONASS SV;
        for the other systems each entry is an array with one frequency per
        carrier band.
    nepochs : int
        Number of observation epochs in the RINEX file.
    t_interval : float
        Observation interval in seconds.
    max_sat : int
        Maximum PRN number for the current GNSS system.
    gnss_svs : np.ndarray
        Matrix of observed satellites for the current GNSS system. Element
        ``GNSS_SVs[epoch, 0]`` is the number of observed satellites at that
        epoch; ``GNSS_SVs[epoch, j>0]`` are the PRNs of the observed
        satellites.
    obs_codes : Mapping[str, List[str]]
        Mapping from GNSS-system letter to the list of observation codes
        available in that system. Each code is a three-character string:
        the first character (capital letter) is the observation type
        (e.g. ``"L"`` or ``"C"``), the second (digit) is the frequency
        code, and the third (capital letter) is the attribute
        (e.g. ``"P"`` or ``"X"``).
    gnss_obs : Mapping[int, np.ndarray]
        Per-epoch dictionary of observation matrices for the current GNSS
        system. ``gnss_obs[epoch][PRN, obs_type_idx]`` holds the observation;
        ``obs_type_idx`` follows the same order as ``obs_codes`` for the
        current system.
    gnss_lli : Mapping[int, np.ndarray]
        Per-epoch dictionary of Loss-of-Lock-Indicator matrices for the
        current GNSS system. Layout matches ``gnss_obs``.
    sat_elevation_angles : np.ndarray
        Satellite elevation angles, shape ``(nepochs, max_sat + 1)``.
        ``sat_elevation_angles[epoch, PRN]`` is the elevation angle in
        degrees. NaN entries denote missing data.
    phase_code_limit : float
        Critical limit (m/s) that flags a cycle slip in the phase-code
        combination. Pass ``0`` to use the built-in default.
    ion_limit : float
        Critical limit (m/s) that flags a cycle slip in the rate of change
        of the ionospheric delay. Pass ``0`` to use the built-in default.
    cutoff_elevation_angle : float
        Critical elevation cutoff in degrees. Estimates and slip periods
        whose elevation falls below this value are removed.
    """

    __slots__ = (
        "gnss_system", "range1_code", "range2_code",
        "phase1_code", "phase2_code",
        "nepochs", "t_interval", "max_sat",
        "gnss_svs", "obs_codes", "gnss_obs", "gnss_lli",
        "sat_elevation_angles",
        "phase_code_limit", "ion_limit", "cutoff_elevation_angle",
        "_carrier_freq1", "_carrier_freq2",
        "_phase1_col_idx",
    )

    def __init__(
        self,
        gnss_system: str,
        range1_code: str,
        range2_code: str,
        gnss_systems: Mapping[int, str],
        frequency_overview: Mapping[int, np.ndarray],
        nepochs: int,
        t_interval: float,
        max_sat: int,
        gnss_svs: np.ndarray,
        obs_codes: Mapping[str, List[str]],
        gnss_obs: Mapping[int, np.ndarray],
        gnss_lli: Mapping[int, np.ndarray],
        sat_elevation_angles: np.ndarray,
        phase_code_limit: float,
        ion_limit: float,
        cutoff_elevation_angle: float,
    ) -> None:
        self.gnss_system = gnss_system
        self.range1_code = range1_code
        self.range2_code = range2_code
        self.phase1_code = "L" + range1_code[1:]
        self.phase2_code = "L" + range2_code[1:]
        self.nepochs = nepochs
        self.t_interval = t_interval
        self.max_sat = max_sat
        self.gnss_svs = gnss_svs
        self.obs_codes = obs_codes
        self.gnss_obs = gnss_obs
        self.gnss_lli = gnss_lli
        self.sat_elevation_angles = sat_elevation_angles
        self.phase_code_limit = phase_code_limit
        self.ion_limit = ion_limit
        self.cutoff_elevation_angle = cutoff_elevation_angle

        # Resolve once: GNSS system index -> carrier frequencies
        gnss_index = next(k for k, v in gnss_systems.items() if v == gnss_system)
        band1 = int(range1_code[1]) - 1
        band2 = int(range2_code[1]) - 1
        freq_table = frequency_overview[gnss_index]
        if gnss_system == "R":
            self._carrier_freq1 = freq_table[band1, :]
            self._carrier_freq2 = freq_table[band2, :]
        else:
            self._carrier_freq1 = freq_table[band1, 0]
            self._carrier_freq2 = freq_table[band2, 0]

        # Resolve phase1 column index in the LLI matrix once
        try:
            self._phase1_col_idx = obs_codes[gnss_system].index(self.phase1_code)
        except ValueError:
            self._phase1_col_idx = -1

    def _build_lli_phase_matrix(self) -> np.ndarray:
        """Return the LLI matrix for the phase1 code: shape (nepochs, n_sat)."""
        col = self._phase1_col_idx
        gnss_lli = self.gnss_lli
        if col < 0:
            # phase code not present in header -> no LLI to read
            n_sat_cols = gnss_lli[1].shape[0]
            return np.zeros((self.nepochs, n_sat_cols))
        nepochs = self.nepochs
        # Pre-allocate and fill (avoids list-of-arrays + np.array concatenation).
        first = gnss_lli[1]
        out = np.empty((nepochs, first.shape[0]), dtype=first.dtype)
        out[0] = first[:, col]
        for ep in range(1, nepochs):
            out[ep] = gnss_lli[ep + 1][:, col]
        return out

    def run(self) -> Tuple[Optional[SignalStats], bool]:
        """Execute the full pipeline. Returns ``(stats, success)``.

        On failure (``estimateSignalDelays`` returns ``success=False``)
        returns ``(None, False)``.
        """
        (
            ion_delay_phase1, multipath_range1,
            range1_slip_periods, ambiguity_slip_periods,
            range1_observations, phase1_observations, success,
        ) = estimateSignalDelays(
            self.range1_code, self.range2_code,
            self.phase1_code, self.phase2_code,
            self._carrier_freq1, self._carrier_freq2,
            self.nepochs, self.max_sat,
            self.gnss_svs, self.obs_codes, self.gnss_obs,
            self.gnss_system, self.t_interval,
            self.phase_code_limit, self.ion_limit,
        )

        if not success:
            return None, success

        # Boolean cutoff mask. NaN elevations are treated as below cutoff so
        # missing-data epochs are filtered out alongside low-elevation ones.
        elev = self.sat_elevation_angles
        valid_elevation = np.isfinite(elev) & (elev >= self.cutoff_elevation_angle)
        # Apply the mask in place using the float view of the boolean.
        # `np.multiply(out=...)` avoids allocating a temporary on every line.
        mask = valid_elevation.astype(np.float64, copy=False)
        np.multiply(ion_delay_phase1, mask, out=ion_delay_phase1)
        np.multiply(multipath_range1, mask, out=multipath_range1)
        np.multiply(range1_observations, mask, out=range1_observations)
        np.multiply(phase1_observations, mask, out=phase1_observations)

        slips = SlipPeriods(
            range1=range1_slip_periods,
            ambiguity=ambiguity_slip_periods,
            lli=getLLISlipPeriods(self._build_lli_phase_matrix()),
        )
        slips.filter_by_elevation(valid_elevation)

        (
            mean_multipath_range1, overall_mean_multipath_range1,
            rms_multipath_range1, average_rms_multipath_range1,
            mean_ion_delay_phase1, overall_mean_ion_delay_phase1,
            mean_sat_elevation_angles, nEstimates, nEstimates_per_sat,
            nRange1Obs_Per_Sat, nRange1Obs,
            range1_slip_distribution_per_sat, range1_slip_distribution,
            ambiguity_slip_distribution_per_sat, ambiguity_slip_distribution,
            LLI_slip_distribution_per_sat, LLI_slip_distribution,
            combined_slip_distribution_per_sat, combined_slip_distribution,
            elevation_weighted_rms_multipath_range1,
            elevation_weighted_average_rms_multipath_range1,
        ) = computeDelayStats(
            ion_delay_phase1, multipath_range1, elev,
            slips.range1, slips.ambiguity, slips.lli,
            range1_observations, self.t_interval,
        )

        return SignalStats(
            mean_multipath_range1_satellitewise=mean_multipath_range1,
            mean_multipath_range1_overall=overall_mean_multipath_range1,
            rms_multipath_range1_satellitewise=rms_multipath_range1,
            rms_multipath_range1_averaged=average_rms_multipath_range1,
            mean_ion_delay_phase1_satellitewise=mean_ion_delay_phase1,
            mean_ion_delay_phase1_overall=overall_mean_ion_delay_phase1,
            mean_sat_elevation_angles=mean_sat_elevation_angles,
            nEstimates=nEstimates,
            nEstimates_per_sat=nEstimates_per_sat,
            n_range1_obs_per_sat=nRange1Obs_Per_Sat,
            nRange1Obs=nRange1Obs,
            range1_slip_distribution_per_sat=range1_slip_distribution_per_sat,
            range1_slip_distribution=range1_slip_distribution,
            cycle_slip_distribution_per_sat=ambiguity_slip_distribution_per_sat,
            cycle_slip_distribution=ambiguity_slip_distribution,
            LLI_slip_distribution_per_sat=LLI_slip_distribution_per_sat,
            LLI_slip_distribution=LLI_slip_distribution,
            slip_distribution_per_sat_LLI_fusion=combined_slip_distribution_per_sat,
            slip_distribution_LLI_fusion=combined_slip_distribution,
            elevation_weighted_rms_multipath_range1_satellitewise=elevation_weighted_rms_multipath_range1,
            elevation_weighted_average_rms_multipath_range1=elevation_weighted_average_rms_multipath_range1,
            range1_observations=range1_observations,
            phase1_observations=phase1_observations,
            ion_delay_phase1=ion_delay_phase1,
            multipath_range1=multipath_range1,
            sat_elevation_angles=elev,
            range1_Code=self.range1_code,
            range2_Code=self.range2_code,
            phase1_Code=self.phase1_code,
            phase2_Code=self.phase2_code,
            range1_slip_periods=slips.range1,
            cycle_slip_periods=slips.ambiguity,
        ), success
