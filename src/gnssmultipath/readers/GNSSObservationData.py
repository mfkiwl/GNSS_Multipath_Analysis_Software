"""
Accessors for RINEX observation data.

Wraps the raw dict-of-dicts produced by ``readRinexObs`` into typed,
attribute-accessible objects.  All observation codes follow the
**RINEX 3/4 three-character convention**::

    Character 1 – observation type
        C  pseudorange          (code)
        L  carrier phase        (cycles)
        D  Doppler              (Hz)
        S  signal strength      (dB-Hz)

    Character 2 – frequency band
        1  L1 / E1 / B1C        (1575.42 MHz)
        2  L2 / B1I             (1227.60 MHz / 1561.098 MHz)
        5  L5 / E5a / B2a       (1176.45 MHz)
        6  E6 / B3              (1278.75 MHz)
        7  E5b / B2b            (1207.14 MHz)
        8  E5(a+b) / B2(a+b)   (1191.795 MHz)

    Character 3 – tracking mode / attribute
        C  C/A code, civilian
        P  P code
        W  Z-tracking (codeless)
        X  Pilot + data combined
        I  Data channel (Galileo I)
        Q  Pilot channel (Galileo Q)
        …  (see RINEX 3.05 / 4.02 specification for full list)


Usage
-----
::

    from gnssmultipath.readers.readRinexObs import readRinexObs

    rinex = readRinexObs("observation.rnx")
    obs   = rinex.observations          # GNSSObservationData

    # Available systems
    obs.systems                         # ['G', 'E', 'C']

    # Per-system accessor (property or bracket)
    gps = obs.gps                       # SystemObservations
    gal = obs['E']                      # SystemObservations

    # List observation codes
    gps.codes                           # ['C1C', 'L1C', 'S1C', 'C5X', ...]
    gps.pseudorange_codes               # ['C1C', 'C5X']
    gps.phase_codes                     # ['L1C', 'L5X']
    gps.snr_codes                       # ['S1C', 'S5X']
    gps.doppler_codes                   # ['D1C']

    # Extract a specific observation  →  ndarray [epochs, max_sat]
    gps['C1C']                          # pseudorange on L1 C/A
    gps['L5X']                          # carrier phase on L5

    # Filter by frequency band
    gps.band(1)                         # dict {'C1C': arr, 'L1C': arr, 'S1C': arr}
    gps.band(5)                         # dict {'C5X': arr, 'L5X': arr, 'S5X': arr}

    # Check if a code exists
    'C1C' in gps                        # True

    # Number of epochs / satellites
    gps.n_epochs                        # int
    gps.n_satellites                    # int (max PRN + 1)

    # LLI and signal-strength sub-accessors (same interface)
    gps.lli['L1C']                      # ndarray [epochs, max_sat]
    gps.ss['S1C']                       # ndarray [epochs, max_sat]
"""

import numpy as np


# ── Observation-type helpers ──────────────────────────────────────────────────

_OBS_TYPE_NAMES = {
    'C': 'pseudorange',
    'L': 'phase',
    'D': 'doppler',
    'S': 'snr',
}

_BAND_DESCRIPTIONS = {
    '1': 'L1/E1/B1C (1575.42 MHz)',
    '2': 'L2/B1I (1227.60 / 1561.098 MHz)',
    '5': 'L5/E5a/B2a (1176.45 MHz)',
    '6': 'E6/B3 (1278.75 MHz)',
    '7': 'E5b/B2b (1207.14 MHz)',
    '8': 'E5(a+b)/B2(a+b) (1191.795 MHz)',
}

_SYSTEM_NAMES = {
    'G': 'GPS',
    'R': 'GLONASS',
    'E': 'Galileo',
    'C': 'BeiDou',
}


# ── Code-indexed sub-accessor (shared by obs, LLI, SS) ───────────────────────

class _CodeAccessor:
    """Lazily stacks per-epoch dicts and indexes by observation code string."""

    __slots__ = ('_epoch_dict', '_codes', '_stacked')

    def __init__(self, epoch_dict, codes):
        self._epoch_dict = epoch_dict
        self._codes = codes
        self._stacked = None

    def _stack(self):
        if self._stacked is None:
            if self._epoch_dict and isinstance(self._epoch_dict, dict):
                self._stacked = np.stack(list(self._epoch_dict.values()))
            else:
                self._stacked = np.empty((0, 0, len(self._codes)))
        return self._stacked

    def __getitem__(self, code):
        """Return 2-D array [epochs, max_sat] for the given obs code."""
        if code not in self._codes:
            raise KeyError(
                f"Observation code '{code}' not available. "
                f"Available codes: {self._codes}"
            )
        idx = self._codes.index(code)
        return self._stack()[:, :, idx]

    def __contains__(self, code):
        return code in self._codes

    def __repr__(self):
        return f"_CodeAccessor(codes={self._codes})"


# ── Per-system observations ──────────────────────────────────────────────────

class SystemObservations:
    """Observations for a single GNSS system, indexed by obs code.

    Parameters
    ----------
    epoch_dict : dict[int, ndarray]
        Per-epoch observation matrices ``{epoch: array[max_sat, n_codes]}``.
    obs_codes : list[str]
        Ordered 3-char observation codes matching the column axis.
    lli_epoch_dict : dict[int, ndarray] or None
        Per-epoch Loss-of-Lock Indicator matrices (same shape).
    ss_epoch_dict : dict[int, ndarray] or None
        Per-epoch Signal-Strength matrices (same shape).
    system_code : str
        Single-character GNSS system identifier ('G', 'R', 'E', 'C').
    """

    def __init__(self, epoch_dict, obs_codes, lli_epoch_dict=None,
                 ss_epoch_dict=None, system_code=''):
        self._obs = _CodeAccessor(epoch_dict, obs_codes)
        self._obs_codes = list(obs_codes)
        self._system_code = system_code

        self._lli = (_CodeAccessor(lli_epoch_dict, obs_codes)
                     if isinstance(lli_epoch_dict, dict) else None)
        self._ss = (_CodeAccessor(ss_epoch_dict, obs_codes)
                    if isinstance(ss_epoch_dict, dict) else None)

    # ── Code listing ─────────────────────────────────────────────────────

    @property
    def codes(self):
        """All available observation codes for this system."""
        return list(self._obs_codes)

    @property
    def pseudorange_codes(self):
        """Codes starting with 'C' (pseudorange / code observations)."""
        return [c for c in self._obs_codes if c[0] == 'C']

    @property
    def phase_codes(self):
        """Codes starting with 'L' (carrier-phase observations)."""
        return [c for c in self._obs_codes if c[0] == 'L']

    @property
    def doppler_codes(self):
        """Codes starting with 'D' (Doppler observations)."""
        return [c for c in self._obs_codes if c[0] == 'D']

    @property
    def snr_codes(self):
        """Codes starting with 'S' (signal-strength / SNR observations)."""
        return [c for c in self._obs_codes if c[0] == 'S']

    @property
    def bands(self):
        """Set of frequency-band digits present in the observation codes."""
        return sorted({c[1] for c in self._obs_codes})

    # ── Shape info ───────────────────────────────────────────────────────

    @property
    def n_epochs(self):
        """Number of observation epochs."""
        arr = self._obs._stack()
        return arr.shape[0] if arr.size else 0

    @property
    def n_satellites(self):
        """Size of the satellite (PRN) axis (includes unused row 0)."""
        arr = self._obs._stack()
        return arr.shape[1] if arr.size else 0

    @property
    def system_code(self):
        """Single-character GNSS system code ('G', 'R', 'E', 'C')."""
        return self._system_code

    @property
    def system_name(self):
        """Full GNSS system name."""
        return _SYSTEM_NAMES.get(self._system_code, self._system_code)

    # ── Data access ──────────────────────────────────────────────────────

    def __getitem__(self, code):
        """``sys_obs['C1C']`` → ndarray [epochs, max_sat]."""
        return self._obs[code]

    def __contains__(self, code):
        return code in self._obs

    @property
    def data(self):
        """Full 3-D observation array [epochs, max_sat, n_codes]."""
        return self._obs._stack()

    def band(self, band_num):
        """Return a dict of ``{code: array}`` for a specific frequency band.

        Parameters
        ----------
        band_num : int or str
            Frequency-band digit (1, 2, 5, 6, 7, 8).
        """
        b = str(band_num)
        return {c: self._obs[c] for c in self._obs_codes if c[1] == b}

    def by_type(self, obs_type):
        """Return a dict of ``{code: array}`` for a specific observation type.

        Parameters
        ----------
        obs_type : str
            Single character: 'C' (pseudorange), 'L' (phase),
            'D' (Doppler), or 'S' (SNR).
        """
        t = obs_type.upper()
        return {c: self._obs[c] for c in self._obs_codes if c[0] == t}

    # ── LLI / SS sub-accessors ───────────────────────────────────────────

    @property
    def lli(self):
        """Loss-of-Lock Indicator accessor (same ``[code]`` interface)."""
        if self._lli is None:
            raise AttributeError(
                "LLI data not available (readLLI was disabled or data missing)."
            )
        return self._lli

    @property
    def ss(self):
        """Signal-Strength accessor (same ``[code]`` interface)."""
        if self._ss is None:
            raise AttributeError(
                "Signal-Strength data not available (readSS was disabled or data missing)."
            )
        return self._ss

    # ── Representation ───────────────────────────────────────────────────

    def __repr__(self):
        name = _SYSTEM_NAMES.get(self._system_code, self._system_code)
        return (f"SystemObservations({name}, "
                f"codes={self._obs_codes}, "
                f"epochs={self.n_epochs}, sats={self.n_satellites})")


# ── Top-level store ──────────────────────────────────────────────────────────

class GNSSObservationData:
    """Top-level observation container, keyed by GNSS system code.

    Constructed automatically from the raw dicts returned by
    ``readRinexObs``.  Access per-system data via properties
    (``.gps``, ``.glonass``, ``.galileo``, ``.beidou``) or bracket
    notation (``store['G']``).
    """

    def __init__(self, gnss_obs, obs_codes, gnss_systems, gnss_lli=None, gnss_ss=None):
        self._systems = {}

        # Flatten obsCodes: {1: {'G': [...]}, 2: {'R': [...]}} → {'G': [...], 'R': [...]}
        flat_codes = {}
        for sys_idx, sys_code in gnss_systems.items():
            if sys_idx in obs_codes and sys_code in obs_codes[sys_idx]:
                flat_codes[sys_code] = obs_codes[sys_idx][sys_code]

        for sys_idx, sys_code in gnss_systems.items():
            epoch_dict = gnss_obs.get(sys_code, {})
            codes = flat_codes.get(sys_code, [])

            # Skip systems with no data
            if not epoch_dict or not isinstance(epoch_dict, dict):
                continue

            lli_dict = None
            if isinstance(gnss_lli, dict):
                lli_val = gnss_lli.get(sys_code)
                if isinstance(lli_val, dict) and lli_val:
                    lli_dict = lli_val

            ss_dict = None
            if isinstance(gnss_ss, dict):
                ss_val = gnss_ss.get(sys_code)
                if isinstance(ss_val, dict) and ss_val:
                    ss_dict = ss_val

            self._systems[sys_code] = SystemObservations(
                epoch_dict, codes, lli_dict, ss_dict, sys_code
            )

    # ── System listing ───────────────────────────────────────────────────

    @property
    def systems(self):
        """List of available system codes (e.g. ``['G', 'E', 'C']``)."""
        return list(self._systems.keys())

    # ── Bracket access ───────────────────────────────────────────────────

    def __getitem__(self, sys_code):
        """``store['G']`` → SystemObservations for GPS."""
        if sys_code not in self._systems:
            available = list(self._systems.keys())
            names = [_SYSTEM_NAMES.get(s, s) for s in available]
            raise KeyError(
                f"System '{sys_code}' not available. "
                f"Available: {dict(zip(available, names))}"
            )
        return self._systems[sys_code]

    def __contains__(self, sys_code):
        return sys_code in self._systems

    def __iter__(self):
        return iter(self._systems.values())

    def __len__(self):
        return len(self._systems)

    # ── Named properties ─────────────────────────────────────────────────

    @property
    def gps(self):
        """GPS observations."""
        return self['G']

    @property
    def glonass(self):
        """GLONASS observations."""
        return self['R']

    @property
    def galileo(self):
        """Galileo observations."""
        return self['E']

    @property
    def beidou(self):
        """BeiDou observations."""
        return self['C']


    def __repr__(self):
        parts = [repr(s) for s in self._systems.values()]
        return f"GNSSObservationData([\n  " + ",\n  ".join(parts) + "\n])"
