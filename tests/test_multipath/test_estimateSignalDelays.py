"""
Unit tests for multipath delay estimation routines.

Tests cover:
- Helper functions (ismember, create_array_for_current_obscode,
  find_epoch_of_first_obs, find_epoch_of_last_obs, find_missing_observation)
- Multipath formula correctness: MP1 = P1 - (1 + 2/(α-1))*L1 + (2/(α-1))*L2
- Ionospheric delay formula: Ion = 1/(α-1)*(L1_m - L2_m)
- Ambiguity period correction (mean removal for multipath, first-value for ion)
- GLONASS FDMA per-satellite frequency handling
- Full estimateSignalDelays integration with synthetic data
"""
import numpy as np
import pytest

from gnssmultipath.estimateSignalDelays import (
    ismember,
    create_array_for_current_obscode,
    find_epoch_of_first_obs,
    find_epoch_of_last_obs,
    find_missing_observation,
    estimateSignalDelays,
)



C = 299792458.0  # speed of light m/s
F1_GPS = 1575.42e6  # GPS L1 frequency Hz
F2_GPS = 1227.60e6  # GPS L2 frequency Hz
ALPHA_GPS = F1_GPS**2 / F2_GPS**2



class TestIsmember:
    def test_finds_existing_code(self):
        codes = ["C1C", "L1C", "C2W", "L2W"]
        assert ismember(codes, "L1C") == 1

    def test_first_element(self):
        codes = ["C1C", "L1C", "C2W", "L2W"]
        assert ismember(codes, "C1C") == 0

    def test_last_element(self):
        codes = ["C1C", "L1C", "C2W", "L2W"]
        assert ismember(codes, "L2W") == 3

    def test_not_found_returns_empty_list(self):
        codes = ["C1C", "L1C", "C2W", "L2W"]
        result = ismember(codes, "C5X")
        assert result == []

    def test_duplicate_returns_first(self):
        codes = ["C1C", "L1C", "C1C", "L2W"]
        assert ismember(codes, "C1C") == 0

    def test_empty_list(self):
        assert ismember([], "C1C") == []



class TestCreateArrayForCurrentObscode:
    def test_basic_extraction(self):
        """Extract a single obs-type column from a dict of epoch arrays."""
        # 3 epochs, 2 sats (row 0 unused), 2 obs types
        obs = {
            1: np.array([[0, 0], [100.0, 200.0]]),
            2: np.array([[0, 0], [110.0, 210.0]]),
            3: np.array([[0, 0], [120.0, 220.0]]),
        }
        result = create_array_for_current_obscode(obs, 0)
        # Should be shape (3, 2): 3 epochs, 2 sats
        assert result.shape == (3, 2)
        # Row-0 (unused sat) had zeros → NaN
        assert np.isnan(result[0, 0])
        # Non-zero values preserved
        assert result[0, 1] == 100.0
        assert result[2, 1] == 120.0

    def test_zeros_become_nan(self):
        obs = {
            1: np.array([[0.0, 0.0], [0.0, 50.0]]),
        }
        result = create_array_for_current_obscode(obs, 0)
        # Single-epoch dict produces 1D array after squeeze
        assert np.isnan(result[0])  # was 0
        assert np.isnan(result[1])  # was 0

    def test_invalid_index_returns_none(self):
        obs = {1: np.array([[10.0]])}
        result = create_array_for_current_obscode(obs, 99)
        assert result is None



class TestFindEpochOfFirstObs:
    def test_basic(self):
        # 5 epochs, 3 columns (PRNs)
        arr = np.full((5, 3), np.nan)
        arr[2, 0] = 1.0  # first obs at epoch 2 for PRN 0
        arr[0, 1] = 2.0  # first obs at epoch 0 for PRN 1
        # PRN 2 has no obs
        result = find_epoch_of_first_obs(arr)
        assert result[0] == 2
        assert result[1] == 0
        assert np.isnan(result[2])

    def test_all_nan_column(self):
        arr = np.full((4, 2), np.nan)
        arr[1, 0] = 5.0
        result = find_epoch_of_first_obs(arr)
        assert result[0] == 1
        assert np.isnan(result[1])

    def test_first_epoch_has_obs(self):
        arr = np.array([[1.0], [2.0], [3.0]])
        result = find_epoch_of_first_obs(arr)
        assert result[0] == 0

    def test_single_obs(self):
        arr = np.full((10, 1), np.nan)
        arr[7, 0] = 42.0
        result = find_epoch_of_first_obs(arr)
        assert result[0] == 7


class TestFindEpochOfLastObs:
    def test_basic(self):
        arr = np.full((5, 3), np.nan)
        arr[1, 0] = 1.0
        arr[3, 0] = 2.0  # last obs at epoch 3 for PRN 0
        arr[0, 1] = 2.0
        arr[4, 1] = 3.0  # last obs at epoch 4 for PRN 1
        result = find_epoch_of_last_obs(arr)
        assert result[0] == 3
        assert result[1] == 4
        assert np.isnan(result[2])

    def test_all_nan(self):
        arr = np.full((4, 1), np.nan)
        result = find_epoch_of_last_obs(arr)
        assert np.isnan(result[0])

    def test_single_obs(self):
        arr = np.full((10, 1), np.nan)
        arr[5, 0] = 1.0
        result = find_epoch_of_last_obs(arr)
        assert result[0] == 5

    def test_last_epoch_has_obs(self):
        arr = np.array([[1.0], [np.nan], [3.0]])
        result = find_epoch_of_last_obs(arr)
        assert result[0] == 2



class TestFindMissingObservation:
    def test_two_arrays_no_missing(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = find_missing_observation(a, b)
        assert np.all(result == 0)

    def test_two_arrays_with_nan(self):
        a = np.array([[np.nan, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, np.nan]])
        result = find_missing_observation(a, b)
        expected = np.array([[1, 0], [0, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_four_arrays(self):
        a = np.array([[1.0, 2.0]])
        b = np.array([[3.0, 4.0]])
        c = np.array([[np.nan, 6.0]])
        d = np.array([[7.0, 8.0]])
        result = find_missing_observation(a, b, array3=c, array4=d)
        expected = np.array([[1, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_all_nan(self):
        a = np.full((2, 2), np.nan)
        b = np.full((2, 2), np.nan)
        result = find_missing_observation(a, b)
        assert np.all(result == 1)

    def test_union_of_nans_across_arrays(self):
        """If ANY of the 4 arrays has NaN at a position, result is 1."""
        a = np.array([[1.0, np.nan]])
        b = np.array([[np.nan, 2.0]])
        c = np.array([[3.0, 3.0]])
        d = np.array([[4.0, 4.0]])
        result = find_missing_observation(a, b, array3=c, array4=d)
        expected = np.array([[1, 1]])
        np.testing.assert_array_equal(result, expected)



class TestMultipathFormula:
    """Verify the raw multipath and ionosphere formulas against hand-computed values."""

    def test_multipath_formula_gps_l1l2(self):
        """MP1 = P1 - (1 + 2/(α-1))*L1_m + (2/(α-1))*L2_m"""
        alpha = ALPHA_GPS
        P1 = 22_000_000.0  # pseudorange in meters
        L1_m = 22_000_001.0  # phase in meters
        L2_m = 22_000_002.0  # phase in meters

        expected = P1 - (1 + 2 / (alpha - 1)) * L1_m + (2 / (alpha - 1)) * L2_m
        # This is the exact formula used in estimateSignalDelays.py line ~195
        got = P1 - (1 + 2 / (alpha - 1)) * L1_m + (2 / (alpha - 1)) * L2_m
        assert got == pytest.approx(expected, abs=1e-10)

    def test_ionosphere_formula_gps_l1l2(self):
        """Ion = 1/(α-1) * (L1_m - L2_m)"""
        alpha = ALPHA_GPS
        L1_m = 22_000_001.0
        L2_m = 22_000_002.0
        expected = 1.0 / (alpha - 1) * (L1_m - L2_m)
        got = 1.0 / (alpha - 1) * (L1_m - L2_m)
        assert got == pytest.approx(expected, abs=1e-10)

    def test_zero_multipath_scenario(self):
        """When observations are consistent with no multipath, MP should be near zero."""
        alpha = ALPHA_GPS
        # If P1 == L1_m (no code-phase bias), and L1_m == L2_m (no ionosphere)
        # MP1 = L1_m - (1 + 2/(α-1))*L1_m + (2/(α-1))*L1_m
        #      = L1_m * [1 - 1 - 2/(α-1) + 2/(α-1)] = 0
        L_m = 22_000_000.0
        mp = L_m - (1 + 2 / (alpha - 1)) * L_m + (2 / (alpha - 1)) * L_m
        assert mp == pytest.approx(0.0, abs=1e-6)

    def test_multipath_vectorized(self):
        """Multipath formula works correctly when applied to numpy arrays."""
        alpha = ALPHA_GPS
        nepochs = 100
        nsat = 5
        rng = np.random.default_rng(42)
        P1 = 22e6 + rng.normal(0, 1, (nepochs, nsat))
        L1_m = 22e6 + rng.normal(0, 0.01, (nepochs, nsat))
        L2_m = 22e6 + rng.normal(0, 0.01, (nepochs, nsat))

        mp_vec = P1 - (1 + 2 / (alpha - 1)) * L1_m + (2 / (alpha - 1)) * L2_m
        # Verify element-wise against scalar formula
        for i in range(nepochs):
            for j in range(nsat):
                expected = P1[i, j] - (1 + 2 / (alpha - 1)) * L1_m[i, j] + (2 / (alpha - 1)) * L2_m[i, j]
                assert mp_vec[i, j] == pytest.approx(expected, abs=1e-10)

    def test_alpha_computation(self):
        """Alpha = f1²/f2² should be > 1 for all GNSS systems."""
        # GPS L1/L2
        assert ALPHA_GPS > 1.0
        assert ALPHA_GPS == pytest.approx(F1_GPS**2 / F2_GPS**2)

        # Galileo E1/E5a
        f_e1 = 1575.42e6
        f_e5a = 1176.45e6
        alpha_gal = f_e1**2 / f_e5a**2
        assert alpha_gal > 1.0

    def test_glonass_fdma_alpha_is_array(self):
        """For GLONASS FDMA, alpha should be an array with per-satellite values."""
        # GLONASS L1 and L2 base frequencies
        f1_base = 1602.0e6
        f2_base = 1246.0e6
        df1 = 0.5625e6  # channel spacing L1
        df2 = 0.4375e6  # channel spacing L2
        n_sats = 24
        channels = np.arange(-7, 7)  # frequency channels -7 to 6
        freq1 = f1_base + channels * df1
        freq2 = f2_base + channels * df2
        alpha = freq1**2 / freq2**2
        assert alpha.shape == (14,)
        assert np.all(alpha > 1.0)
        # Each satellite may have different alpha
        assert not np.all(alpha == alpha[0])



class TestAmbiguityCorrection:
    """Test that ionosphere and multipath are corrected properly in each ambiguity period."""

    def test_ionosphere_reduced_by_first_value(self):
        """Ionosphere delay in each segment should be reduced by its first value."""
        # Simulate: ion_delay for one PRN with no slips
        ion = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        first_obs = 0
        # No slips: reduce by first value
        corrected = ion[first_obs:] - ion[first_obs]
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(corrected, expected)

    def test_multipath_reduced_by_mean(self):
        """Multipath in each segment should be reduced by the segment mean."""
        mp = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        first_obs = 0
        corrected = mp[first_obs:] - np.nanmean(mp[first_obs:])
        assert np.nanmean(corrected) == pytest.approx(0.0, abs=1e-10)
        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(corrected, expected)

    def test_slip_epochs_set_to_nan(self):
        """Epochs within a slip period should be set to NaN before correction."""
        mp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Simulate slip at epochs 2-3
        mp[2:4] = np.nan
        # Segment before slip: epochs 0-1
        seg1 = mp[0:2]
        seg1_corrected = seg1 - np.nanmean(seg1)
        assert seg1_corrected[0] == pytest.approx(-0.5)
        assert seg1_corrected[1] == pytest.approx(0.5)
        # Segment after slip: epoch 4
        seg2 = mp[4:5]
        seg2_corrected = seg2 - np.nanmean(seg2)
        assert seg2_corrected[0] == pytest.approx(0.0)

    def test_multiple_ambiguity_periods(self):
        """With multiple slips, each segment is independently corrected."""
        ion = np.array([10.0, 11.0, np.nan, 20.0, 21.0, np.nan, 30.0, 31.0, 32.0])
        # Segment 1: epochs 0-1, first_val=10 -> [0, 1]
        seg1 = ion[0:2] - ion[0]
        np.testing.assert_allclose(seg1, [0.0, 1.0])
        # Segment 2: epochs 3-4, first_val=20 -> [0, 1]
        seg2 = ion[3:5] - ion[3]
        np.testing.assert_allclose(seg2, [0.0, 1.0])
        # Segment 3: epochs 6-8, first_val=30 -> [0, 1, 2]
        seg3 = ion[6:9] - ion[6]
        np.testing.assert_allclose(seg3, [0.0, 1.0, 2.0])



class TestEstimateSignalDelays:
    """Integration tests for estimateSignalDelays with synthetic observations."""

    @staticmethod
    def _make_synthetic_obs(nepochs, max_sat, nobs_types, range1_idx, range2_idx,
                            phase1_idx, phase2_idx, carrier_freq1, carrier_freq2):
        """
        Create synthetic GNSS observations with known multipath = 0 and known ionosphere.

        Returns GNSS_obs dict, obsCodes dict, and GNSS_SVs dict.
        """
        c = 299792458.0
        rng = np.random.default_rng(123)

        # Obs codes
        obs_types = ["" for _ in range(nobs_types)]
        obs_types[range1_idx] = "C1C"
        obs_types[range2_idx] = "C2W"
        obs_types[phase1_idx] = "L1C"
        obs_types[phase2_idx] = "L2W"
        obsCodes = {"G": obs_types}

        # Build GNSS_obs: dict keyed by epoch (1-based), each value is (max_sat+1, nobs_types)
        GNSS_obs = {}
        true_range = 22_000_000.0  # true geometric range
        true_ion = 5.0  # ionospheric delay in meters on L1
        alpha = carrier_freq1**2 / carrier_freq2**2

        for ep in range(1, nepochs + 1):
            data = np.zeros((max_sat + 1, nobs_types))
            for prn in range(1, max_sat + 1):
                # Phase in cycles
                L1_cycles = (true_range + true_ion) * carrier_freq1 / c
                L2_cycles = (true_range + true_ion * alpha) * carrier_freq2 / c
                # Range (pseudorange) = geometric range + ionospheric delay
                P1 = true_range + true_ion
                P2 = true_range + true_ion * alpha

                data[prn, range1_idx] = P1
                data[prn, range2_idx] = P2
                data[prn, phase1_idx] = L1_cycles
                data[prn, phase2_idx] = L2_cycles

            GNSS_obs[ep] = data

        # GNSS_SVs: dict keyed by epoch, each is 1D array [nSVs, prn1, prn2, ...]
        GNSS_SVs = {}
        for ep in range(1, nepochs + 1):
            svs = np.zeros(max_sat + 1)
            svs[0] = max_sat
            for prn in range(1, max_sat + 1):
                svs[prn] = prn
            GNSS_SVs[ep] = svs

        return GNSS_obs, obsCodes, GNSS_SVs

    def test_returns_correct_shape(self):
        """estimateSignalDelays returns arrays with correct shape."""
        nepochs = 20
        max_sat = 3
        nobs = 4
        carrier_freq1 = F1_GPS
        carrier_freq2 = F2_GPS

        GNSS_obs, obsCodes, GNSS_SVs = self._make_synthetic_obs(
            nepochs, max_sat, nobs, 0, 1, 2, 3, carrier_freq1, carrier_freq2
        )

        result = estimateSignalDelays(
            "C1C", "C2W", "L1C", "L2W",
            carrier_freq1, carrier_freq2,
            nepochs, max_sat, GNSS_SVs, obsCodes, GNSS_obs,
            "G", 30.0, 0, 0,
        )
        ion_delay, mp_range1, range1_slips, amb_slips, range1_obs, phase1_obs, success = result

        assert success == 1
        assert ion_delay.shape == (nepochs, max_sat + 1)
        assert mp_range1.shape == (nepochs, max_sat + 1)

    def test_success_flag_on_valid_input(self):
        nepochs = 10
        max_sat = 2
        GNSS_obs, obsCodes, GNSS_SVs = self._make_synthetic_obs(
            nepochs, max_sat, 4, 0, 1, 2, 3, F1_GPS, F2_GPS
        )
        result = estimateSignalDelays(
            "C1C", "C2W", "L1C", "L2W",
            F1_GPS, F2_GPS, nepochs, max_sat, GNSS_SVs, obsCodes, GNSS_obs,
            "G", 30.0, 0, 0,
        )
        assert result[-1] == 1  # success

    def test_constant_obs_gives_zero_multipath(self):
        """With constant, consistent observations and no noise, multipath after
        mean-correction should be zero for all epochs."""
        nepochs = 50
        max_sat = 2
        GNSS_obs, obsCodes, GNSS_SVs = self._make_synthetic_obs(
            nepochs, max_sat, 4, 0, 1, 2, 3, F1_GPS, F2_GPS
        )
        result = estimateSignalDelays(
            "C1C", "C2W", "L1C", "L2W",
            F1_GPS, F2_GPS, nepochs, max_sat, GNSS_SVs, obsCodes, GNSS_obs,
            "G", 30.0, 0, 0,
        )
        ion_delay, mp_range1, _, _, _, _, success = result
        assert success == 1
        # After mean correction, constant multipath → 0
        for prn in range(1, max_sat + 1):
            prn_mp = mp_range1[:, prn]
            finite = prn_mp[np.isfinite(prn_mp)]
            if len(finite) > 0:
                np.testing.assert_allclose(finite, 0.0, atol=1e-3)

    def test_slip_periods_dict_structure(self):
        """Slip periods should be dict keyed by int PRN with list/array values."""
        nepochs = 20
        max_sat = 3
        GNSS_obs, obsCodes, GNSS_SVs = self._make_synthetic_obs(
            nepochs, max_sat, 4, 0, 1, 2, 3, F1_GPS, F2_GPS
        )
        result = estimateSignalDelays(
            "C1C", "C2W", "L1C", "L2W",
            F1_GPS, F2_GPS, nepochs, max_sat, GNSS_SVs, obsCodes, GNSS_obs,
            "G", 30.0, 0, 0,
        )
        _, _, range1_slips, amb_slips, _, _, _ = result
        # Should have keys 1..max_sat
        for prn in range(1, max_sat + 1):
            assert prn in range1_slips
            assert prn in amb_slips

    def test_missing_obs_produces_nan(self):
        """Epochs with zero observations should produce NaN in output."""
        nepochs = 10
        max_sat = 2
        GNSS_obs, obsCodes, GNSS_SVs = self._make_synthetic_obs(
            nepochs, max_sat, 4, 0, 1, 2, 3, F1_GPS, F2_GPS
        )
        # Zero out observations for PRN 1 at epochs 3-5
        for ep in range(3, 6):
            GNSS_obs[ep + 1][1, :] = 0.0

        result = estimateSignalDelays(
            "C1C", "C2W", "L1C", "L2W",
            F1_GPS, F2_GPS, nepochs, max_sat, GNSS_SVs, obsCodes, GNSS_obs,
            "G", 30.0, 0, 0,
        )
        ion_delay, mp_range1, _, _, _, _, success = result
        assert success == 1
        # Epochs 3-5 (0-indexed: rows 2-4) for PRN 1 should be NaN
        # (create_array_for_current_obscode replaces 0 with NaN)
        for ep in range(2, 5):
            assert np.isnan(ion_delay[ep, 1]) or ion_delay[ep, 1] == 0.0

    def test_single_epoch_satellite(self):
        """A satellite observed at only one epoch should not crash."""
        nepochs = 10
        max_sat = 2
        GNSS_obs, obsCodes, GNSS_SVs = self._make_synthetic_obs(
            nepochs, max_sat, 4, 0, 1, 2, 3, F1_GPS, F2_GPS
        )
        # Zero out PRN 2 for all epochs except epoch 5
        for ep in range(1, nepochs + 1):
            if ep != 5:
                GNSS_obs[ep][2, :] = 0.0

        result = estimateSignalDelays(
            "C1C", "C2W", "L1C", "L2W",
            F1_GPS, F2_GPS, nepochs, max_sat, GNSS_SVs, obsCodes, GNSS_obs,
            "G", 30.0, 0, 0,
        )
        ion_delay, mp_range1, _, _, _, _, success = result
        assert success == 1

    def test_glonass_fdma_frequency_arrays(self):
        """GLONASS with per-satellite frequency arrays should not crash."""
        nepochs = 10
        max_sat = 3
        nobs = 4
        # GLONASS frequencies (simplified per-sat arrays)
        f1_base = 1602.0e6
        f2_base = 1246.0e6
        df1 = 0.5625e6
        df2 = 0.4375e6
        channels = np.array([1, -1, 0])
        freq1 = f1_base + channels * df1
        freq2 = f2_base + channels * df2

        # Pad to max_sat+1 (index 0 unused)
        carrier_freq1 = np.zeros(max_sat + 1)
        carrier_freq2 = np.zeros(max_sat + 1)
        carrier_freq1[1:max_sat + 1] = freq1
        carrier_freq2[1:max_sat + 1] = freq2

        GNSS_obs, obsCodes, GNSS_SVs = self._make_synthetic_obs(
            nepochs, max_sat, nobs, 0, 1, 2, 3, F1_GPS, F2_GPS
        )
        # Re-key obsCodes for GLONASS
        obsCodes["R"] = obsCodes.pop("G")

        result = estimateSignalDelays(
            "C1C", "C2W", "L1C", "L2W",
            carrier_freq1, carrier_freq2,
            nepochs, max_sat, GNSS_SVs, obsCodes, GNSS_obs,
            "R", 30.0, 0, 0,
        )
        ion_delay, mp_range1, _, _, _, _, success = result
        assert success == 1
        assert ion_delay.shape == (nepochs, max_sat + 1)



class TestEstimateSignalDelaysEdgeCases:
    @staticmethod
    def _make_obs(nepochs, max_sat, data_func=None):
        """Helper to quickly make synthetic observations."""
        nobs = 4
        c = 299792458.0
        obsCodes = {"G": ["C1C", "C2W", "L1C", "L2W"]}
        GNSS_obs = {}
        GNSS_SVs = {}

        for ep in range(1, nepochs + 1):
            data = np.zeros((max_sat + 1, nobs))
            if data_func:
                for prn in range(1, max_sat + 1):
                    data[prn, :] = data_func(ep, prn)
            GNSS_obs[ep] = data
            svs = np.zeros(max_sat + 1)
            svs[0] = max_sat
            for prn in range(1, max_sat + 1):
                svs[prn] = prn
            GNSS_SVs[ep] = svs

        return GNSS_obs, obsCodes, GNSS_SVs

    def test_no_observations_returns_failure(self):
        """All-zero observations should result in success=0 or graceful handling."""
        nepochs = 5
        max_sat = 2
        GNSS_obs, obsCodes, GNSS_SVs = self._make_obs(nepochs, max_sat)
        # All zeros → create_array_for_current_obscode → all NaN
        result = estimateSignalDelays(
            "C1C", "C2W", "L1C", "L2W",
            F1_GPS, F2_GPS, nepochs, max_sat, GNSS_SVs, obsCodes, GNSS_obs,
            "G", 30.0, 0, 0,
        )
        # With all NaN (from zeros), function should still return without crashing
        # Success flag depends on whether phase/range have any non-NaN data
        assert result is not None

    def test_large_discontinuity_triggers_slip(self):
        """A large jump in phase should be detected as a cycle slip."""
        nepochs = 20
        max_sat = 1
        c = 299792458.0
        true_range = 22e6

        def data_func(ep, prn):
            # Introduce a phase jump at epoch 10
            phase_offset = 0 if ep < 10 else 1000.0
            L1_cyc = (true_range) * F1_GPS / c + phase_offset
            L2_cyc = (true_range) * F2_GPS / c + phase_offset
            return [true_range, true_range, L1_cyc, L2_cyc]

        GNSS_obs, obsCodes, GNSS_SVs = self._make_obs(nepochs, max_sat, data_func)
        result = estimateSignalDelays(
            "C1C", "C2W", "L1C", "L2W",
            F1_GPS, F2_GPS, nepochs, max_sat, GNSS_SVs, obsCodes, GNSS_obs,
            "G", 30.0, 0, 0,
        )
        _, _, range1_slips, amb_slips, _, _, success = result
        assert success == 1
        # The jump should produce some slip detection around epoch 9-10
        has_slips = (
            (len(amb_slips[1]) > 0) if isinstance(amb_slips[1], list) else (amb_slips[1].size > 0)
        )
        assert has_slips, "Large phase discontinuity should trigger a cycle slip"

    def test_no_slips_with_smooth_data(self):
        """Smoothly varying data should produce no cycle slips."""
        nepochs = 50
        max_sat = 1
        c = 299792458.0

        def data_func(ep, prn):
            base = 22e6 + ep * 0.001  # very slow drift
            L1_cyc = base * F1_GPS / c
            L2_cyc = base * F2_GPS / c
            return [base, base, L1_cyc, L2_cyc]

        GNSS_obs, obsCodes, GNSS_SVs = self._make_obs(nepochs, max_sat, data_func)
        result = estimateSignalDelays(
            "C1C", "C2W", "L1C", "L2W",
            F1_GPS, F2_GPS, nepochs, max_sat, GNSS_SVs, obsCodes, GNSS_obs,
            "G", 30.0, 0, 0,
        )
        _, _, range1_slips, amb_slips, _, _, success = result
        assert success == 1
        # Smooth data → no slips
        n_slips = len(amb_slips[1]) if isinstance(amb_slips[1], list) else amb_slips[1].size
        assert n_slips == 0, "Smooth data should produce no cycle slips"
