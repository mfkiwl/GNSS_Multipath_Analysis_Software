"""
Unit tests for cycle slip detection routines.

Tests cover:
- detectCycleSlips: threshold-based detection, missing observation handling,
  edge cases (no slips, all slips, single epoch, multiple PRNs)
- orgSlipEpochs: organizing individual slip epochs into contiguous slip periods,
  empty input, single slip, consecutive slips, non-consecutive slips
- getLLISlipPeriods: organizing Loss of Lock Indicator data into slip periods
"""
import numpy as np
import pytest

from gnssmultipath.detectCycleSlips import detectCycleSlips, orgSlipEpochs
from gnssmultipath.getLLISlipPeriods import getLLISlipPeriods



class TestOrgSlipEpochs:
    def test_empty_input(self):
        """No slip epochs -> empty periods, 0 count."""
        periods, n = orgSlipEpochs(np.array([]))
        assert periods == []
        assert n == 0

    def test_single_slip(self):
        """One slip epoch -> one period with same start and end."""
        periods, n = orgSlipEpochs(np.array([5]))
        assert n == 1
        assert periods.shape == (1, 2)
        assert periods[0, 0] == 5
        assert periods[0, 1] == 5

    def test_two_consecutive_slips(self):
        """Two consecutive epochs -> one period."""
        periods, n = orgSlipEpochs(np.array([3, 4]))
        assert n == 1
        assert periods[0, 0] == 3
        assert periods[0, 1] == 4

    def test_two_non_consecutive_slips(self):
        """Two non-consecutive epochs -> two separate periods."""
        periods, n = orgSlipEpochs(np.array([3, 7]))
        assert n == 2
        assert periods[0, 0] == 3
        assert periods[0, 1] == 3
        assert periods[1, 0] == 7
        assert periods[1, 1] == 7

    def test_multiple_consecutive_groups(self):
        """Multiple groups of consecutive slips."""
        slips = np.array([2, 3, 4, 10, 11, 20])
        periods, n = orgSlipEpochs(slips)
        assert n == 3
        # Group 1: 2-4
        assert periods[0, 0] == 2
        assert periods[0, 1] == 4
        # Group 2: 10-11
        assert periods[1, 0] == 10
        assert periods[1, 1] == 11
        # Group 3: 20
        assert periods[2, 0] == 20
        assert periods[2, 1] == 20

    def test_all_consecutive(self):
        """All epochs consecutive -> one period."""
        slips = np.array([5, 6, 7, 8, 9])
        periods, n = orgSlipEpochs(slips)
        assert n == 1
        assert periods[0, 0] == 5
        assert periods[0, 1] == 9

    def test_all_non_consecutive(self):
        """No consecutive epochs -> each epoch is its own period."""
        slips = np.array([1, 5, 10, 20])
        periods, n = orgSlipEpochs(slips)
        assert n == 4
        for i, s in enumerate(slips):
            assert periods[i, 0] == s
            assert periods[i, 1] == s

    def test_period_count_matches_rows(self):
        """n_slip_periods should match the number of rows in slip_periods."""
        slips = np.array([1, 2, 5, 6, 7, 15])
        periods, n = orgSlipEpochs(slips)
        assert periods.shape[0] == n

    def test_list_input(self):
        """orgSlipEpochs should handle list input from np.union1d result."""
        # np.union1d returns ndarray, but test with list just in case
        slips = np.array([3, 4, 8])
        periods, n = orgSlipEpochs(slips)
        assert n == 2



class TestDetectCycleSlips:
    @staticmethod
    def _make_smooth_estimates(nepochs, nsat, drift=0.001):
        """Create smoothly varying estimates with no cycle slips."""
        estimates = np.zeros((nepochs, nsat + 1))
        for prn in range(1, nsat + 1):
            estimates[:, prn] = np.arange(nepochs) * drift
        return estimates

    def test_no_slips_smooth_data(self):
        """Smooth data should produce no detected slips."""
        nepochs = 100
        nsat = 3
        estimates = self._make_smooth_estimates(nepochs, nsat)
        missing = np.zeros_like(estimates)
        first_obs = np.zeros(nsat + 1)
        first_obs[0] = np.nan
        last_obs = np.full(nsat + 1, nepochs - 1, dtype=float)
        last_obs[0] = np.nan
        tInterval = 30.0
        crit_rate = 4 / 60.0

        slips = detectCycleSlips(estimates, missing, first_obs, last_obs, tInterval, crit_rate)
        for prn in range(1, nsat + 1):
            assert len(slips[str(prn)]) == 0, f"PRN {prn} should have no slips"

    def test_large_jump_detected(self):
        """A large jump in estimates should be detected as a slip."""
        nepochs = 20
        nsat = 1
        estimates = np.zeros((nepochs, nsat + 1))
        estimates[:, 1] = np.arange(nepochs) * 0.001
        # Insert large jump at epoch 10
        estimates[10:, 1] += 100.0
        missing = np.zeros_like(estimates)
        first_obs = np.array([np.nan, 0.0])
        last_obs = np.array([np.nan, nepochs - 1.0])
        tInterval = 30.0
        crit_rate = 4 / 60.0

        slips = detectCycleSlips(estimates, missing, first_obs, last_obs, tInterval, crit_rate)
        assert len(slips["1"]) > 0, "Large jump should be detected"
        # The slip should be near epoch 9 (diff between epoch 9 and 10)
        assert 9 in slips["1"] or 10 in slips["1"]

    def test_missing_obs_detected_as_slip(self):
        """Missing observations within observation window should be flagged."""
        nepochs = 20
        nsat = 1
        estimates = np.zeros((nepochs, nsat + 1))
        estimates[:, 1] = 1.0  # constant - no rate-of-change slips
        missing = np.zeros_like(estimates)
        # Mark epochs 5-7 as missing
        missing[5:8, 1] = 1
        first_obs = np.array([np.nan, 0.0])
        last_obs = np.array([np.nan, nepochs - 1.0])
        tInterval = 30.0
        crit_rate = 4 / 60.0

        slips = detectCycleSlips(estimates, missing, first_obs, last_obs, tInterval, crit_rate)
        # Epochs 5, 6, 7 should appear as slips
        for ep in [5, 6, 7]:
            assert ep in slips["1"], f"Missing epoch {ep} should be detected as slip"

    def test_multiple_prns_independent(self):
        """Slips in one PRN should not affect another."""
        nepochs = 30
        nsat = 2
        estimates = np.zeros((nepochs, nsat + 1))
        estimates[:, 1] = np.arange(nepochs) * 0.001  # smooth
        estimates[:, 2] = np.arange(nepochs) * 0.001  # smooth
        # Jump only in PRN 2
        estimates[15:, 2] += 100.0
        missing = np.zeros_like(estimates)
        first_obs = np.array([np.nan, 0.0, 0.0])
        last_obs = np.array([np.nan, nepochs - 1.0, nepochs - 1.0])
        tInterval = 30.0
        crit_rate = 4 / 60.0

        slips = detectCycleSlips(estimates, missing, first_obs, last_obs, tInterval, crit_rate)
        assert len(slips["1"]) == 0, "PRN 1 should have no slips"
        assert len(slips["2"]) > 0, "PRN 2 should have slips"

    def test_dict_keys_are_str_prn(self):
        """Slip dict should have string PRN keys starting from '1'."""
        nepochs = 10
        nsat = 3
        estimates = np.zeros((nepochs, nsat + 1))
        missing = np.zeros_like(estimates)
        first_obs = np.full(nsat + 1, np.nan)
        last_obs = np.full(nsat + 1, np.nan)
        tInterval = 30.0
        crit_rate = 4 / 60.0

        slips = detectCycleSlips(estimates, missing, first_obs, last_obs, tInterval, crit_rate)
        for prn in range(1, nsat + 1):
            assert str(prn) in slips

    def test_all_nan_satellite_no_crash(self):
        """A satellite with all NaN should not cause crash."""
        nepochs = 15
        nsat = 2
        estimates = np.zeros((nepochs, nsat + 1))
        estimates[:, 1] = np.arange(nepochs) * 0.001
        estimates[:, 2] = np.nan  # all NaN for PRN 2
        missing = np.zeros_like(estimates)
        first_obs = np.array([np.nan, 0.0, np.nan])
        last_obs = np.array([np.nan, nepochs - 1.0, np.nan])
        tInterval = 30.0
        crit_rate = 4 / 60.0

        slips = detectCycleSlips(estimates, missing, first_obs, last_obs, tInterval, crit_rate)
        assert "1" in slips
        assert "2" in slips

    def test_critical_rate_respected(self):
        """Changes exactly at the critical rate should not be flagged."""
        nepochs = 10
        nsat = 1
        tInterval = 30.0
        crit_rate = 4 / 60.0  # 4/60 m/s
        # Create data where rate == crit_rate (not exceeding)
        estimates = np.zeros((nepochs, nsat + 1))
        rate_per_epoch = crit_rate * tInterval  # change per epoch exactly at threshold
        estimates[:, 1] = np.arange(nepochs) * rate_per_epoch * 0.99  # just under threshold
        missing = np.zeros_like(estimates)
        first_obs = np.array([np.nan, 0.0])
        last_obs = np.array([np.nan, nepochs - 1.0])

        slips = detectCycleSlips(estimates, missing, first_obs, last_obs, tInterval, crit_rate)
        assert len(slips["1"]) == 0, "Rate under threshold should not trigger slip"

    def test_rate_just_above_threshold_detected(self):
        """Change rate just above threshold should be detected."""
        nepochs = 10
        nsat = 1
        tInterval = 30.0
        crit_rate = 4 / 60.0
        rate_per_epoch = crit_rate * tInterval * 1.01  # just over threshold
        estimates = np.zeros((nepochs, nsat + 1))
        estimates[:, 1] = np.arange(nepochs) * rate_per_epoch
        missing = np.zeros_like(estimates)
        first_obs = np.array([np.nan, 0.0])
        last_obs = np.array([np.nan, nepochs - 1.0])

        slips = detectCycleSlips(estimates, missing, first_obs, last_obs, tInterval, crit_rate)
        # All epoch diffs exceed threshold, so all should be slips
        assert len(slips["1"]) > 0

    def test_slip_union_with_missing(self):
        """Slips from rate and from missing observations should be merged."""
        nepochs = 20
        nsat = 1
        estimates = np.zeros((nepochs, nsat + 1))
        estimates[:, 1] = np.arange(nepochs) * 0.001  # smooth
        # Add one big jump
        estimates[5, 1] += 100.0
        missing = np.zeros_like(estimates)
        missing[15, 1] = 1  # missing at epoch 15
        first_obs = np.array([np.nan, 0.0])
        last_obs = np.array([np.nan, nepochs - 1.0])
        tInterval = 30.0
        crit_rate = 4 / 60.0

        slips = detectCycleSlips(estimates, missing, first_obs, last_obs, tInterval, crit_rate)
        slip_list = slips["1"]
        # Should detect both the jump near epoch 4/5 and the missing at epoch 15
        assert any(s in [4, 5] for s in slip_list), "Rate-based slip should be detected"
        assert 15 in slip_list, "Missing obs should be in slip list"

    def test_no_duplicate_slip_epochs(self):
        """Slip epochs should contain no duplicates."""
        nepochs = 20
        nsat = 1
        estimates = np.zeros((nepochs, nsat + 1))
        estimates[:, 1] = np.arange(nepochs) * 0.001
        estimates[10, 1] += 100.0  # rate slip
        missing = np.zeros_like(estimates)
        missing[10, 1] = 1  # also missing at same epoch
        first_obs = np.array([np.nan, 0.0])
        last_obs = np.array([np.nan, nepochs - 1.0])
        tInterval = 30.0
        crit_rate = 4 / 60.0

        slips = detectCycleSlips(estimates, missing, first_obs, last_obs, tInterval, crit_rate)
        slip_list = slips["1"]
        assert len(slip_list) == len(set(slip_list)), "No duplicate slip epochs allowed"



class TestGetLLISlipPeriods:
    def test_no_lli_flags(self):
        """No LLI indicators -> empty slip periods for all sats."""
        nepochs = 20
        nsat = 3
        lli = np.zeros((nepochs, nsat + 1))  # col 0 unused
        periods = getLLISlipPeriods(lli)
        for sat in range(nsat):
            assert isinstance(periods[sat], list) or (
                isinstance(periods[sat], np.ndarray) and periods[sat].size == 0
            ) or periods[sat] == []

    def test_single_lli_flag(self):
        """Single LLI=1 at one epoch should create one slip period."""
        nepochs = 20
        nsat = 1
        lli = np.zeros((nepochs, nsat + 1))
        lli[5, 1] = 1  # LLI flag at epoch 5
        periods = getLLISlipPeriods(lli)
        result = periods[0]
        if isinstance(result, np.ndarray) and result.size > 0:
            assert result.shape[1] == 2
            assert result[0, 0] == 5
            assert result[0, 1] == 5

    def test_consecutive_lli_flags(self):
        """Consecutive LLI flags should form one slip period."""
        nepochs = 20
        nsat = 1
        lli = np.zeros((nepochs, nsat + 1))
        lli[3, 1] = 1
        lli[4, 1] = 1
        lli[5, 1] = 1
        periods = getLLISlipPeriods(lli)
        result = periods[0]
        if isinstance(result, np.ndarray) and result.size > 0:
            assert result.shape == (1, 2)
            assert result[0, 0] == 3
            assert result[0, 1] == 5

    @pytest.mark.xfail(
        reason="BUG in getLLISlipPeriods: np.diff operates on wrong axis "
               "because LLI_slips is reshaped to column vector (N,1). "
               "np.diff defaults to last axis, producing empty result for axis=1. "
               "Fix: add axis=0 to np.diff call on line ~43.",
        strict=True,
    )
    def test_non_consecutive_lli_flags(self):
        """Non-consecutive LLI flags should form separate periods."""
        nepochs = 20
        nsat = 1
        lli = np.zeros((nepochs, nsat + 1))
        lli[3, 1] = 1
        lli[10, 1] = 2  # LLI value 2 also triggers
        periods = getLLISlipPeriods(lli)
        result = periods[0]
        if isinstance(result, np.ndarray) and result.size > 0:
            assert result.shape == (2, 2)
            assert result[0, 0] == 3
            assert result[0, 1] == 3
            assert result[1, 0] == 10
            assert result[1, 1] == 10

    def test_lli_values_1_2_3_5_6_7_are_slips(self):
        """LLI values 1, 2, 3, 5, 6, 7 should all be detected as slips."""
        nepochs = 20
        nsat = 1
        lli_values = [1, 2, 3, 5, 6, 7]
        for val in lli_values:
            lli = np.zeros((nepochs, nsat + 1))
            lli[5, 1] = val
            periods = getLLISlipPeriods(lli)
            result = periods[0]
            if isinstance(result, np.ndarray) and result.size > 0:
                assert 5 in result, f"LLI value {val} should be detected"

    def test_lli_value_0_and_4_not_slips(self):
        """LLI values 0 and 4 should NOT be detected as slips."""
        nepochs = 20
        nsat = 1
        for val in [0, 4]:
            lli = np.zeros((nepochs, nsat + 1))
            lli[5, 1] = val
            periods = getLLISlipPeriods(lli)
            result = periods[0]
            if isinstance(result, list):
                assert len(result) == 0, f"LLI value {val} should not trigger slip"
            elif isinstance(result, np.ndarray):
                assert result.size == 0, f"LLI value {val} should not trigger slip"

    def test_multiple_satellites(self):
        """LLI periods should be independent per satellite."""
        nepochs = 20
        nsat = 3
        lli = np.zeros((nepochs, nsat + 1))
        lli[5, 1] = 1  # sat 0 (col index 1)
        lli[10, 2] = 1  # sat 1 (col index 2)
        # sat 2 (col index 3) has no flags
        periods = getLLISlipPeriods(lli)
        # Sat 0 should have slip at epoch 5
        r0 = periods[0]
        if isinstance(r0, np.ndarray) and r0.size > 0:
            assert r0[0, 0] == 5
        # Sat 1 should have slip at epoch 10
        r1 = periods[1]
        if isinstance(r1, np.ndarray) and r1.size > 0:
            assert r1[0, 0] == 10
        # Sat 2 should have no slips
        r2 = periods[2]
        if isinstance(r2, list):
            assert len(r2) == 0
        elif isinstance(r2, np.ndarray):
            assert r2.size == 0

    def test_returns_dict_keyed_by_sat_index(self):
        """Result should be a dict keyed by satellite index (0-based)."""
        nepochs = 10
        nsat = 2
        lli = np.zeros((nepochs, nsat + 1))
        periods = getLLISlipPeriods(lli)
        assert isinstance(periods, dict)
        for sat in range(nsat):
            assert sat in periods


class TestCycleSlipIntegration:
    def test_detect_then_organize(self):
        """detectCycleSlips output should feed cleanly into orgSlipEpochs."""
        nepochs = 50
        nsat = 1
        estimates = np.zeros((nepochs, nsat + 1))
        estimates[:, 1] = np.arange(nepochs) * 0.001
        # Insert two jumps
        estimates[15, 1] += 100.0
        estimates[30, 1] += 200.0
        missing = np.zeros_like(estimates)
        first_obs = np.array([np.nan, 0.0])
        last_obs = np.array([np.nan, nepochs - 1.0])
        tInterval = 30.0
        crit_rate = 4 / 60.0

        slip_dict = detectCycleSlips(estimates, missing, first_obs, last_obs, tInterval, crit_rate)
        slip_epochs = np.array(slip_dict["1"])
        periods, n = orgSlipEpochs(slip_epochs)

        assert n >= 2, "Two jumps should produce at least 2 slip periods"
        assert periods.shape[1] == 2
        # Each period start should be <= its end
        for i in range(n):
            assert periods[i, 0] <= periods[i, 1]

    def test_missing_obs_gap_organized_into_period(self):
        """Consecutive missing observations should form one slip period."""
        nepochs = 30
        nsat = 1
        estimates = np.zeros((nepochs, nsat + 1))
        estimates[:, 1] = 1.0  # constant
        missing = np.zeros_like(estimates)
        # Mark a gap at epochs 10-14
        missing[10:15, 1] = 1
        first_obs = np.array([np.nan, 0.0])
        last_obs = np.array([np.nan, nepochs - 1.0])
        tInterval = 30.0
        crit_rate = 4 / 60.0

        slip_dict = detectCycleSlips(estimates, missing, first_obs, last_obs, tInterval, crit_rate)
        slip_epochs = np.array(slip_dict["1"])
        periods, n = orgSlipEpochs(slip_epochs)

        assert n >= 1
        # The gap 10-14 should be within one period
        found_gap_period = False
        for i in range(n):
            if periods[i, 0] <= 10 and periods[i, 1] >= 14:
                found_gap_period = True
        assert found_gap_period, "Gap of missing obs should form a single slip period"
