"""
Tests for the SkyPlotSummary module.

Tests the data collection, binning logic, and plot generation
for the azimuth vs elevation summary heatmaps (issue #55).
"""

import os
import sys
import numpy as np
import pytest

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_path, 'src'))

from gnssmultipath.SkyPlotSummary import (
    _collect_multipath,
    _collect_snr,
    _bin_data,
    make_skyplot_summary,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_analysis_results(nepochs=100, n_prn=10, include_snr=True):
    """Build a minimal analysisResults dict with synthetic data."""
    rng = np.random.RandomState(42)
    az = rng.uniform(0, 360, (nepochs, n_prn))
    el = rng.uniform(5, 85, (nepochs, n_prn))
    mp = rng.uniform(0, 1.5, (nepochs, n_prn))
    snr = rng.uniform(20, 55, (nepochs, n_prn))

    # Insert some NaNs to mimic missing satellites
    az[:, 0] = np.nan
    el[:, 0] = np.nan
    mp[:, 0] = np.nan
    snr[:, 0] = np.nan

    result = {
        'GNSSsystems': ['GPS'],
        'Sat_position': {
            'G': {'azimuth': az, 'elevation': el},
        },
        'GPS': {
            'Bands': ['Band_1'],
            'Band_1': {
                'Codes': ['C1C'],
                'C1C': {'multipath_range1': mp},
            },
        },
    }
    if include_snr:
        result['GPS']['SNR'] = {'S1C': snr}
    return result


def _make_multi_system_results():
    """Build analysisResults with GPS + Galileo data."""
    rng = np.random.RandomState(99)
    nepochs, n_prn = 50, 8

    result = {
        'GNSSsystems': ['GPS', 'Galileo'],
        'Sat_position': {
            'G': {
                'azimuth': rng.uniform(0, 360, (nepochs, n_prn)),
                'elevation': rng.uniform(10, 80, (nepochs, n_prn)),
            },
            'E': {
                'azimuth': rng.uniform(0, 360, (nepochs, n_prn)),
                'elevation': rng.uniform(10, 80, (nepochs, n_prn)),
            },
        },
        'GPS': {
            'Bands': ['Band_1', 'Band_2'],
            'Band_1': {
                'Codes': ['C1C'],
                'C1C': {'multipath_range1': rng.uniform(0, 1.0, (nepochs, n_prn))},
            },
            'Band_2': {
                'Codes': ['C2W'],
                'C2W': {'multipath_range1': rng.uniform(0, 0.8, (nepochs, n_prn))},
            },
            'SNR': {'S1C': rng.uniform(25, 50, (nepochs, n_prn))},
        },
        'Galileo': {
            'Bands': ['Band_1'],
            'Band_1': {
                'Codes': ['C1X'],
                'C1X': {'multipath_range1': rng.uniform(0, 0.5, (nepochs, n_prn))},
            },
            'SNR': {'S1X': rng.uniform(30, 55, (nepochs, n_prn))},
        },
    }
    return result


# ── Tests: _collect_multipath ─────────────────────────────────────────────────

class TestCollectMultipath:
    def test_returns_arrays(self):
        res = _make_analysis_results()
        az, el, mp = _collect_multipath(res)
        assert az is not None
        assert az.shape == el.shape == mp.shape

    def test_multipath_is_absolute(self):
        res = _make_analysis_results()
        res['GPS']['Band_1']['C1C']['multipath_range1'] *= -1  # make negative
        _, _, mp = _collect_multipath(res)
        assert np.all(mp[np.isfinite(mp)] >= 0)

    def test_multi_system_collects_all(self):
        res = _make_multi_system_results()
        az, el, mp = _collect_multipath(res)
        # GPS has 2 bands (C1C + C2W) + Galileo has 1 (C1X) = 3 signal combos
        # Each: 50 epochs * 8 PRNs = 400 points
        expected_total = 3 * 50 * 8
        assert len(az) == expected_total

    def test_returns_none_when_no_data(self):
        res = {'GNSSsystems': []}
        az, el, mp = _collect_multipath(res)
        assert az is None

    def test_missing_sat_position_skips_system(self):
        res = _make_analysis_results()
        del res['Sat_position']['G']
        az, el, mp = _collect_multipath(res)
        assert az is None

    def test_empty_codes_skipped(self):
        res = _make_analysis_results()
        res['GPS']['Band_1']['Codes'] = [[], 'C1C']
        az, el, mp = _collect_multipath(res)
        assert az is not None

    def test_system_filter(self):
        res = _make_multi_system_results()
        az_all, _, _ = _collect_multipath(res)
        az_gps, _, _ = _collect_multipath(res, system_filter='GPS')
        az_gal, _, _ = _collect_multipath(res, system_filter='Galileo')
        assert len(az_gps) + len(az_gal) == len(az_all)
        # GPS has 2 codes * 50*8, Galileo has 1 code * 50*8
        assert len(az_gps) == 2 * 50 * 8
        assert len(az_gal) == 1 * 50 * 8


# ── Tests: _collect_snr ───────────────────────────────────────────────────────

class TestCollectSNR:
    def test_returns_arrays(self):
        res = _make_analysis_results(include_snr=True)
        az, el, snr = _collect_snr(res)
        assert az is not None

    def test_zeros_become_nan(self):
        res = _make_analysis_results(include_snr=True)
        res['GPS']['SNR']['S1C'][:, 1] = 0  # force zeros
        _, _, snr = _collect_snr(res)
        # After collection, those zeros should be NaN
        assert not np.any(snr == 0)

    def test_returns_none_when_no_snr(self):
        res = _make_analysis_results(include_snr=False)
        az, el, snr = _collect_snr(res)
        assert az is None

    def test_multi_system_collects_all(self):
        res = _make_multi_system_results()
        az, el, snr = _collect_snr(res)
        # GPS S1C: 50*8=400, Galileo S1X: 50*8=400
        assert len(az) == 800

    def test_system_filter(self):
        res = _make_multi_system_results()
        az_gps, _, _ = _collect_snr(res, system_filter='GPS')
        az_gal, _, _ = _collect_snr(res, system_filter='Galileo')
        assert len(az_gps) == 50 * 8
        assert len(az_gal) == 50 * 8


# ── Tests: _bin_data ──────────────────────────────────────────────────────────

class TestBinData:
    def test_grid_shape_default_bins(self):
        az = np.array([10, 20, 30, 350])
        el = np.array([5, 15, 45, 85])
        vals = np.array([0.5, 1.0, 0.3, 0.8])
        az_edges, el_edges, grid, counts = _bin_data(az, el, vals)
        assert grid.shape == (45, 72)  # 90/2=45 el bins, 360/5=72 az bins

    def test_custom_bin_sizes(self):
        az = np.array([10, 20, 30])
        el = np.array([5, 15, 45])
        vals = np.array([0.5, 1.0, 0.3])
        _, _, grid, _ = _bin_data(az, el, vals, az_bin_size=10, el_bin_size=5)
        assert grid.shape == (18, 36)  # 90/5=18, 360/10=36

    def test_single_point_in_bin(self):
        az = np.array([7.5])
        el = np.array([3.0])
        vals = np.array([0.42])
        _, _, grid, counts = _bin_data(az, el, vals, az_bin_size=5, el_bin_size=2)
        # Point at az=7.5 -> bin index 1 (5-10), el=3.0 -> bin index 1 (2-4)
        assert counts[1, 1] == 1
        assert np.isclose(grid[1, 1], 0.42)

    def test_mean_of_multiple_points_in_same_bin(self):
        az = np.array([1.0, 2.0, 3.0])
        el = np.array([0.5, 0.5, 0.5])
        vals = np.array([1.0, 2.0, 3.0])
        _, _, grid, counts = _bin_data(az, el, vals, az_bin_size=5, el_bin_size=2)
        assert counts[0, 0] == 3
        assert np.isclose(grid[0, 0], 2.0)

    def test_nan_values_excluded(self):
        az = np.array([10, 10, np.nan])
        el = np.array([5, 5, 5])
        vals = np.array([1.0, 2.0, 99.0])
        _, _, grid, counts = _bin_data(az, el, vals, az_bin_size=5, el_bin_size=2)
        # The NaN-az point should be excluded
        assert counts[2, 2] == 2
        assert np.isclose(grid[2, 2], 1.5)

    def test_empty_bins_are_nan(self):
        az = np.array([10.0])
        el = np.array([5.0])
        vals = np.array([1.0])
        _, _, grid, counts = _bin_data(az, el, vals)
        assert np.sum(counts > 0) == 1
        # All other cells should be NaN
        assert np.sum(np.isfinite(grid)) == 1

    def test_edge_values(self):
        az = np.array([0.0, 359.9, 360.0])
        el = np.array([0.0, 89.9, 90.0])
        vals = np.array([1.0, 2.0, 3.0])
        _, _, grid, counts = _bin_data(az, el, vals, az_bin_size=5, el_bin_size=2)
        assert np.sum(counts > 0) >= 2  # at least 2 distinct bins populated


# ── Tests: make_skyplot_summary (integration) ─────────────────────────────────

class TestMakeSkyplotSummary:
    def test_creates_multipath_file(self, tmp_path):
        res = _make_analysis_results(include_snr=False)
        make_skyplot_summary(res, str(tmp_path))
        assert (tmp_path / 'Summary_Multipath_AzEl.png').exists()

    def test_creates_snr_file(self, tmp_path):
        res = _make_analysis_results(include_snr=True)
        make_skyplot_summary(res, str(tmp_path))
        assert (tmp_path / 'Summary_SNR_AzEl.png').exists()

    def test_creates_both_files(self, tmp_path):
        res = _make_analysis_results(include_snr=True)
        make_skyplot_summary(res, str(tmp_path))
        assert (tmp_path / 'Summary_Multipath_AzEl.png').exists()
        assert (tmp_path / 'Summary_SNR_AzEl.png').exists()

    def test_multi_system(self, tmp_path):
        res = _make_multi_system_results()
        make_skyplot_summary(res, str(tmp_path))
        # Combined
        assert (tmp_path / 'Summary_Multipath_AzEl.png').exists()
        assert (tmp_path / 'Summary_SNR_AzEl.png').exists()
        # Per-system
        assert (tmp_path / 'Summary_Multipath_AzEl_GPS.png').exists()
        assert (tmp_path / 'Summary_Multipath_AzEl_Galileo.png').exists()
        assert (tmp_path / 'Summary_SNR_AzEl_GPS.png').exists()
        assert (tmp_path / 'Summary_SNR_AzEl_Galileo.png').exists()

    def test_per_system_files_created(self, tmp_path):
        res = _make_analysis_results(include_snr=True)
        make_skyplot_summary(res, str(tmp_path))
        assert (tmp_path / 'Summary_Multipath_AzEl_GPS.png').exists()
        assert (tmp_path / 'Summary_SNR_AzEl_GPS.png').exists()

    def test_no_crash_on_empty_results(self, tmp_path):
        res = {'GNSSsystems': []}
        make_skyplot_summary(res, str(tmp_path))
        assert not (tmp_path / 'Summary_Multipath_AzEl.png').exists()
        assert not (tmp_path / 'Summary_SNR_AzEl.png').exists()

    def test_custom_bin_sizes(self, tmp_path):
        res = _make_analysis_results()
        make_skyplot_summary(res, str(tmp_path), az_bin_size=10, el_bin_size=5)
        assert (tmp_path / 'Summary_Multipath_AzEl.png').exists()

    def test_creates_output_dir_if_missing(self, tmp_path):
        out_dir = str(tmp_path / 'subdir' / 'plots')
        res = _make_analysis_results()
        make_skyplot_summary(res, out_dir)
        assert os.path.isfile(os.path.join(out_dir, 'Summary_Multipath_AzEl.png'))

    def test_file_sizes_reasonable(self, tmp_path):
        res = _make_analysis_results(include_snr=True)
        make_skyplot_summary(res, str(tmp_path))
        mp_size = (tmp_path / 'Summary_Multipath_AzEl.png').stat().st_size
        snr_size = (tmp_path / 'Summary_SNR_AzEl.png').stat().st_size
        assert mp_size > 10_000  # should be a real PNG, not empty
        assert snr_size > 10_000
