"""
Tests for writeOutputFile module.

Tests all sections of the output report file using synthetic analysisResults
dictionaries, as well as helper functions and the integration with real
analysis results from the pickle test fixture.

Made by: Per Helge Aarnes
E-mail: per.helge.aarnes@gmail.com
"""

import sys
import os
import pytest
import numpy as np

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_path, 'src'))

from gnssmultipath.writeOutputFile import (
    writeOutputFile,
    _safe_int,
    _slip_bins,
    _interleaved_slip_bins,
    _check_lli_active,
    _SEP,
)
from gnssmultipath.PickleHandler import PickleHandler

def _make_slip_distribution(n_tot=0, n_0_10=0, n_10_20=0, n_20_30=0,
                            n_30_40=0, n_40_50=0, n_over50=0, n_NaN=0):
    return {
        'n_slips_Tot': n_tot,
        'n_slips_0_10': n_0_10,
        'n_slips_10_20': n_10_20,
        'n_slips_20_30': n_20_30,
        'n_slips_30_40': n_30_40,
        'n_slips_40_50': n_40_50,
        'n_slips_over50': n_over50,
        'n_slips_NaN': n_NaN,
    }


def _make_code_struct(range1_code='C1C', range2_code='C2W', nSat=2, nObs=100,
                      nEst=90, rms=0.5, wrms=0.4, has_lli=False, sat_indices=None):
    """Build a minimal code_struct dict.
    
    sat_indices: explicit list of satellite indices for per-sat arrays.
                 Defaults to range(nSat) for 0-based (GPS/Galileo/BeiDou).
                 For GLONASS, pass the slot IDs (1-based) used as dict keys.
    """
    if sat_indices is None:
        sat_indices = list(range(nSat))
    arr_size = max(sat_indices) + 1  # arrays must accommodate max index

    slip_dist = _make_slip_distribution(n_tot=3, n_0_10=1, n_10_20=1, n_over50=1)
    cycle_slip_dist = _make_slip_distribution(n_tot=2, n_0_10=1, n_20_30=1)

    per_sat_slips = {}
    per_sat_lli = {}
    per_sat_both = {}
    for s in sat_indices:
        per_sat_slips[s] = _make_slip_distribution(n_tot=1, n_10_20=1)
        per_sat_lli[s] = _make_slip_distribution(n_tot=1 if has_lli else 0, n_10_20=1 if has_lli else 0)
        per_sat_both[s] = _make_slip_distribution(n_tot=1 if has_lli else 0, n_10_20=1 if has_lli else 0)

    est_per_sat = np.zeros(arr_size, dtype=int)
    obs_per_sat = np.zeros((1, arr_size), dtype=int)
    rms_per_sat = np.full(arr_size, np.nan)
    wrms_per_sat = np.full(arr_size, np.nan)
    elev_per_sat = np.full(arr_size, np.nan)
    for idx, s in enumerate(sat_indices):
        est_per_sat[s] = nEst // nSat
        obs_per_sat[0, s] = nObs // nSat
        rms_per_sat[s] = rms + 0.01 * idx
        wrms_per_sat[s] = wrms + 0.01 * idx
        elev_per_sat[s] = 30.0 + idx

    return {
        'range1_Code': range1_code,
        'range2_Code': range2_code,
        'rms_multipath_range1_averaged': rms,
        'elevation_weighted_average_rms_multipath_range1': wrms,
        'nRange1Obs': nObs,
        'nEstimates': nEst,
        'range1_slip_distribution': slip_dist,
        'cycle_slip_distribution': cycle_slip_dist,
        'range1_slip_distribution_per_sat': per_sat_slips,
        'LLI_slip_distribution': _make_slip_distribution(n_tot=2 if has_lli else 0),
        'LLI_slip_distribution_per_sat': per_sat_lli,
        'slip_distribution_per_sat_LLI_fusion': per_sat_both,
        'nEstimates_per_sat': est_per_sat,
        'n_range1_obs_per_sat': obs_per_sat,
        'rms_multipath_range1_satellitewise': rms_per_sat,
        'elevation_weighted_rms_multipath_range1_satellitewise': wrms_per_sat,
        'mean_sat_elevation_angles': elev_per_sat,
    }


def _make_band_struct(codes, sat_indices=None, **kwargs):
    """Build a minimal band struct with given code names."""
    struct = {
        'nCodes': len(codes),
        'Codes': list(codes),
    }
    for c in codes:
        struct[c] = _make_code_struct(range1_code=c, sat_indices=sat_indices, **kwargs)
    return struct


def _make_gps_only_results(nSat=2, has_lli=False):
    """Build a minimal analysisResults dict with GPS-only system."""
    obs_overview = {}
    for s in range(1, nSat + 1):
        obs_overview['Sat_' + str(s)] = {'Band_1': 'C1C', 'Band_2': 'C2W', 'Band_5': ''}

    gps_struct = {
        'nBands': 1,
        'Bands': ['Band_1'],
        'Band_1': _make_band_struct(['C1C'], nSat=nSat, has_lli=has_lli),
        'observationOverview': obs_overview,
    }

    return {
        'GNSSsystems': ['G'],
        'GPS': gps_struct,
        'ExtraOutputInfo': {
            'rinex_obs_filename': 'test_obs.rnx',
            'SP3_filename': None,
            'rinex_nav_filename': ['test_nav.rnx'],
            'markerName': 'TEST',
            'rinexVersion': '3.04 ',
            'rinexProgr': 'TestSoftware',
            'recType': 'TestReceiver',
            'Rinex_Receiver_Approx_Pos': '3149785.9652, 598260.8822, 5495348.4927',
            'tFirstObs': np.array([[2022, 1, 1, 0, 0, 0.0]]),
            'tLastObs': np.array([[2022, 1, 1, 23, 59, 30.0]]),
            'tInterval': 30,
            'GLO_Slot2ChannelMap': {},
            'nClockJumps': 0,
            'stdClockJumpInterval': 0.0,
            'meanClockJumpInterval': 'N/A',
            'ionLimit': 0.400,
            'phaseCodeLimit': 6.000,
            'elevation_cutoff': 10,
        },
    }


def _make_gps_glonass_results(nSat_gps=2, nSat_glo=2, has_lli=False):
    """Build analysisResults with both GPS and GLONASS."""
    result = _make_gps_only_results(nSat=nSat_gps, has_lli=has_lli)
    result['GNSSsystems'] = ['G', 'R']

    glo_obs = {}
    glo_slot2channel = {}
    glo_slot_ids = list(range(1, nSat_glo + 1))  # 1-based slot IDs
    for s in glo_slot_ids:
        glo_obs['Sat_' + str(s)] = {
            'Band_1': 'C1C', 'Band_2': 'C2P', 'Band_3': '', 'Band_4': '', 'Band_6': ''
        }
        glo_slot2channel[s] = s - 8  # typical GLONASS frequency channels

    result['GLONASS'] = {
        'nBands': 1,
        'Bands': ['Band_1'],
        'Band_1': _make_band_struct(['C1C'], nSat=nSat_glo, has_lli=has_lli,
                                    sat_indices=glo_slot_ids),
        'observationOverview': glo_obs,
    }
    result['ExtraOutputInfo']['GLO_Slot2ChannelMap'] = glo_slot2channel
    return result

class TestSafeInt:
    def test_plain_int(self):
        assert _safe_int(42) == 42

    def test_numpy_scalar(self):
        assert _safe_int(np.int64(7)) == 7

    def test_numpy_array_element(self):
        arr = np.array([10, 20, 30])
        assert _safe_int(arr[1]) == 20

    def test_float_truncation(self):
        assert _safe_int(3.9) == 3


class TestSlipBins:
    def test_all_zeros(self):
        dist = _make_slip_distribution()
        assert _slip_bins(dist) == (0, 0, 0, 0, 0, 0, 0)

    def test_specific_values(self):
        dist = _make_slip_distribution(n_0_10=1, n_10_20=2, n_20_30=3,
                                       n_30_40=4, n_40_50=5, n_over50=6, n_NaN=7)
        assert _slip_bins(dist) == (1, 2, 3, 4, 5, 6, 7)


class TestInterleavedSlipBins:
    def test_returns_21_values(self):
        d1 = _make_slip_distribution(n_0_10=1)
        d2 = _make_slip_distribution(n_0_10=2)
        d3 = _make_slip_distribution(n_0_10=3)
        result = _interleaved_slip_bins(d1, d2, d3)
        assert len(result) == 21
        assert result[0:3] == (1, 2, 3)  # 0-10 bin: range1, lli, both


class TestCheckLLIActive:
    def test_no_lli(self):
        results = _make_gps_only_results(has_lli=False)
        # Need to resolve system names for _check_lli_active
        name2code = {'GPS': 'G'}
        assert _check_lli_active(results, name2code, 1) is False

    def test_with_lli(self):
        results = _make_gps_only_results(has_lli=True)
        name2code = {'GPS': 'G'}
        assert _check_lli_active(results, name2code, 1) is True

class TestWriteOutputFileHeader:
    """Test that the header section is written correctly."""

    def test_header_contains_software_info(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'GNSS_MultipathAnalysis' in content
        assert 'Software version:' in content
        assert 'END OF HEADER' in content

    def test_header_contains_rinex_info(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'test_obs.rnx' in content
        assert 'test_nav.rnx' in content
        assert '3.04' in content
        assert 'TestReceiver' in content
        assert 'TEST' in content

    def test_header_contains_observation_times(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert '2022/1/1' in content
        assert '23:59:30.00' in content

    def test_header_broadcast_nav(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'Broadcast navigation filename' in content

    def test_header_sp3_nav(self, tmp_path):
        results = _make_gps_only_results()
        results['ExtraOutputInfo']['SP3_filename'] = ['test.sp3']
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'SP3 filename' in content
        assert 'test.sp3' in content

    def test_header_estimated_position(self, tmp_path):
        results = _make_gps_only_results()
        results['ExtraOutputInfo']['Estimated_Receiver_Approx_Pos'] = (3149785.0, 598260.0, 5495348.0)
        results['ExtraOutputInfo']['Estimated_Receiver_Approx_Pos_stats'] = {
            'Standard Deviations': {'Sx': 0.1, 'Sy': 0.2, 'Sz': 0.3}
        }
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'Est. approx position' in content
        assert 'St.dev of the est. position' in content

    def test_header_no_lli_note(self, tmp_path):
        results = _make_gps_only_results(has_lli=False)
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'no "Loss-of-Lock" indicators' in content

    def test_header_system_names(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert '- GPS' in content

    def test_header_user_options(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 1, 1, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'Include compact summary of analysis estimates' in content

    def test_ends_with_end_of_file(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert content.rstrip().endswith('END OF OUTPUT FILE')


class TestObservationOverview:
    """Test the observation completeness overview section."""

    def test_gps_overview(self, tmp_path):
        results = _make_gps_only_results(nSat=3)
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 1, 0)

        content = (tmp_path / outfile).read_text()
        assert 'OBSERVATION COMPLETENESS OVERVIEW' in content
        assert 'GPS Observation overview' in content
        assert 'END OF OBSERVATION COMPLETENESS OVERVIEW' in content

    def test_gps_prn_listed(self, tmp_path):
        results = _make_gps_only_results(nSat=3)
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 1, 0)

        content = (tmp_path / outfile).read_text()
        assert 'G1' in content
        assert 'G2' in content
        assert 'G3' in content

    def test_glonass_overview(self, tmp_path):
        results = _make_gps_glonass_results(nSat_glo=2)
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 1, 0)

        content = (tmp_path / outfile).read_text()
        assert 'GLONASS Observation overview' in content
        assert 'Frequency Channel' in content

    def test_no_overview_when_disabled(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'OBSERVATION COMPLETENESS OVERVIEW' not in content


class TestCompactSummary:
    """Test the compact analysis results summary section."""

    def test_compact_summary_present(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 1, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'ANALYSIS RESULTS SUMMARY (COMPACT)' in content
        assert 'END OF ANALYSIS RESULTS SUMMARY (COMPACT)' in content

    def test_compact_summary_rms_values(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 1, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'RMS multipath[meters]' in content
        assert 'Weighted RMS multipath[meters]' in content
        assert '0.500' in content  # rms
        assert '0.400' in content  # wrms

    def test_compact_summary_slip_info(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 1, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'N ambiguity slips periods' in content
        assert 'N detected cycle slips' in content
        assert 'DETECTED CYCLE SLIPS IN TOTAL' in content

    def test_compact_summary_code_name(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 1, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'C1C' in content

    def test_compact_summary_not_present_when_disabled(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'ANALYSIS RESULTS SUMMARY (COMPACT)' not in content

    def test_compact_summary_multi_system(self, tmp_path):
        results = _make_gps_glonass_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 1, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'GPS ANALYSIS SUMMARY' in content
        assert 'GLONASS ANALYSIS SUMMARY' in content


class TestDetailedSummary:
    """Test the detailed per-satellite analysis section."""

    def test_detailed_summary_present(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 1, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'BEGINNING OF GPS ANALYSIS' in content
        assert 'END OF GPS ANALYSIS' in content

    def test_detailed_summary_band_info(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 1, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'Band_1 (L1)' in content
        assert 'Frequency of carrier band [MHz]' in content
        assert '1575.42' in content

    def test_detailed_summary_code_info(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 1, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'Code signal:' in content
        assert 'RMS multipath (All SVs) [meters]' in content
        assert 'Weighted RMS multipath (All SVs) [meters]' in content

    def test_detailed_summary_satellite_table_no_lli(self, tmp_path):
        results = _make_gps_only_results(nSat=2, has_lli=False)
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 1, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'Satellite Overview' in content
        assert '|PRN|Observations' in content

    def test_detailed_summary_satellite_table_with_lli(self, tmp_path):
        results = _make_gps_only_results(nSat=2, has_lli=True)
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 1, 0, 0, 1)

        content = (tmp_path / outfile).read_text()
        assert 'Satellite Overview' in content
        assert 'Analysed' in content
        assert 'LLI' in content
        assert 'Both' in content

    def test_detailed_glonass_table_no_lli(self, tmp_path):
        results = _make_gps_glonass_results(nSat_glo=2, has_lli=False)
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 1, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'BEGINNING OF GLONASS ANALYSIS' in content
        assert 'Sat ID' in content
        assert 'Channel' in content

    def test_detailed_glonass_table_with_lli(self, tmp_path):
        results = _make_gps_glonass_results(nSat_glo=2, has_lli=True)
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 1, 0, 0, 1)

        content = (tmp_path / outfile).read_text()
        assert 'BEGINNING OF GLONASS ANALYSIS' in content
        assert 'Analysed' in content

    def test_detailed_not_present_when_disabled(self, tmp_path):
        results = _make_gps_only_results()
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 0, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'BEGINNING OF GPS ANALYSIS' not in content


class TestOutputFileCreation:
    """Test file/directory creation behavior."""

    def test_creates_output_directory(self, tmp_path):
        results = _make_gps_only_results()
        new_dir = str(tmp_path / 'new_subdir')
        writeOutputFile('test_report.txt', new_dir, results, 0, 0, 0, 0)
        assert os.path.isfile(os.path.join(new_dir, 'test_report.txt'))

    def test_default_output_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        results = _make_gps_only_results()
        writeOutputFile('test_report.txt', '', results, 0, 0, 0, 0)
        assert os.path.isfile(os.path.join('Outputs_Files', 'test_report.txt'))

    def test_none_output_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        results = _make_gps_only_results()
        writeOutputFile('test_report.txt', None, results, 0, 0, 0, 0)
        assert os.path.isfile(os.path.join('Outputs_Files', 'test_report.txt'))

    def test_all_sections_enabled(self, tmp_path):
        results = _make_gps_only_results(nSat=2, has_lli=True)
        outfile = 'test_report.txt'
        writeOutputFile(outfile, str(tmp_path), results, 1, 1, 1, 1)

        content = (tmp_path / outfile).read_text()
        assert 'END OF HEADER' in content
        assert 'OBSERVATION COMPLETENESS OVERVIEW' in content
        assert 'ANALYSIS RESULTS SUMMARY (COMPACT)' in content
        assert 'BEGINNING OF GPS ANALYSIS' in content
        assert 'END OF OUTPUT FILE' in content


class TestSeparatorConstant:
    """Test the separator line constant."""

    def test_separator_length(self):
        assert len(_SEP) == 342

    def test_separator_is_equals(self):
        assert all(c == '=' for c in _SEP)


class TestWithRealPickle:
    """Integration test using the real analysis results pickle."""

    @pytest.fixture
    def real_results(self):
        pkl_path = os.path.join(project_path, 'tests',
                                'analysisResults_OPEC00NOR_S_20220010000_01D_30S_MO_3.04_croped.pkl.zst')
        if not os.path.exists(pkl_path):
            pytest.skip("Real pickle fixture not available")
        import copy
        results = copy.deepcopy(PickleHandler.read_zstd_pickle(pkl_path))
        # writeOutputFile expects single-letter GNSS system codes;
        # the pickle may have full names if it was saved after a previous run.
        name_to_code = {'GPS': 'G', 'GLONASS': 'R', 'Galileo': 'E', 'BeiDou': 'C'}
        results['GNSSsystems'] = [
            name_to_code.get(s, s) for s in results['GNSSsystems']
        ]
        return results

    def test_real_results_header_only(self, tmp_path, real_results):
        outfile = 'real_report.txt'
        writeOutputFile(outfile, str(tmp_path), real_results, 0, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'GNSS_MultipathAnalysis' in content
        assert 'END OF HEADER' in content
        assert 'END OF OUTPUT FILE' in content

    def test_real_results_all_sections(self, tmp_path, real_results):
        outfile = 'real_report.txt'
        writeOutputFile(outfile, str(tmp_path), real_results, 1, 1, 1, 1)

        content = (tmp_path / outfile).read_text()
        assert 'OBSERVATION COMPLETENESS OVERVIEW' in content
        assert 'ANALYSIS RESULTS SUMMARY (COMPACT)' in content
        assert 'BEGINNING OF GPS ANALYSIS' in content
        assert 'BEGINNING OF GLONASS ANALYSIS' in content
        assert 'BEGINNING OF Galileo ANALYSIS' in content
        assert 'BEGINNING OF BeiDou ANALYSIS' in content
        assert 'END OF OUTPUT FILE' in content

    def test_real_results_contains_gnss_systems(self, tmp_path, real_results):
        outfile = 'real_report.txt'
        writeOutputFile(outfile, str(tmp_path), real_results, 0, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert '- GPS' in content
        assert '- GLONASS' in content
        assert '- Galileo' in content
        assert '- BeiDou' in content

    def test_real_results_compact_summary_has_rms(self, tmp_path, real_results):
        outfile = 'real_report.txt'
        writeOutputFile(outfile, str(tmp_path), real_results, 0, 1, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'RMS multipath[meters]' in content
        assert 'C1C' in content
        assert 'C1X' in content  # Galileo code

    def test_real_results_detailed_gps(self, tmp_path, real_results):
        outfile = 'real_report.txt'
        writeOutputFile(outfile, str(tmp_path), real_results, 1, 0, 0, 0)

        content = (tmp_path / outfile).read_text()
        assert 'Band_1 (L1)' in content
        assert 'Satellite Overview' in content
        assert 'RMS multipath (All SVs) [meters]' in content

    def test_real_results_observation_overview_all_systems(self, tmp_path, real_results):
        outfile = 'real_report.txt'
        writeOutputFile(outfile, str(tmp_path), real_results, 0, 0, 1, 0)

        content = (tmp_path / outfile).read_text()
        assert 'GPS Observation overview' in content
        assert 'GLONASS Observation overview' in content
        assert 'Galileo Observation overview' in content
        assert 'BeiDou Observation overview' in content
