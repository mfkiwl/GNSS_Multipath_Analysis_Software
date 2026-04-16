"""
Tests for GNSSObservationData — the pythonic observation data accessor.
"""
import sys
import os
import pytest
import numpy as np

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_path, 'src'))

from gnssmultipath.readers.readRinexObs import readRinexObs
from gnssmultipath.readers.GNSSObservationData import (
    GNSSObservationData, SystemObservations, _CodeAccessor,
)


class TestGNSSObservationDataFromSynthetic:
    """Unit tests with hand-crafted synthetic data (no I/O)."""

    @pytest.fixture(autouse=True, scope="class")
    def _build_store(self):
        cls = type(self)
        n_epochs, n_sat, n_codes = 5, 4, 3
        rng = np.random.default_rng(42)

        # Build epoch dicts (1-based keys like readRinexObs produces)
        gps_obs = {ep: rng.random((n_sat, n_codes)) for ep in range(1, n_epochs + 1)}
        gal_obs = {ep: rng.random((n_sat, 2)) for ep in range(1, n_epochs + 1)}
        gps_lli = {ep: np.zeros((n_sat, n_codes)) for ep in range(1, n_epochs + 1)}
        gps_ss  = {ep: rng.random((n_sat, n_codes)) for ep in range(1, n_epochs + 1)}

        cls.gnss_obs = {'G': gps_obs, 'E': gal_obs}
        cls.gnss_lli = {'G': gps_lli, 'E': np.nan}
        cls.gnss_ss  = {'G': gps_ss,  'E': np.nan}
        cls.obs_codes = {
            1: {'G': ['C1C', 'L1C', 'S1C']},
            2: {'E': ['C1X', 'L1X']},
        }
        cls.gnss_systems = {1: 'G', 2: 'E'}

        cls.store = GNSSObservationData(
            cls.gnss_obs, cls.obs_codes, cls.gnss_systems,
            cls.gnss_lli, cls.gnss_ss,
        )
        cls.n_epochs = n_epochs
        cls.n_sat = n_sat

    # ── Store-level tests ────────────────────────────────────────────────

    def test_systems(self):
        assert self.store.systems == ['G', 'E']

    def test_len(self):
        assert len(self.store) == 2

    def test_contains(self):
        assert 'G' in self.store
        assert 'E' in self.store
        assert 'R' not in self.store

    def test_iter(self):
        items = list(self.store)
        assert len(items) == 2
        assert all(isinstance(s, SystemObservations) for s in items)

    def test_getitem_invalid_raises(self):
        with pytest.raises(KeyError, match="System 'R' not available"):
            self.store['R']

    def test_gps_property(self):
        assert self.store.gps.system_code == 'G'

    def test_galileo_property(self):
        assert self.store.galileo.system_code == 'E'

    def test_glonass_property_raises(self):
        with pytest.raises(KeyError):
            self.store.glonass

    def test_beidou_property_raises(self):
        with pytest.raises(KeyError):
            self.store.beidou

    # ── SystemObservations tests (GPS) ───────────────────────────────────

    def test_gps_codes(self):
        assert self.store.gps.codes == ['C1C', 'L1C', 'S1C']

    def test_gps_pseudorange_codes(self):
        assert self.store.gps.pseudorange_codes == ['C1C']

    def test_gps_phase_codes(self):
        assert self.store.gps.phase_codes == ['L1C']

    def test_gps_snr_codes(self):
        assert self.store.gps.snr_codes == ['S1C']

    def test_gps_doppler_codes(self):
        assert self.store.gps.doppler_codes == []

    def test_gps_bands(self):
        assert self.store.gps.bands == ['1']

    def test_gps_n_epochs(self):
        assert self.store.gps.n_epochs == self.n_epochs

    def test_gps_n_satellites(self):
        assert self.store.gps.n_satellites == self.n_sat

    def test_gps_system_name(self):
        assert self.store.gps.system_name == 'GPS'

    def test_gal_system_name(self):
        assert self.store.galileo.system_name == 'Galileo'

    # ── Code-level data access ───────────────────────────────────────────

    def test_getitem_shape(self):
        c1c = self.store.gps['C1C']
        assert c1c.shape == (self.n_epochs, self.n_sat)

    def test_getitem_values_match_raw(self):
        c1c = self.store.gps['C1C']
        # Must equal column 0 of the stacked raw data
        stacked = np.stack(list(self.gnss_obs['G'].values()))
        np.testing.assert_array_equal(c1c, stacked[:, :, 0])

    def test_getitem_second_code(self):
        l1c = self.store.gps['L1C']
        stacked = np.stack(list(self.gnss_obs['G'].values()))
        np.testing.assert_array_equal(l1c, stacked[:, :, 1])

    def test_getitem_invalid_code_raises(self):
        with pytest.raises(KeyError, match="'C5X' not available"):
            self.store.gps['C5X']

    def test_contains_code(self):
        assert 'C1C' in self.store.gps
        assert 'L1C' in self.store.gps
        assert 'C5X' not in self.store.gps

    def test_data_property_shape(self):
        data = self.store.gps.data
        assert data.shape == (self.n_epochs, self.n_sat, 3)

    def test_band_filter(self):
        band1 = self.store.gps.band(1)
        assert sorted(band1.keys()) == ['C1C', 'L1C', 'S1C']
        assert band1['C1C'].shape == (self.n_epochs, self.n_sat)

    def test_band_filter_empty(self):
        band5 = self.store.gps.band(5)
        assert band5 == {}

    def test_by_type(self):
        pseudoranges = self.store.gps.by_type('C')
        assert list(pseudoranges.keys()) == ['C1C']
        assert pseudoranges['C1C'].shape == (self.n_epochs, self.n_sat)

    # ── LLI / SS sub-accessors ───────────────────────────────────────────

    def test_lli_accessor(self):
        lli_l1c = self.store.gps.lli['L1C']
        assert lli_l1c.shape == (self.n_epochs, self.n_sat)
        np.testing.assert_array_equal(lli_l1c, 0)  # all zeros

    def test_ss_accessor(self):
        ss_s1c = self.store.gps.ss['S1C']
        assert ss_s1c.shape == (self.n_epochs, self.n_sat)

    def test_galileo_lli_unavailable(self):
        with pytest.raises(AttributeError, match="LLI data not available"):
            self.store.galileo.lli

    def test_galileo_ss_unavailable(self):
        with pytest.raises(AttributeError, match="Signal-Strength data not available"):
            self.store.galileo.ss

    # ── repr ─────────────────────────────────────────────────────────────

    def test_system_repr(self):
        r = repr(self.store.gps)
        assert "GPS" in r
        assert "C1C" in r

    def test_store_repr(self):
        r = repr(self.store)
        assert "GNSSObservationData" in r


class TestGNSSObservationDataFromRealData:
    """Integration tests using a real RINEX 3.04 observation file."""

    @pytest.fixture(autouse=True, scope="class")
    def _read_obs(self, rinex304_obs_file):
        cls = type(self)
        cls.rinex = readRinexObs(rinex304_obs_file)
        cls.store = cls.rinex.observations

    def test_store_created(self):
        assert isinstance(self.store, GNSSObservationData)

    def test_systems_present(self):
        systems = self.store.systems
        assert 'G' in systems
        assert 'E' in systems

    def test_gps_codes_not_empty(self):
        assert len(self.store.gps.codes) > 0

    def test_gps_has_c1c(self):
        assert 'C1C' in self.store.gps

    def test_gps_c1c_shape(self):
        c1c = self.store.gps['C1C']
        assert c1c.ndim == 2
        assert c1c.shape[0] == self.rinex.nepochs

    def test_gps_c1c_matches_raw(self):
        """Verify the store provides the same data as the raw dict."""
        gps_codes = self.rinex.obsCodes[
            next(k for k, v in self.rinex.GNSSsystems.items() if v == 'G')
        ]['G']
        c1c_col = gps_codes.index('C1C')
        raw_stacked = np.stack(list(self.rinex.GNSS_obs['G'].values()))
        np.testing.assert_array_equal(
            self.store.gps['C1C'],
            raw_stacked[:, :, c1c_col],
        )

    def test_gps_pseudorange_codes_start_with_C(self):
        for code in self.store.gps.pseudorange_codes:
            assert code[0] == 'C'

    def test_gps_phase_codes_start_with_L(self):
        for code in self.store.gps.phase_codes:
            assert code[0] == 'L'

    def test_galileo_band_1(self):
        if 'E' in self.store:
            band1 = self.store.galileo.band(1)
            for code in band1:
                assert code[1] == '1'

    def test_n_epochs_matches(self):
        assert self.store.gps.n_epochs == self.rinex.nepochs

    def test_observations_property_caches(self):
        """Accessing .observations twice returns the same object."""
        obs1 = self.rinex.observations
        obs2 = self.rinex.observations
        assert obs1 is obs2

    def test_lli_available_if_read(self):
        """LLI data should be accessible if the file was read with LLI."""
        if isinstance(self.rinex.GNSS_LLI, dict) and 'G' in self.rinex.GNSS_LLI:
            gps = self.store.gps
            if gps.phase_codes:
                lli = gps.lli[gps.phase_codes[0]]
                assert lli.ndim == 2
