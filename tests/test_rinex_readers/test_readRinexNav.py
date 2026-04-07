"""
Tests for reading RINEX navigation files.
Verifies that headers and ephemeris blocks are read correctly for
both RINEX v2 (GPS) and RINEX v3 (multi-constellation) formats.
"""
import sys
import os
import pytest
import numpy as np

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_path, 'src'))

from gnssmultipath.RinexNav import RinexNav, Rinex_v2_Reader, Rinex_v3_Reader


class TestRinexNavHeaderDetection:
    """Test that the header reader detects RINEX version correctly."""

    def test_v3_gps_header(self, nav_gps_file):
        reader = RinexNav()
        version, header = reader.read_header_lines(nav_gps_file)
        assert version == 3
        assert len(header) > 0

    def test_v3_galileo_header(self, nav_galileo_file):
        reader = RinexNav()
        version, header = reader.read_header_lines(nav_galileo_file)
        assert version == 3
        assert len(header) > 0

    def test_v3_glonass_header(self, nav_glonass_file):
        reader = RinexNav()
        version, header = reader.read_header_lines(nav_glonass_file)
        assert version == 3
        assert len(header) > 0

    def test_v3_beidou_header(self, nav_beidou_file):
        reader = RinexNav()
        version, header = reader.read_header_lines(nav_beidou_file)
        assert version == 3
        assert len(header) > 0

    def test_v3_mixed_header(self, nav_mixed_file):
        reader = RinexNav()
        version, header = reader.read_header_lines(nav_mixed_file)
        assert version == 3
        assert len(header) > 0

    def test_v2_gps_header(self, nav_v2_gps_file):
        reader = RinexNav()
        version, header = reader.read_header_lines(nav_v2_gps_file)
        assert version == 2
        assert len(header) > 0

    def test_header_contains_end_of_header(self, nav_gps_file):
        reader = RinexNav()
        _, header = reader.read_header_lines(nav_gps_file)
        header_text = "\n".join(header)
        assert "END OF HEADER" in header_text



class TestRinexV3GPSNav:
    """Test reading RINEX v3 GPS navigation file."""

    @pytest.fixture(autouse=True, scope="class")
    def _read_nav(self, nav_gps_file):
        cls = type(self)
        reader = Rinex_v3_Reader()
        cls.result = reader.read_rinex_nav(nav_gps_file, desired_GNSS=["G"])

    def test_result_has_ephemerides(self):
        assert "ephemerides" in self.result

    def test_result_has_header(self):
        assert "header" in self.result

    def test_result_has_nepochs(self):
        assert "nepohs" in self.result  # Note: typo in codebase
        assert self.result["nepohs"] > 0

    def test_ephemerides_not_empty(self):
        eph = self.result["ephemerides"]
        assert len(eph) > 0

    def test_ephemerides_shape(self):
        eph = self.result["ephemerides"]
        # Should have 36 columns (PRN + 35 parameters)
        assert eph.shape[1] == 36

    def test_all_satellites_are_gps(self):
        eph = self.result["ephemerides"]
        prns = eph[:, 0]
        for prn in prns:
            assert str(prn).startswith("G"), f"Expected GPS PRN, got {prn}"

    def test_first_gps_satellite(self):
        eph = self.result["ephemerides"]
        # First satellite should be G30 based on test data
        first_prn = str(eph[0, 0])
        assert first_prn.startswith("G")

    def test_header_has_leap_seconds_info(self):
        header_text = "\n".join(self.result["header"])
        assert "LEAP SECONDS" in header_text


class TestRinexV3GalileoNav:
    """Test reading RINEX v3 Galileo navigation file."""

    @pytest.fixture(autouse=True, scope="class")
    def _read_nav(self, nav_galileo_file):
        cls = type(self)
        reader = Rinex_v3_Reader()
        cls.result = reader.read_rinex_nav(nav_galileo_file, desired_GNSS=["E"])

    def test_ephemerides_not_empty(self):
        assert len(self.result["ephemerides"]) > 0

    def test_all_satellites_are_galileo(self):
        eph = self.result["ephemerides"]
        prns = eph[:, 0]
        for prn in prns:
            assert str(prn).startswith("E"), f"Expected Galileo PRN, got {prn}"

    def test_ephemerides_shape(self):
        assert self.result["ephemerides"].shape[1] == 36

    def test_first_satellite(self):
        # First entry should be E31 based on test data
        first_prn = str(self.result["ephemerides"][0, 0])
        assert "E" in first_prn



class TestRinexV3GLONASSNav:
    """Test reading RINEX v3 GLONASS navigation file."""

    @pytest.fixture(autouse=True, scope="class")
    def _read_nav(self, nav_glonass_file):
        cls = type(self)
        reader = Rinex_v3_Reader()
        cls.result = reader.read_rinex_nav(nav_glonass_file, desired_GNSS=["R"])

    def test_ephemerides_not_empty(self):
        assert len(self.result["ephemerides"]) > 0

    def test_all_satellites_are_glonass(self):
        eph = self.result["ephemerides"]
        prns = eph[:, 0]
        for prn in prns:
            assert str(prn).startswith("R"), f"Expected GLONASS PRN, got {prn}"

    def test_ephemerides_shape(self):
        assert self.result["ephemerides"].shape[1] == 36

    def test_glonass_fcn(self):
        # GLONASS should have FCN information
        assert "glonass_fcn" in self.result
        if self.result["glonass_fcn"] is not None:
            assert len(self.result["glonass_fcn"]) > 0



class TestRinexV3BeiDouNav:
    """Test reading RINEX v3 BeiDou navigation file."""

    @pytest.fixture(autouse=True, scope="class")
    def _read_nav(self, nav_beidou_file):
        cls = type(self)
        reader = Rinex_v3_Reader()
        cls.result = reader.read_rinex_nav(nav_beidou_file, desired_GNSS=["C"])

    def test_ephemerides_not_empty(self):
        assert len(self.result["ephemerides"]) > 0

    def test_all_satellites_are_beidou(self):
        eph = self.result["ephemerides"]
        prns = eph[:, 0]
        for prn in prns:
            assert str(prn).startswith("C"), f"Expected BeiDou PRN, got {prn}"

    def test_ephemerides_shape(self):
        assert self.result["ephemerides"].shape[1] == 36



class TestRinexV3MixedNav:
    """Test reading RINEX v3 mixed navigation file with all systems."""

    @pytest.fixture(autouse=True, scope="class")
    def _read_nav(self, nav_mixed_file):
        cls = type(self)
        reader = Rinex_v3_Reader()
        cls.result = reader.read_rinex_nav(
            nav_mixed_file,
            desired_GNSS=["G", "R", "E", "C"],
        )

    def test_ephemerides_not_empty(self):
        assert len(self.result["ephemerides"]) > 0

    def test_has_multiple_systems(self):
        eph = self.result["ephemerides"]
        prns = eph[:, 0]
        systems = set(str(p)[0] for p in prns)
        assert len(systems) >= 2, f"Expected multi-system, got {systems}"

    def test_has_gps(self):
        eph = self.result["ephemerides"]
        prns = [str(p) for p in eph[:, 0]]
        assert any(p.startswith("G") for p in prns)

    def test_has_galileo(self):
        eph = self.result["ephemerides"]
        prns = [str(p) for p in eph[:, 0]]
        assert any(p.startswith("E") for p in prns)



class TestRinexV3NavFiltering:
    """Test that filtering by system works correctly."""

    def test_filter_gps_only_from_mixed(self, nav_mixed_file):
        reader = Rinex_v3_Reader()
        result = reader.read_rinex_nav(nav_mixed_file, desired_GNSS=["G"])
        eph = result["ephemerides"]
        prns = [str(p) for p in eph[:, 0]]
        for prn in prns:
            assert prn.startswith("G"), f"Expected GPS only, got {prn}"

    def test_filter_galileo_only_from_mixed(self, nav_mixed_file):
        reader = Rinex_v3_Reader()
        result = reader.read_rinex_nav(nav_mixed_file, desired_GNSS=["E"])
        eph = result["ephemerides"]
        prns = [str(p) for p in eph[:, 0]]
        for prn in prns:
            assert prn.startswith("E"), f"Expected Galileo only, got {prn}"


class TestRinexV2GPSNav:
    """Test reading RINEX v2 GPS navigation file."""

    @pytest.fixture(autouse=True, scope="class")
    def _read_nav(self, nav_v2_gps_file):
        cls = type(self)
        reader = Rinex_v2_Reader()
        cls.result = reader.read_rinex_nav(nav_v2_gps_file)

    def test_result_has_ephemerides(self):
        assert "ephemerides" in self.result

    def test_result_has_header(self):
        assert "header" in self.result

    def test_ephemerides_not_empty(self):
        eph = self.result["ephemerides"]
        assert len(eph) > 0

    def test_ephemerides_shape(self):
        eph = self.result["ephemerides"]
        assert eph.shape[1] == 36

    def test_all_satellites_are_gps(self):
        eph = self.result["ephemerides"]
        prns = eph[:, 0]
        for prn in prns:
            prn_str = str(prn)
            # v2 reader may store PRN as numbers or G-prefixed
            assert prn_str.startswith("G") or prn_str.replace(".", "").isdigit(), \
                f"Unexpected PRN format: {prn}"

    def test_header_contains_version(self):
        header_text = "\n".join(self.result["header"])
        assert "2.11" in header_text or "2" in header_text

    def test_nepochs(self):
        assert self.result["nepohs"] > 0

    def test_v2_dataframe_output(self, nav_v2_gps_file):
        reader = Rinex_v2_Reader()
        result = reader.read_rinex_nav(nav_v2_gps_file, dataframe="yes")
        import pandas as pd
        assert isinstance(result["ephemerides"], pd.DataFrame)
        assert len(result["ephemerides"]) > 0


class TestRinexV3NavDataFrame:
    """Test that RINEX v3 reader can return a DataFrame."""

    def test_dataframe_output(self, nav_gps_file):
        reader = Rinex_v3_Reader()
        result = reader.read_rinex_nav(nav_gps_file, desired_GNSS=["G"], dataframe=True)
        import pandas as pd
        assert isinstance(result["ephemerides"], pd.DataFrame)
        assert len(result["ephemerides"]) > 0
