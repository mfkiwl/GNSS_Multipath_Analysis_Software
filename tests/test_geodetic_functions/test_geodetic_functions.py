"""
Tests for gnssmultipath.Geodetic_functions — time conversion and coordinate transforms.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from gnssmultipath.Geodetic_functions import (
    date2gpstime,
    date2gpstime_vectorized,
    gpstime2date,
    ECEF2geodb,
    ECEF2enu,
    ECEF2enu_batch,
)

class TestDate2GpsTime:
    """Tests for the scalar date2gpstime function."""

    def test_gps_epoch_origin(self):
        """GPS epoch (6 Jan 1980) should be week 0, tow 0."""
        week, tow = date2gpstime(1980, 1, 6, 0, 0, 0)
        assert week == 0
        assert tow == 0

    def test_one_day_after_epoch(self):
        """7 Jan 1980 should still be week 0, tow = 86400."""
        week, tow = date2gpstime(1980, 1, 7, 0, 0, 0)
        assert week == 0
        assert tow == 86400

    def test_known_gps_week_2190(self):
        """1 Jan 2022 00:00:00 should be GPS week 2190."""
        week, tow = date2gpstime(2022, 1, 1, 0, 0, 0)
        assert round(week) == 2190
        assert round(tow) == 518400  # Saturday → 6 * 86400

    def test_intraday_time(self):
        """Hours, minutes, seconds should be correctly added to tow."""
        week, tow = date2gpstime(2022, 1, 1, 12, 30, 15)
        assert round(week) == 2190
        assert round(tow) == 518400 + 12 * 3600 + 30 * 60 + 15

    def test_end_of_week(self):
        """Last second of a GPS week (Saturday 23:59:59)."""
        week, tow = date2gpstime(2022, 12, 31, 23, 59, 59)
        assert round(tow) == 604799

    def test_pre_2000_date(self):
        """A date before Y2K should produce a valid GPS week."""
        week, tow = date2gpstime(1999, 7, 15, 8, 45, 30)
        assert round(week) == 1018

    def test_leap_year_feb_29(self):
        """Feb 29 in a leap year should work correctly."""
        week, tow = date2gpstime(2024, 2, 29, 0, 0, 0)
        assert round(week) == 2303

class TestDate2GpsTimeVectorized:
    """Tests for the vectorized array version of date2gpstime."""

    # Reference data: (year, month, day, hour, min, sec) → (week, tow)
    REFERENCE_DATES = np.array([
        [2022,  1,  1,  0,  0,  0],
        [2022,  1,  1, 12, 30, 15],
        [2020, 10, 30, 13, 22, 14],
        [1999,  7, 15,  8, 45, 30],
        [2026,  4, 10, 18,  0,  0],
        [1980,  1,  6,  0,  0,  0],
        [1980,  1,  7,  0,  0,  0],
        [2000,  1,  1,  0,  0,  0],
        [2000,  2, 29, 23, 59, 59],
        [1996,  2, 29, 12,  0,  0],
        [2023,  1,  1,  0,  0,  0],
        [2023,  2, 28, 23, 59, 59],
        [2024,  2, 29,  0,  0,  0],
        [2022,  6, 15,  6, 30, 45],
        [2022, 12, 31, 23, 59, 59],
    ])

    def test_matches_scalar_version(self):
        """Vectorized version must produce identical results to the scalar version."""
        weeks_v, tows_v = date2gpstime_vectorized(self.REFERENCE_DATES)
        for i, row in enumerate(self.REFERENCE_DATES):
            w_s, t_s = date2gpstime(*row)
            assert weeks_v[i] == round(w_s), f"Week mismatch at row {i}: {row}"
            assert tows_v[i] == round(t_s), f"TOW mismatch at row {i}: {row}"

    def test_single_row(self):
        """Should work with a single-row input."""
        dates = np.array([[2022, 1, 1, 0, 0, 0]])
        weeks, tows = date2gpstime_vectorized(dates)
        assert weeks[0] == 2190
        assert tows[0] == 518400

    def test_gps_epoch(self):
        """GPS epoch should return week=0, tow=0."""
        dates = np.array([[1980, 1, 6, 0, 0, 0]])
        weeks, tows = date2gpstime_vectorized(dates)
        assert weeks[0] == 0
        assert tows[0] == 0

    def test_returns_numpy_arrays(self):
        """Output should be numpy arrays."""
        dates = np.array([[2022, 1, 1, 0, 0, 0]])
        weeks, tows = date2gpstime_vectorized(dates)
        assert isinstance(weeks, np.ndarray)
        assert isinstance(tows, np.ndarray)

    def test_large_batch(self):
        """Should handle a large batch without error and match scalar results."""
        # Generate 1000 random dates between 2000 and 2025
        rng = np.random.default_rng(42)
        n = 1000
        years = rng.integers(2000, 2026, size=n)
        months = rng.integers(1, 13, size=n)
        days = rng.integers(1, 29, size=n)  # use 1-28 to avoid invalid dates
        hours = rng.integers(0, 24, size=n)
        minutes = rng.integers(0, 60, size=n)
        seconds = rng.integers(0, 60, size=n)
        dates = np.column_stack([years, months, days, hours, minutes, seconds])

        weeks_v, tows_v = date2gpstime_vectorized(dates)

        for i in range(n):
            w_s, t_s = date2gpstime(*dates[i])
            assert weeks_v[i] == round(w_s), f"Week mismatch at index {i}"
            assert tows_v[i] == round(t_s), f"TOW mismatch at index {i}"

class TestGpsTime2Date:
    """Tests for the scalar gpstime2date function."""

    def test_gps_epoch(self):
        """Week 0, tow 0 should return 6 Jan 1980."""
        year, month, day, hour, minute, sec = gpstime2date(0, 0)
        assert int(year) == 1980
        assert int(month) == 1
        assert int(day) == 6
        assert int(hour) == 0

    def test_known_date(self):
        """Week 2190, tow 518400 should return 1 Jan 2022 (Saturday)."""
        year, month, day, hour, minute, sec = gpstime2date(2190, 518400)
        assert int(year) == 2022
        assert int(month) == 1
        assert int(day) == 1
        assert int(hour) == 0

    def test_roundtrip(self):
        """date2gpstime → gpstime2date should recover the original date."""
        week, tow = date2gpstime(2023, 6, 15, 14, 30, 0)
        year, month, day, hour, minute, sec = gpstime2date(round(week), round(tow))
        assert int(year) == 2023
        assert int(month) == 6
        assert int(day) == 15
        assert int(hour) == 14
        assert int(minute) == 30

class TestECEF2enu:
    """Tests for ECEF-to-ENU conversion (scalar and batch)."""

    # Receiver at OPEC (approximate): lat ≈ 59.66° N, lon ≈ 10.78° E
    LAT = np.deg2rad(59.66)
    LON = np.deg2rad(10.78)

    def test_batch_matches_scalar(self):
        """ECEF2enu_batch should give identical results to looping ECEF2enu."""
        rng = np.random.default_rng(123)
        dX = rng.standard_normal(50) * 1e6
        dY = rng.standard_normal(50) * 1e6
        dZ = rng.standard_normal(50) * 1e6

        # Batch
        e_b, n_b, u_b = ECEF2enu_batch(self.LAT, self.LON, dX, dY, dZ)

        # Scalar loop
        e_s = np.empty(50)
        n_s = np.empty(50)
        u_s = np.empty(50)
        for i in range(50):
            e_s[i], n_s[i], u_s[i] = ECEF2enu(self.LAT, self.LON, dX[i], dY[i], dZ[i])

        assert_allclose(e_b, e_s, atol=1e-8)
        assert_allclose(n_b, n_s, atol=1e-8)
        assert_allclose(u_b, u_s, atol=1e-8)

    def test_zero_difference_gives_zero_enu(self):
        """Zero coordinate differences should produce zero ENU."""
        e, n, u = ECEF2enu_batch(self.LAT, self.LON,
                                  np.array([0.0]), np.array([0.0]), np.array([0.0]))
        assert_allclose(e, [0.0], atol=1e-15)
        assert_allclose(n, [0.0], atol=1e-15)
        assert_allclose(u, [0.0], atol=1e-15)

class TestECEF2geodb:
    """Tests for ECEF to geodetic coordinate conversion (Bowring's method)."""

    def test_high_latitude(self):
        """A point near the north pole should give lat close to 90°."""
        a = 6378137.0
        b = 6356752.314245
        # Tromsø, Norway: 69.65° N, 18.96° E, h ≈ 0
        lat_ref, lon_ref = np.deg2rad(69.65), np.deg2rad(18.96)
        X = a * np.cos(lat_ref) * np.cos(lon_ref)
        Y = a * np.cos(lat_ref) * np.sin(lon_ref)
        Z = b * np.sin(lat_ref)
        lat, lon, h = ECEF2geodb(a, b, X, Y, Z)
        assert_allclose(np.rad2deg(lat), 69.65, atol=0.15)
        assert_allclose(np.rad2deg(lon), 18.96, atol=0.01)

    def test_equator_prime_meridian(self):
        """A point on the equator at the prime meridian."""
        a = 6378137.0
        b = 6356752.314245
        lat, lon, h = ECEF2geodb(a, b, a, 0.0, 0.0)
        assert_allclose(np.rad2deg(lat), 0.0, atol=0.01)
        assert_allclose(np.rad2deg(lon), 0.0, atol=0.01)
