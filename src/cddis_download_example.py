"""
Example script: Downloading GNSS data from CDDIS using CDDISDownloader
=======================================================================

This script demonstrates how to use the CDDISDownloader class to download
various GNSS data products from NASA's CDDIS archive:

    - Broadcast navigation files (RINEX v2, v3, v4)
    - Multi-GNSS merged navigation files (DLR BRDM)
    - Observation files (RINEX v2 and v3)
    - SP3 precise orbit files
    - All-in-one daily data bundles

CDDIS uses anonymous FTPS login. You only need to provide your email
address (used as identification for the anonymous session). No Earthdata
account registration is needed for public GNSS data.

Usage
-----
    python cddis_download_example.py

Adjust the examples below (dates, stations, output directories) to suit
your needs. Comment/uncomment sections as desired.
"""

from pathlib import Path
from typing import Optional, Union

from gnssmultipath.utils import CDDISDownloader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMAIL = "your_email@example.com"   # Replace with your email
OUTPUT_DIR = Path("./downloaded_gnss_data")


def download_broadcast_nav(
    year: int,
    doy: int,
    rinex_version: int = 3,
    output_dir: Union[str, Path] = OUTPUT_DIR / "nav",
):
    """
    Download a daily merged broadcast navigation file.

    Parameters
    ----------
    year : int
        4-digit year.
    doy : int
        Day of year (1-366).
    rinex_version : int
        RINEX version: 2 (GPS-only .YYn), 3 (IGS multi-GNSS), or 4 (DLR).
    output_dir : str or Path
        Local directory for the downloaded file.
    """
    label = {2: "v2 GPS-only", 3: "v3 multi-GNSS (IGS)", 4: "v4 multi-GNSS (DLR)"}
    print(f"\nDownloading broadcast nav — RINEX {label.get(rinex_version, rinex_version)}")
    print(f"  Date: {year} DOY {doy:03d}  |  Output: {output_dir}")

    with CDDISDownloader(username=EMAIL) as dl:
        nav_file = dl.download_broadcast_nav(
            year=year,
            doy=doy,
            rinex_version=rinex_version,
            output_dir=output_dir,
        )
        print(f"  -> Saved to: {nav_file}")
    return nav_file


def download_multi_gnss_nav(
    year: int,
    doy: int,
    output_dir: Union[str, Path] = OUTPUT_DIR / "nav_brdm",
):
    """
    Download the DLR multi-GNSS merged broadcast nav file (BRDM).

    Contains GPS, GLONASS, Galileo, BeiDou, QZSS, and SBAS ephemerides
    generated from real-time streams by TUM/DLR.

    Parameters
    ----------
    year : int
        4-digit year.
    doy : int
        Day of year (1-366).
    output_dir : str or Path
        Local directory for the downloaded file.
    """
    print(f"\nDownloading DLR multi-GNSS merged nav (BRDM)")
    print(f"  Date: {year} DOY {doy:03d}  |  Output: {output_dir}")

    with CDDISDownloader(username=EMAIL) as dl:
        nav_file = dl.download_multi_gnss_nav(
            year=year,
            doy=doy,
            output_dir=output_dir,
        )
        print(f"  -> Saved to: {nav_file}")
    return nav_file


def download_observation(
    year: int,
    doy: int,
    station: Optional[str] = None,
    rinex_version: int = 3,
    output_dir: Union[str, Path] = OUTPUT_DIR / "obs",
):
    """
    Download a daily observation file.

    Parameters
    ----------
    year : int
        4-digit year.
    doy : int
        Day of year (1-366).
    station : str, optional
        4-character station name (e.g. "BRUX", "OPEC").
        If *None*, the first available file in the directory is downloaded.
    rinex_version : int
        2 for RINEX v2 (.YYo), 3 for RINEX v3 Hatanaka (.crx).
    output_dir : str or Path
        Local directory for the downloaded file.

    Notes
    -----
    RINEX v3 observation files on CDDIS are Hatanaka-compressed (.crx).
    You may need the external tool ``crx2rnx`` to convert them.
    """
    label = {2: "v2", 3: "v3 (Hatanaka .crx)", 4: "v4 (Hatanaka .crx)"}
    station_label = station or "(any)"
    print(f"\nDownloading observation file — RINEX {label.get(rinex_version, rinex_version)}")
    print(f"  Station: {station_label}  |  Date: {year} DOY {doy:03d}  |  Output: {output_dir}")

    with CDDISDownloader(username=EMAIL) as dl:
        obs_file = dl.download_observation(
            year=year,
            doy=doy,
            station=station,
            rinex_version=rinex_version,
            output_dir=output_dir,
        )
        print(f"  -> Saved to: {obs_file}")
        if rinex_version in (3, 4):
            print("  Note: .crx files need crx2rnx for full decompression.")
    return obs_file


def download_sp3(
    year: int,
    doy: int,
    product: str = "igs",
    output_dir: Union[str, Path] = OUTPUT_DIR / "sp3",
):
    """
    Download an SP3 precise orbit file.

    Parameters
    ----------
    year : int
        4-digit year.
    doy : int
        Day of year (1-366).
    product : str
        Orbit product: "igs" (final), "igr" (rapid), "igu" (ultra-rapid).
    output_dir : str or Path
        Local directory for the downloaded file.
    """
    print(f"\nDownloading SP3 precise orbits — product: {product}")
    print(f"  Date: {year} DOY {doy:03d}  |  Output: {output_dir}")

    with CDDISDownloader(username=EMAIL) as dl:
        sp3_file = dl.download_sp3(
            year=year,
            doy=doy,
            product=product,
            output_dir=output_dir,
        )
        print(f"  -> Saved to: {sp3_file}")
    return sp3_file


def download_sp3_by_gps_week(
    gps_week: int,
    day_of_week: int = 0,
    product: str = "igs",
    output_dir: Union[str, Path] = OUTPUT_DIR / "sp3",
):
    """
    Download an SP3 precise orbit file using GPS week number.

    Parameters
    ----------
    gps_week : int
        GPS week number.
    day_of_week : int
        Day of GPS week (0=Sunday .. 6=Saturday).
    product : str
        Orbit product: "igs" (final), "igr" (rapid), "igu" (ultra-rapid).
    output_dir : str or Path
        Local directory for the downloaded file.
    """
    print(f"\nDownloading SP3 precise orbits — product: {product}")
    print(f"  GPS week: {gps_week}  day: {day_of_week}  |  Output: {output_dir}")

    with CDDISDownloader(username=EMAIL) as dl:
        sp3_file = dl.download_sp3(
            gps_week=gps_week,
            day_of_week=day_of_week,
            product=product,
            output_dir=output_dir,
        )
        print(f"  -> Saved to: {sp3_file}")
    return sp3_file


def download_daily_bundle(
    year: int,
    doy: int,
    station: Optional[str] = None,
    nav_version: int = 3,
    include_sp3: bool = False,
    output_dir: Union[str, Path] = OUTPUT_DIR / "daily_bundle",
):
    """
    Download nav (and optionally obs + SP3) for a single date.

    Parameters
    ----------
    year : int
        4-digit year.
    doy : int
        Day of year (1-366).
    station : str, optional
        If provided, also download the observation file for this station.
    nav_version : int
        RINEX version for broadcast nav (2, 3, or 4).
    include_sp3 : bool
        If True, also download IGS final SP3 orbits.
    output_dir : str or Path
        Local directory for all downloaded files.
    """
    parts = [f"nav(v{nav_version})"]
    if station:
        parts.append(f"obs({station})")
    if include_sp3:
        parts.append("sp3")

    print(f"\nDownloading daily bundle: {' + '.join(parts)}")
    print(f"  Date: {year} DOY {doy:03d}  |  Output: {output_dir}")

    with CDDISDownloader(username=EMAIL) as dl:
        result = dl.download_daily_data(
            year=year,
            doy=doy,
            station=station,
            nav_version=nav_version,
            include_sp3=include_sp3,
            output_dir=output_dir,
        )
        print("  Downloaded files:")
        for key, path in result.items():
            print(f"    {key:>4s}: {path}")
    return result


def list_remote_directory(
    remote_dir: str,
    max_display: int = 20,
):
    """
    List available files in a CDDIS remote directory.

    Parameters
    ----------
    remote_dir : str
        Remote path, e.g. "gnss/data/daily/2024/brdc/".
    max_display : int
        Maximum number of filenames to print.
    """
    print(f"\nListing remote directory: {remote_dir}")

    with CDDISDownloader(username=EMAIL) as dl:
        files = dl.list_directory(remote_dir)

    print(f"  Total files: {len(files)}")
    for f in sorted(files)[:max_display]:
        print(f"    {f}")
    if len(files) > max_display:
        print(f"    ... and {len(files) - max_display} more")
    return files


# ---------------------------------------------------------------------------
# Main — uncomment the examples you want to run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("CDDISDownloader - GNSS Data Download Examples")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Email: {EMAIL}")

    if EMAIL == "your_email@example.com":
        print(
            "\n  WARNING: Update the EMAIL variable at the top of this "
            "script with your real email address before running.\n"
        )

    # -- Broadcast navigation files --
    # download_broadcast_nav(year=2022, doy=1, rinex_version=2)     # RINEX v2 GPS
    # download_broadcast_nav(year=2024, doy=75, rinex_version=3)    # RINEX v3 IGS
    # download_broadcast_nav(year=2024, doy=75, rinex_version=4)    # RINEX v4 DLR
    # download_multi_gnss_nav(year=2024, doy=153)                   # DLR BRDM

    # -- Observation files --
    download_observation(year=2022, doy=1, rinex_version=2)
    # download_observation(year=2024, doy=1, station="BRUX", rinex_version=3)
    # download_observation(year=2025, doy=1, rinex_version=4)

    # -- Broadcast navigation --
    # download_broadcast_nav(year=2022, doy=1, rinex_version=4)

    # -- SP3 precise orbits --
    # download_sp3(year=2022, doy=1, product="igs")
    # download_sp3_by_gps_week(gps_week=2295, day_of_week=0, product="igs")

    # -- Daily bundle (nav + obs + SP3 in one call) --
    # download_daily_bundle(year=2022, doy=1, station="BRUX", nav_version=3, include_sp3=True)

    # -- Browse remote directories --
    # list_remote_directory("gnss/data/daily/2024/brdc/")
