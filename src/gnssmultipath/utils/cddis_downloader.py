"""
Utility for downloading GNSS data from the NASA CDDIS archive via FTPS.

CDDIS (Crustal Dynamics Data Information System) hosts GNSS observation files,
broadcast navigation files, and precise orbit (SP3) products. Access requires
a free NASA Earthdata account: https://urs.earthdata.nasa.gov/users/new

Usage examples
--------------
Download daily RINEX v3 merged broadcast navigation file::

    from gnssmultipath.utils import CDDISDownloader

    dl = CDDISDownloader(username="your_email@example.com")
    dl.download_broadcast_nav(year=2024, doy=1, output_dir="./nav")

Download observation file for a specific station::

    dl.download_observation(year=2024, doy=1, station="BRUX", output_dir="./obs")

Download SP3 precise orbits for a GPS week::

    dl.download_sp3(gps_week=2295, day_of_week=0, output_dir="./sp3")

List files in a remote directory::

    files = dl.list_directory("gnss/data/daily/2024/001/24p/")
"""

import gzip
import os
import re
import shutil
import subprocess
import tarfile
from datetime import datetime, timedelta
from ftplib import FTP_TLS
from pathlib import Path
from typing import List, Optional, Union


_CDDIS_HOST = "gdc.cddis.eosdis.nasa.gov"


def _date_to_year_doy(year: int, month: Optional[int] = None, day: Optional[int] = None,
                       doy: Optional[int] = None) -> tuple:
    """Convert year + (month, day) or year + doy to (year, doy) tuple."""
    if doy is not None:
        return year, doy
    if month is not None and day is not None:
        dt = datetime(year, month, day)
        return year, dt.timetuple().tm_yday
    raise ValueError("Provide either 'doy' or both 'month' and 'day'.")


def _date_to_gps_week(year: int, doy: int) -> tuple:
    """Convert year + doy to GPS week and day-of-week."""
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1)
    gps_epoch = datetime(1980, 1, 6)
    delta = dt - gps_epoch
    gps_week = delta.days // 7
    day_of_week = delta.days % 7
    return gps_week, day_of_week


class CDDISDownloader:
    """
    Client for downloading GNSS data from the NASA CDDIS FTPS archive.

    Parameters
    ----------
    username : str
        Your email address, used as the password for anonymous FTPS login.
        CDDIS uses anonymous FTP with your email as identification.
    host : str, optional
        CDDIS FTPS hostname. Default: ``gdc.cddis.eosdis.nasa.gov``

    Notes
    -----
    CDDIS requires FTPS (FTP over TLS) with anonymous login. The username
    ``anonymous`` is used automatically, and your email is sent as the password.
    """

    def __init__(self, username: str, host: str = _CDDIS_HOST):
        self.email = username
        self.host = host
        self._ftps: Optional[FTP_TLS] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> FTP_TLS:
        """Establish an FTPS connection to CDDIS and return the handle."""
        ftps = FTP_TLS(host=self.host)
        ftps.login(user="anonymous", passwd=self.email)
        ftps.prot_p()  # switch to protected (encrypted) data channel
        self._ftps = ftps
        return ftps

    def disconnect(self):
        """Close the FTPS connection."""
        if self._ftps is not None:
            try:
                self._ftps.quit()
            except Exception:
                self._ftps.close()
            finally:
                self._ftps = None

    @property
    def ftps(self) -> FTP_TLS:
        """Return an active FTPS connection, reconnecting if needed."""
        if self._ftps is None:
            self.connect()
        return self._ftps

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def list_directory(self, remote_dir: str) -> List[str]:
        """
        List filenames in a remote CDDIS directory.

        Parameters
        ----------
        remote_dir : str
            Remote directory path, e.g. ``"gnss/data/daily/2024/001/24p/"``

        Returns
        -------
        list of str
            Filenames found in the directory.
        """
        self.ftps.cwd("/")
        self.ftps.cwd(remote_dir)
        return self.ftps.nlst()

    def download_file(self, remote_path: str, local_path: Union[str, Path],
                      decompress: bool = True) -> Path:
        """
        Download a single file from CDDIS.

        Parameters
        ----------
        remote_path : str
            Full remote path relative to the server root,
            e.g. ``"gnss/data/daily/2024/001/24p/BRDC00IGS_R_20240010000_01D_MN.rnx.gz"``
        local_path : str or Path
            Local destination file path.
        decompress : bool, optional
            If True and the file ends with ``.gz`` or ``.Z``, decompress
            after download and remove the compressed file. Default: True.
            ``.Z`` (Unix compress) requires ``unlzw3`` (pure-Python) or
            the system ``uncompress`` / ``gzip`` command.

        Returns
        -------
        Path
            Path to the downloaded (and possibly decompressed) file.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        remote_dir = "/".join(remote_path.split("/")[:-1])
        remote_file = remote_path.split("/")[-1]

        self.ftps.cwd("/")
        if remote_dir:
            self.ftps.cwd(remote_dir)

        with open(local_path, "wb") as f:
            self.ftps.retrbinary(f"RETR {remote_file}", f.write)

        print(f"Downloaded: {local_path}")

        if decompress and local_path.suffix == ".gz":
            decompressed_path = local_path.with_suffix("")
            with gzip.open(local_path, "rb") as gz_in:
                with open(decompressed_path, "wb") as f_out:
                    f_out.write(gz_in.read())
            local_path.unlink()
            print(f"Decompressed: {decompressed_path}")
            return decompressed_path

        if decompress and local_path.suffix == ".Z":
            decompressed_path = local_path.with_suffix("")
            # Try unlzw3 (pure-Python LZW/compress decoder)
            try:
                import unlzw3
                with open(local_path, "rb") as f_in:
                    compressed = f_in.read()
                with open(decompressed_path, "wb") as f_out:
                    f_out.write(unlzw3.unlzw(compressed))
            except ImportError:
                # Fall back to system gzip -d (handles .Z on most systems)
                result = subprocess.run(
                    ["gzip", "-df", str(local_path)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to decompress {local_path}. "
                        f"Install 'unlzw3' (pip install unlzw3) or ensure "
                        f"'gzip' is available on your system.\n{result.stderr}"
                    )
            else:
                local_path.unlink()
            print(f"Decompressed: {decompressed_path}")
            return decompressed_path

        return local_path

    @staticmethod
    def _convert_hatanaka(crx_path: Path, remove_crx: bool = True) -> Path:
        """Convert a Hatanaka-compressed .crx file to standard RINEX .rnx.

        Uses the ``hatanaka`` Python package.

        Parameters
        ----------
        crx_path : Path
            Path to the ``.crx`` file.
        remove_crx : bool
            Delete the ``.crx`` after successful conversion.

        Returns
        -------
        Path
            Path to the resulting ``.rnx`` file.
        """
        import hatanaka

        rnx_path = crx_path.with_suffix(".rnx")
        with open(crx_path, "rb") as f:
            crx_bytes = f.read()
        rnx_bytes = hatanaka.decompress(crx_bytes)
        with open(rnx_path, "wb") as f:
            f.write(rnx_bytes)
        if remove_crx:
            crx_path.unlink()
        print(f"Hatanaka decompressed: {rnx_path}")
        return rnx_path

    # ------------------------------------------------------------------
    # Broadcast navigation files
    # ------------------------------------------------------------------

    def download_broadcast_nav(self, year: int, doy: Optional[int] = None,
                               month: Optional[int] = None, day: Optional[int] = None,
                               rinex_version: int = 3,
                               output_dir: Union[str, Path] = ".",
                               decompress: bool = True) -> Path:
        """
        Download a daily merged broadcast navigation file.

        Parameters
        ----------
        year : int
            4-digit year.
        doy : int, optional
            Day of year (1-366). Provide this OR (month, day).
        month, day : int, optional
            Calendar month and day. Ignored if ``doy`` is given.
        rinex_version : int, optional
            RINEX version to download: 2 (GPS-only), 3 (multi-GNSS IGS),
            or 4 (multi-GNSS DLR). Default: 3.
        output_dir : str or Path, optional
            Local directory for the downloaded file. Default: current dir.
        decompress : bool, optional
            Decompress the ``.gz`` file after download. Default: True.

        Returns
        -------
        Path
            Path to the downloaded file.

        Notes
        -----
        CDDIS directory structure:

        - **v2 GPS**: ``gnss/data/daily/YYYY/brdc/brdcDDD0.YYn.gz``
        - **v3 multi-GNSS (IGS)**: ``gnss/data/daily/YYYY/brdc/BRDC00IGS_R_YYYYDDD0000_01D_MN.rnx.gz``
        - **v4 multi-GNSS (DLR)**: ``gnss/data/daily/YYYY/brdc/BRD400DLR_S_YYYYDDD0000_01D_MN.rnx.gz``
        - **multi-GNSS (DLR)**: ``gnss/data/daily/YYYY/brdc/BRDM00DLR_S_YYYYDDD0000_01D_MN.rnx.gz``
        """
        year, doy = _date_to_year_doy(year, month, day, doy)
        yy = f"{year % 100:02d}"
        ddd = f"{doy:03d}"

        if rinex_version == 2:
            filename = f"brdc{ddd}0.{yy}n.gz"
        elif rinex_version == 3:
            filename = f"BRDC00IGS_R_{year}{ddd}0000_01D_MN.rnx.gz"
        elif rinex_version == 4:
            filename = f"BRD400DLR_S_{year}{ddd}0000_01D_MN.rnx.gz"
        else:
            raise ValueError(f"Unsupported rinex_version={rinex_version}. Use 2, 3, or 4.")

        remote_path = f"gnss/data/daily/{year}/brdc/{filename}"
        local_path = Path(output_dir) / filename
        return self.download_file(remote_path, local_path, decompress=decompress)

    def download_multi_gnss_nav(self, year: int, doy: Optional[int] = None,
                                month: Optional[int] = None, day: Optional[int] = None,
                                output_dir: Union[str, Path] = ".",
                                decompress: bool = True) -> Path:
        """
        Download the daily DLR multi-GNSS merged broadcast navigation file.

        This file (BRDM) contains GPS, GLONASS, Galileo, BeiDou, QZSS, and
        SBAS ephemerides generated from real-time streams by TUM/DLR.

        Parameters
        ----------
        year : int
            4-digit year.
        doy : int, optional
            Day of year. Provide this OR (month, day).
        month, day : int, optional
            Calendar month and day.
        output_dir : str or Path, optional
            Local directory. Default: current dir.
        decompress : bool, optional
            Decompress ``.gz`` after download. Default: True.

        Returns
        -------
        Path
            Path to the downloaded file.
        """
        year, doy = _date_to_year_doy(year, month, day, doy)
        ddd = f"{doy:03d}"
        filename = f"BRDM00DLR_S_{year}{ddd}0000_01D_MN.rnx.gz"
        remote_path = f"gnss/data/daily/{year}/brdc/{filename}"
        local_path = Path(output_dir) / filename
        return self.download_file(remote_path, local_path, decompress=decompress)

    # ------------------------------------------------------------------
    # Observation files
    # ------------------------------------------------------------------

    def download_observation(self, year: int,
                             station: Optional[str] = None,
                             doy: Optional[int] = None,
                             month: Optional[int] = None, day: Optional[int] = None,
                             rinex_version: int = 3,
                             output_dir: Union[str, Path] = ".",
                             decompress: bool = True) -> Path:
        """
        Download a daily 30-second observation file.

        Parameters
        ----------
        year : int
            4-digit year.
        station : str, optional
            4-character station name (e.g. ``"BRUX"``, ``"OPEC"``).
            If *None*, the first available file in the directory is
            downloaded.
        doy : int, optional
            Day of year. Provide this OR (month, day).
        month, day : int, optional
            Calendar month and day.
        rinex_version : int, optional
            2 for RINEX v2 (``.YYo.gz``), 3 or 4 for RINEX v3/v4
            long-name format (Hatanaka ``.crx.gz``). Default: 3.
        output_dir : str or Path, optional
            Local directory. Default: current dir.
        decompress : bool, optional
            Decompress ``.gz`` after download. Default: True.

        Returns
        -------
        Path
            Path to the downloaded file.

        Notes
        -----
        RINEX v3/v4 observation files are archived in Hatanaka-compressed
        format (``.crx``). You may need an external tool (``crx2rnx``)
        to convert them to standard RINEX if required.
        RINEX v4 obs files use the same CDDIS directory and long-name
        convention as v3.
        """
        year, doy = _date_to_year_doy(year, month, day, doy)
        yy = f"{year % 100:02d}"
        ddd = f"{doy:03d}"
        if station is not None:
            station = station.upper()

        if rinex_version == 2:
            # v2 obs have short names: ssss{ddd}s.{yy}o or ssss{ddd}s.{yy}d
            # Must not match v3 long-name files in the same directory.
            if station is not None:
                pat = re.compile(
                    rf"^{re.escape(station.lower())}{ddd}\d\.{yy}[od]", re.I
                )
            else:
                pat = re.compile(
                    rf"^\w{{4}}{ddd}\d\.{yy}[od]", re.I
                )
            filename = None
            for subdir in (f"{yy}o", f"{yy}d"):
                remote_dir = f"gnss/data/daily/{year}/{ddd}/{subdir}"
                try:
                    files = self.list_directory(remote_dir)
                    matches = [f for f in files if pat.match(f)]
                    if matches:
                        filename = matches[0]
                        break
                except Exception:
                    continue
            if filename is None:
                label = f"station '{station}'" if station else "any station"
                raise FileNotFoundError(
                    f"No RINEX v2 observation file found for {label} "
                    f"in gnss/data/daily/{year}/{ddd}/{{YYo,YYd}}."
                )
        elif rinex_version in (3, 4):
            # v3 and v4 obs files share the same CDDIS directory (YYd)
            # Long-name format: SSSS00CCC_R_YYYYDDDHHMM_...
            subdir = f"{yy}d"
            remote_dir = f"gnss/data/daily/{year}/{ddd}/{subdir}"
            # Pattern to identify v3+ long-name files (9-char station ID)
            longname_pat = re.compile(r"^\w{9}_R_", re.I)
            try:
                files = self.list_directory(remote_dir)
                if station is not None:
                    matches = [f for f in files if f.upper().startswith(station)]
                else:
                    matches = [f for f in files if longname_pat.match(f)]
                if not matches:
                    label = f"station '{station}'" if station else "any station"
                    raise FileNotFoundError(
                        f"No RINEX v{rinex_version} observation file found for {label} "
                        f"in {remote_dir}. Available files: {len(files)} total."
                    )
                filename = matches[0]
            except Exception as e:
                if "FileNotFoundError" in type(e).__name__:
                    raise
                raise FileNotFoundError(
                    f"Could not access remote directory: {remote_dir}"
                ) from e
        else:
            raise ValueError(f"Unsupported rinex_version={rinex_version}. Use 2, 3, or 4.")

        remote_path = f"gnss/data/daily/{year}/{ddd}/{subdir}/{filename}"
        local_path = Path(output_dir) / filename
        return self.download_file(remote_path, local_path, decompress=decompress)

    def download_highrate_observation(
        self,
        year: int,
        station: str,
        doy: Optional[int] = None,
        month: Optional[int] = None,
        day: Optional[int] = None,
        output_dir: Union[str, Path] = ".",
        convert_to_rnx: bool = True,
        cleanup_temp_files: bool = True,
    ) -> Path:
        """Download a full-day 1-second (high-rate) observation file.

        CDDIS stores high-rate data as a ``.tar`` archive per station per
        day containing 96 Hatanaka-compressed 15-minute segments
        (``.crx.gz``).  This method:

        1. Downloads the ``.tar`` or ``.tar.gz`` archive.
        2. Extracts and decompresses all ``.crx.gz`` segments.
        3. Optionally converts each ``.crx`` to ``.rnx`` using Hatanaka
           decompression and concatenates them into a single daily file.

        Parameters
        ----------
        year : int
            4-digit year.
        station : str
            9-character station name (e.g. ``"BRUX00BEL"``).  A 4-char
            short name (e.g. ``"BRUX"``) is also accepted — the method
            will search the remote directory for a matching file.
        doy : int, optional
            Day of year. Provide this OR (month, day).
        month, day : int, optional
            Calendar month and day.
        output_dir : str or Path, optional
            Local directory for the final file. Default: current dir.
        convert_to_rnx : bool, optional
            If True (default), convert from Hatanaka ``.crx`` to standard
            RINEX ``.rnx``.  Uses the ``hatanaka`` Python package.
        cleanup_temp_files : bool, optional
            If True (default), delete the downloaded ``.tar`` archive and
            all intermediate per-segment files after the merged daily
            file has been written.  Set to False to keep the individual
            15-minute segment files for inspection.

        Returns
        -------
        Path
            Path to the merged daily observation file (``.rnx`` or
            ``.crx`` if *convert_to_rnx* is False).

        Notes
        -----
        The ``hatanaka`` package only handles ``.crx`` <-> ``.rnx``
        conversion; the 15-minute -> 24-hour merging is performed here
        by keeping the header from the first segment and concatenating
        the observation epochs from all segments in chronological order.
        """
        year, doy_val = _date_to_year_doy(year, month, day, doy)
        yy = f"{year % 100:02d}"
        ddd = f"{doy_val:03d}"
        station = station.upper()

        remote_dir = f"gnss/data/highrate/{year}/{ddd}"

        try:
            all_files = self.list_directory(remote_dir)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not access high-rate directory: {remote_dir}"
            ) from e

        tar_matches = [
            f for f in all_files
            if f.upper().startswith(station[:4]) and ".tar" in f.lower()
        ]
        if not tar_matches:
            raise FileNotFoundError(
                f"No high-rate tar archive found for station '{station}' "
                f"in {remote_dir}. Available files: {all_files[:20]}"
            )

        tar_name = tar_matches[0]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tar_local = output_dir / tar_name

        remote_path = f"{remote_dir}/{tar_name}"
        self.download_file(remote_path, tar_local, decompress=False)

        # If .tar.gz, decompress first
        if tar_local.suffix == ".gz":
            plain_tar = tar_local.with_suffix("")
            with gzip.open(tar_local, "rb") as gz_in:
                with open(plain_tar, "wb") as f_out:
                    f_out.write(gz_in.read())
            tar_local.unlink()
            tar_local = plain_tar

        # Extract all .crx.gz segments from the tar
        extract_dir = output_dir / f"_highrate_{station[:4]}_{year}{ddd}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tar_local, "r") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            for m in members:
                m.name = Path(m.name).name
            tf.extractall(extract_dir, members=members)
        tar_local.unlink()

        # Decompress .gz files
        gz_files = sorted(extract_dir.glob("*.gz"))
        for gz_file in gz_files:
            out_file = gz_file.with_suffix("")
            with gzip.open(gz_file, "rb") as gz_in:
                with open(out_file, "wb") as f_out:
                    f_out.write(gz_in.read())
            gz_file.unlink()

        # Convert .crx to .rnx if requested
        crx_files = sorted(extract_dir.glob("*.crx"))
        if convert_to_rnx and crx_files:
            for crx in crx_files:
                self._convert_hatanaka(crx, remove_crx=cleanup_temp_files)
            segment_files = sorted(extract_dir.glob("*.rnx"))
            ext = ".rnx"
        elif crx_files:
            segment_files = crx_files
            ext = ".crx"
        else:
            segment_files = sorted(
                f for f in extract_dir.iterdir() if f.is_file()
            )
            ext = segment_files[0].suffix if segment_files else ".rnx"

        if not segment_files:
            raise FileNotFoundError(
                f"No observation segments found after extracting {tar_name}"
            )

        # Build merged daily filename from the first segment's name
        first_name = segment_files[0].name
        merged_name = re.sub(
            r"_\d{2}M_",
            "_01D_",
            first_name,
        )
        merged_name = re.sub(
            r"(\d{7})\d{4}_",
            r"\g<1>0000_",
            merged_name,
        )
        merged_path = output_dir / merged_name

        # Concatenate: keep header from first file, append data from rest
        header_lines = []
        data_lines = []
        for i, seg in enumerate(segment_files):
            with open(seg, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            eoh_idx = None
            for j, ln in enumerate(lines):
                if "END OF HEADER" in ln:
                    eoh_idx = j
                    break
            if eoh_idx is None:
                data_lines.extend(lines)
                continue
            if i == 0:
                header_lines = lines[: eoh_idx + 1]
            data_lines.extend(lines[eoh_idx + 1 :])

        with open(merged_path, "w", encoding="utf-8") as f:
            f.writelines(header_lines)
            f.writelines(data_lines)

        n_segments = len(segment_files)

        # Clean up segment directory and tar archive if requested
        if cleanup_temp_files:
            shutil.rmtree(extract_dir, ignore_errors=True)
        else:
            print(f"Kept {n_segments} segment files in {extract_dir}")

        print(f"Merged {n_segments} segments -> {merged_path}")
        return merged_path

    # ------------------------------------------------------------------
    # SP3 precise orbits
    # ------------------------------------------------------------------

    def download_sp3(self, gps_week: Optional[int] = None,
                     day_of_week: Optional[int] = None,
                     year: Optional[int] = None, doy: Optional[int] = None,
                     month: Optional[int] = None, day: Optional[int] = None,
                     product: str = "igs",
                     output_dir: Union[str, Path] = ".",
                     decompress: bool = True) -> Path:
        """
        Download an SP3 precise orbit file from the CDDIS products archive.

        Provide either (gps_week, day_of_week) or (year, doy) or (year, month, day).

        Parameters
        ----------
        gps_week : int, optional
            GPS week number.
        day_of_week : int, optional
            Day of GPS week (0=Sunday, 6=Saturday).
        year : int, optional
            4-digit year.
        doy : int, optional
            Day of year.
        month, day : int, optional
            Calendar month and day.
        product : str, optional
            Orbit product type. Common values: ``"igs"`` (final),
            ``"igr"`` (rapid), ``"igu"`` (ultra-rapid). Default: ``"igs"``.
        output_dir : str or Path, optional
            Local directory. Default: current dir.
        decompress : bool, optional
            Decompress ``.gz`` after download. Default: True.

        Returns
        -------
        Path
            Path to the downloaded file.

        Notes
        -----
        SP3 files live under ``gnss/products/WWWW/`` where WWWW is the
        GPS week number. File naming: ``<product>WWWWD.sp3.Z`` (legacy)
        or long-name format for newer products.
        """
        if gps_week is not None and day_of_week is not None:
            week = gps_week
            dow = day_of_week
        elif year is not None:
            year, d = _date_to_year_doy(year, month, day, doy)
            week, dow = _date_to_gps_week(year, d)
        else:
            raise ValueError(
                "Provide either (gps_week, day_of_week), (year, doy), "
                "or (year, month, day)."
            )

        product = product.lower()
        remote_dir = f"gnss/products/{week:04d}"

        # Try long-name format first, then legacy short-name
        try:
            files = self.list_directory(remote_dir)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not access remote directory: {remote_dir}"
            ) from e

        # Search for matching SP3 file
        sp3_matches = [f for f in files if f.lower().startswith(product) and ".sp3" in f.lower()
                       and f"{week:04d}{dow}" in f]
        if not sp3_matches:
            # Try legacy short-name format: igsWWWWD.sp3.Z
            sp3_matches = [f for f in files if f.lower().startswith(product)
                           and f.lower().endswith((".sp3.z", ".sp3.gz"))
                           and str(dow) in f]

        if not sp3_matches:
            raise FileNotFoundError(
                f"No SP3 file found for product='{product}', "
                f"GPS week={week}, day={dow} in {remote_dir}."
            )

        filename = sp3_matches[0]
        remote_path = f"{remote_dir}/{filename}"
        local_path = Path(output_dir) / filename
        return self.download_file(remote_path, local_path, decompress=decompress)

    # ------------------------------------------------------------------
    # Convenience: download everything for a date
    # ------------------------------------------------------------------

    def download_daily_data(self, year: int, doy: Optional[int] = None,
                            month: Optional[int] = None, day: Optional[int] = None,
                            station: Optional[str] = None,
                            nav_version: int = 3,
                            include_sp3: bool = False,
                            output_dir: Union[str, Path] = ".") -> dict:
        """
        Download broadcast nav (and optionally obs + SP3) for a given date.

        Parameters
        ----------
        year, doy, month, day :
            Date specification (same as other methods).
        station : str, optional
            If provided, also download the observation file for this station.
        nav_version : int, optional
            RINEX version for the broadcast nav file (2, 3, or 4). Default: 3.
        include_sp3 : bool, optional
            If True, also download the IGS final SP3 file. Default: False.
        output_dir : str or Path, optional
            Local directory.

        Returns
        -------
        dict
            Keys: ``"nav"``, ``"obs"`` (if station given), ``"sp3"``
            (if requested). Values: Path to each downloaded file.
        """
        year, d = _date_to_year_doy(year, month, day, doy)
        result = {}

        result["nav"] = self.download_broadcast_nav(
            year=year, doy=d, rinex_version=nav_version, output_dir=output_dir
        )

        if station:
            result["obs"] = self.download_observation(
                year=year, doy=d, station=station, output_dir=output_dir
            )

        if include_sp3:
            result["sp3"] = self.download_sp3(
                year=year, doy=d, output_dir=output_dir
            )

        return result
