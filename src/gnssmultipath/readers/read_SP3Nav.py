"""
Module for reading SP3 files.

Made by: Per Helge Aarnes
E-mail: per.helge.aarnes@gmail.com
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional, Tuple, Union


@dataclass
class SP3NavData:
    """Container for parsed SP3 navigation data.

    Attributes:
        sat_pos:          Nested dict: sat_pos[system][epoch_idx][PRN] = np.array([[X, Y, Z]])
        epoch_dates:      np.ndarray of shape (nEpochs, 6) with string date components
        navGNSSsystems:   List of GNSS system codes ["G", "R", "E", "C"]
        nEpochs:          Number of position epochs
        epochInterval:    Interval between epochs in seconds
        success:          1 if parsing succeeded, 0 otherwise
    """

    sat_pos: Dict[str, Dict[int, Dict[int, np.ndarray]]] = field(default_factory=dict)
    epoch_dates: np.ndarray = field(default_factory=lambda: np.empty((0, 6)))
    navGNSSsystems: List[str] = field(default_factory=lambda: ["G", "R", "E", "C"])
    nEpochs: int = 0
    epochInterval: float = 0.0
    success: int = 1

    def as_tuple(self) -> Tuple:
        """Return as legacy 6-tuple for backward compatibility."""
        return (self.sat_pos, self.epoch_dates, self.navGNSSsystems,
                self.nEpochs, self.epochInterval, self.success)

    def as_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame with columns [Epoch, Satellite, X, Y, Z].

        Each row represents one satellite position at one epoch.
        Epoch is a string like '2022-01-01 00:00:00', Satellite is e.g. 'G01'.
        X, Y, Z are in meters.
        """
        rows = []
        for sys_code, epoch_dict in self.sat_pos.items():
            for epoch_idx, prn_dict in epoch_dict.items():
                if epoch_idx < len(self.epoch_dates):
                    parts = self.epoch_dates[epoch_idx]
                    epoch_str = f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d} {int(parts[3]):02d}:{int(parts[4]):02d}:{int(float(parts[5])):02d}"
                else:
                    epoch_str = str(epoch_idx)
                for prn, coords in prn_dict.items():
                    rows.append((
                        epoch_str,
                        f"{sys_code}{prn:02d}",
                        coords[0, 0],
                        coords[0, 1],
                        coords[0, 2],
                    ))
        df = pd.DataFrame(rows, columns=["Epoch", "Satellite", "X", "Y", "Z"])
        df["Epoch"] = pd.to_datetime(df["Epoch"])
        return df.sort_values(["Epoch", "Satellite"]).reset_index(drop=True)


class SP3NavReader:
    """Parser for SP3 precise orbit files producing dict-based position output.

    Optimized for speed via bulk line reading, pre-computed satellite lookup
    table, and minimal per-satellite overhead in the inner loop.

    Example:
        reader = SP3NavReader("orbit.sp3", desiredGNSSsystems=["G", "E"])
        data = reader.read()
        gps_epoch0_prn1 = data.sat_pos["G"][0][1]  # np.array([[X, Y, Z]])
    """

    _GNSS_SYSTEMS = ["G", "R", "E", "C"]
    _SYS_TO_INDEX = {"G": 1, "R": 2, "E": 3, "C": 4}
    _INDEX_TO_SYS = {1: "G", 2: "R", 3: "E", 4: "C"}
    _SATS_PER_HEADER_LINE = 17

    def __init__(self, filename: str, desiredGNSSsystems: Optional[List[str]] = None,
                 output_format: Literal["data", "tuple", "dataframe"] = "data"):
        self.filename = filename
        self.desiredGNSSsystems = desiredGNSSsystems if desiredGNSSsystems is not None else list(self._GNSS_SYSTEMS)
        self._desired_set = frozenset(self.desiredGNSSsystems)
        self.output_format = output_format

    def read(self) -> Union[SP3NavData, Tuple, pd.DataFrame]:
        """Read the SP3 file and return data in the configured output_format.

        Returns:
            - SP3NavData   when output_format="data"  (default)
            - 6-tuple      when output_format="tuple" (legacy)
            - pd.DataFrame when output_format="dataframe"
        """
        try:
            with open(self.filename, 'r', encoding='utf-8') as fid:
                lines = fid.readlines()
        except Exception as exc:
            raise ValueError('No file selected!') from exc

        data = self._parse(lines)
        if self.output_format == "tuple":
            return data.as_tuple()
        if self.output_format == "dataframe":
            return data.as_dataframe()
        return data

    def _parse_header(self, lines: list, n_lines: int):
        """Parse the SP3 header lines.

        Returns:
            Tuple of (line_idx, n_sat, n_epochs, epoch_interval, sat_lookup)
            where sat_lookup maps satellite index -> (sys_code, PRN) for desired systems.
        """
        line_idx = 0
        n_sat = 0
        n_epochs = 0
        epoch_interval = 0.0
        sp3_version = None
        sat_lookup = {}
        header_num = 0

        while line_idx < n_lines:
            line = lines[line_idx]
            if line and line[0] == '*':
                break
            line_idx += 1
            header_num += 1

            if header_num == 1:
                sp3_version = line[0:2]
                if sp3_version not in ('#c', '#d'):
                    print(f'ERROR(readSP3Nav): SP3 Navigation file is version {sp3_version}, must be version c or d!')
                    raise ValueError(f'SP3 version {sp3_version} not supported, must be c or d')
                if line[2] != 'P':
                    print('ERROR(readSP3Nav): SP3 Navigation file has velocity flag, should have position flag!')
                    raise ValueError('SP3 file has velocity flag, should have position flag')
                n_epochs = int(line[32:39])

            elif header_num == 2:
                epoch_interval = float(line[24:38])

            elif header_num == 3:
                n_sat = int(line[4:6]) if sp3_version == '#c' else int(line[3:6])
                sat_line = line[9:60]

                for k in range(n_sat):
                    sys_char = sat_line[0]
                    prn_num = int(sat_line[1:3])
                    sat_line = sat_line[3:]

                    if sys_char in self._desired_set:
                        sat_lookup[k] = (sys_char, prn_num)

                    if (k + 1) % self._SATS_PER_HEADER_LINE == 0 and k != 0 and (k + 1) < n_sat:
                        line_idx += 1
                        sat_line = lines[line_idx][9:60]
                        header_num += 1

        return line_idx, n_sat, n_epochs, epoch_interval, sat_lookup

    def _parse(self, lines: list) -> SP3NavData:
        """Parse all lines and return populated SP3NavData."""
        data = SP3NavData()
        n_lines = len(lines)

        line_idx, n_sat, n_epochs, epoch_interval, sat_lookup = self._parse_header(lines, n_lines)
        data.nEpochs = n_epochs
        data.epochInterval = epoch_interval

        prn_dicts = {s: {} for s in self.desiredGNSSsystems}
        epoch_dates = []

        for epoch_idx in range(n_epochs):
            if line_idx >= n_lines:
                break

            line = lines[line_idx]
            if not line or line[0] != '*':
                print(f'The number of epochs given in the headers is not correct!\n'
                      f'Instead of {n_epochs} epochs, the file contains {epoch_idx} epochs.\n'
                      f'SP3-file {self.filename} has been read successfully')
                data.nEpochs = epoch_idx
                break

            epoch_dates.append(line[3:31].split())
            line_idx += 1

            obs_by_sys = {s: {} for s in self.desiredGNSSsystems}

            for sat_idx in range(n_sat):
                if line_idx >= n_lines:
                    break
                line = lines[line_idx]
                line_idx += 1

                entry = sat_lookup.get(sat_idx)
                if entry is None:
                    continue

                sys_code, prn = entry
                parts = line[5:46].split()
                coords = np.array([[float(p) * 1000.0 for p in parts]])
                # SP3 marks bad/missing satellite positions with 0.000000 in
                # all three coordinate columns; replace with NaN so downstream
                # interpolation rejects rather than passes through (0, 0, 0).
                if coords.shape[1] >= 3 and np.all(coords[0, :3] == 0.0):
                    coords[0, :3] = np.nan
                obs_by_sys[sys_code][prn] = coords

            for sys_code in self.desiredGNSSsystems:
                if obs_by_sys[sys_code]:
                    prn_dicts[sys_code][epoch_idx] = obs_by_sys[sys_code]

        data.epoch_dates = np.array(epoch_dates) if epoch_dates else np.empty((0, 6))
        data.sat_pos = prn_dicts
        print(f'SP3 Navigation file "{self.filename}" has been read successfully.')
        return data


def readSP3Nav(filename, desiredGNSSsystems=None):
    """
    Function that reads the GNSS satellite position data from a SP3 position file.

    INPUTS:
    -------

    filename:             path and filename of sp3 position file, string

    desiredGNSSsystems:   List of strings. Each string is a code for a
                          GNSS system that should have its position data stored
                          in sat_pos. Must be one of: "G", "R", "E",
                          "C". If left undefined, it is automatically set to
                          ["G", "R", "E", "C"]

    OUTPUTS:
    -------

    sat_pos:          dict. Each elements contains position data for a
                      specific GNSS system. Order is defined by order of
                      navGNSSsystems. Each key element is another dict that
                      stores position data of a specific epoch of that
                      GNSS system. Each of these dict  is an array
                      with [X, Y, Z] position for each satellite. Each satellite
                      has their PRN number as a key.

                      sat_pos[GNSSsystem][epoch][PNR] = [X, Y, Z]
                      Ex:
                          sat_pos['G'][100][24] = [X, Y, Z]
                      This command will extract the coordinates for GPS at epoch
                      100 for satellite PRN 24.

    epoch_dates:      matrix. Each row contains date of one of the epochs.
                      [nEpochs x 6]

    navGNSSsystems:   list of strings. Each string is a code for a
                      GNSS system with position data stored in sat_pos.
                      Must be one of: "G", "R", "E", "C"

    nEpochs:          number of position epochs, integer

    epochInterval:    interval of position epochs, seconds

    success:          boolean, 1 if no error occurs, 0 otherwise

   """
    reader = SP3NavReader(filename, desiredGNSSsystems)
    return reader.read().as_tuple()
