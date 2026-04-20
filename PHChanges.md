## What's New in v1.7.0

### New Features

- **RINEX v4 support** — Full support for RINEX v4 observation and navigation files. Navigation file parsing handles all v4 message types: GPS LNAV, GLONASS FDMA, Galileo INAV/FNAV/IFNV, and BeiDou D1/D2/D1D2.
- **Azimuth-vs-elevation heatmaps** — New plot type that summarizes multipath and C/N₀ (SNR) as rectangular 2D heatmaps across azimuth (0–360°) and elevation (0–90°). Generated both as a combined overview and per GNSS system, making it easy to identify reflection zones and signal obstructions.
- **CDDIS downloader** — Built-in utility (`CDDISDownloader`) for downloading GNSS data from NASA's CDDIS archive via anonymous FTPS. Supports broadcast navigation files (RINEX v2, v3, v4), observation files, SP3 precise orbit files, and multi-GNSS merged navigation files (DLR BRDM). Downloaded `.gz` and `.Z` compressed files are automatically decompressed. Users can specify the desired RINEX version, date, and station.
- **RINEX v2 navigation file support** — RINEX v2 navigation files can now be used in combination with any observation file version. Includes robust year handling (2-digit to 4-digit conversion) and zero-padded PRN numbers.
- **Compressed pickle export** — Results can now be saved as Zstandard-compressed pickle files (`.pkl.zst`) using the `save_results_as_compressed_pickle` option, significantly reducing file size.
- **Fourth navigation file** — Added `broadcastNav4` parameter, allowing up to four separate navigation files in a single analysis run.
- **`SignalAnalyzer` class** — Refactored the signal analysis module into a `SignalAnalyzer` class for improved clarity and structure.
- **`GNSSObservationData` class** — New structured class for accessing GNSS observation data, replacing loose tuple-based returns.
- **`RinexObsData` class** — The `readRinexObs` function now returns a structured `RinexObsData` object, simplifying unpacking and improving readability.
- **`BroadcastColumns` class** — New class for RINEX broadcast ephemeris parameters with per-system DataFrame access via `RinexNavData`.
- **`SP3NavData` / `SP3NavReader` classes** — Refactored SP3 data handling with improved structure and optional DataFrame output.

### Improvements

- **Performance** — Substantially improved speed of the RINEX observation reader by eliminating redundant file passes and optimizing epoch counting. Vectorized ECEF-to-ENU conversions, eliminated epoch-level loops in LLI slip detection, and optimized SP3 interpolation with vectorized DataFrame operations. Vectorized `date2gpstime` conversion using pure NumPy arithmetic.
- **Report output** — Refactored the text report module with consistent formatting, per-elevation-band cycle slip counts, and software version in the header. Decomposed into clean section-writer functions for maintainability. Improved alignment and readability of headers and user-specified options.
- **Observation reader robustness** — Improved handling of RINEX 2.11 event flags (e.g., file splices), proper interval detection from observation data when not in the header, and context-managed file handles. Enhanced SP3 and RINEX readers to handle missing satellite positions and clock biases gracefully.
- **General refactoring** — Extracted the main analysis module (`GNSS_MultipathAnalysis.py`) from 912 to 758 lines by extracting helper functions and module-level constants. Improved input validation and error handling throughout. Removed commented-out sections and improved code readability across multiple files.
- **Project structure** — Reorganized modules into `readers/`, `plot/`, and `utils/` sub-packages for better modularity. Moved `readRinexObs`, `SP3Reader`, `RinexNav`, and `read_SP3Nav` into `readers/`; `make_polarplot`, `plotResults`, and `SkyPlotSummary` into `plot/`; `PickleHandler`, `StatisticalAnalysis`, `createCSVfile`, `writeOutputFile`, and `cddis_downloader` into `utils/`.
- **Polar plots** — Refactored polar plot functions to support LaTeX rendering; removed the separate non-TEX polar plot file (`make_polarplot_dont_use_TEX.py`).
- **Cycle slip detection** — Refactored detection functions for improved clarity and performance; updated LLI slip detection logic and improved Newton method for eccentric anomaly iteration.
- **Azimuth/elevation calculations** — Refactored for improved clarity and performance; updated cycle slip detection logic to handle edge cases.

### Bug Fixes

- Fixed `RuntimeWarning: divide by zero` in signal delay computations by adding safety checks for carrier frequencies.
- Fixed RINEX 2.11 event flag handler incorrectly matching normal epochs as special events.
- Fixed 2-digit year display (`0022` → `2022`) in RINEX 2.11 epoch warnings.
- Fixed file handle leaks in SP3 and navigation file readers.
- Replaced deprecated `numpy.in1d` with `numpy.isin`.
- Fixed RK4 sign in GLONASS coordinate calculations.
- Fixed slip period end detection in `getLLISlipPeriods` function.
- Fixed SP3 reading routine bug.
- Fixed GNSS multipath analysis to handle non-string observation codes.

### Testing

- Significantly expanded test coverage with new test modules for RINEX readers (observation, navigation, SP3), cycle slip detection, signal delay estimation, sky plot summary, report output, geodetic functions, `GNSSObservationData`, and performance regression tests. Total test count increased to 369.