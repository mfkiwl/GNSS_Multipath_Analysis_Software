"""
This module create writes the results to a text file.

Made by: Per Helge Aarnes
E-mail: per.helge.aarnes@gmail.com
"""

import os
import logging
from gnssmultipath import __version__
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEP = '=' * 342

_GNSS_BAND_NAME_MAP = {
    'GPS':     {1: 'L1', 2: 'L2', 5: 'L5'},
    'GLONASS': {1: 'G1', 2: 'G2', 3: 'G3', 4: 'G1a', 6: 'G2a'},
    'Galileo': {1: 'E1', 5: 'E5a', 6: 'E6', 7: 'E5b', 8: 'E5(a+b)'},
    'BeiDou':  {1: 'B1', 2: 'B1-2', 5: 'B2a', 6: 'B3', 7: 'B2b', 8: 'B2(a+b)'},
}

_GNSS_BAND_FREQ_MAP = {
    'GPS':     {1: '1575.42', 2: '1227.60', 5: '1176.45'},
    'GLONASS': {1: '1602 + k*9/16', 2: '1246 + k*7/16', 3: '1202.025', 4: '1600.995', 6: '1248.06'},
    'Galileo': {1: '1575.42', 5: '1176.45', 6: '1278.75', 7: '1207.140', 8: '1191.795'},
    'BeiDou':  {1: '1575.42', 2: '1561.098', 5: '1176.45', 6: '1268.52', 7: '1207.140', 8: '1191.795'},
}

_YES_NO = {1: 'Yes', 0: 'No'}

_GNSS_NAME2CODE_FULL = {'GPS': 'G', 'GLONASS': 'R', 'Galileo': 'E', 'BeiDou': 'C'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(val):
    """Extract a Python int from a value that may be a numpy scalar."""
    return int(val.item()) if hasattr(val, "item") else int(val)


def _slip_bins(dist):
    """Return the 7 elevation-bin slip counts as a tuple."""
    return (
        int(dist['n_slips_0_10']),
        int(dist['n_slips_10_20']),
        int(dist['n_slips_20_30']),
        int(dist['n_slips_30_40']),
        int(dist['n_slips_40_50']),
        int(dist['n_slips_over50']),
        int(dist['n_slips_NaN']),
    )


def _interleaved_slip_bins(range1_dist, lli_dist, both_dist):
    """Return 21 interleaved slip-bin values: (r1_0_10, lli_0_10, both_0_10, ...)."""
    keys = ['n_slips_0_10', 'n_slips_10_20', 'n_slips_20_30',
            'n_slips_30_40', 'n_slips_40_50', 'n_slips_over50', 'n_slips_NaN']
    result = []
    for key in keys:
        result.extend([int(range1_dist[key]), int(lli_dist[key]), int(both_dist[key])])
    return tuple(result)


def _check_lli_active(analysisResults, GNSS_Name2Code, nGNSSsystems):
    """Return True if any LLI indicators are present in the analysis."""
    for i in range(nGNSSsystems):
        sys_struct = analysisResults[list(GNSS_Name2Code.keys())[i]]
        for j in range(sys_struct['nBands']):
            band_struct = sys_struct[sys_struct['Bands'][j]]
            for k in range(band_struct['nCodes']):
                try:
                    code_struct = band_struct[band_struct['Codes'][k]]
                except:
                    continue
                if code_struct['LLI_slip_distribution']['n_slips_Tot'] > 0:
                    return True
    return False


# ---------------------------------------------------------------------------
# Section writers
# ---------------------------------------------------------------------------

def _write_header(fid, analysisResults, GNSSsystems, nGNSSsystems,
                  LLI_Active, includeResultSummary, includeCompactSummary,
                  includeObservationOverview, includeLLIOverview):
    """Write the file header block."""
    extra = analysisResults['ExtraOutputInfo']

    rinex_obs_filename    = extra['rinex_obs_filename']
    sp3_filename          = extra.get('SP3_filename', None)
    broad_filename        = extra.get('rinex_nav_filename', None)
    markerName            = extra['markerName']
    rinexVersion          = extra['rinexVersion']
    rinexProgr            = extra['rinexProgr']
    recType               = extra['recType']
    rinex_rec_approx_pos  = extra['Rinex_Receiver_Approx_Pos']
    estimated_approx_pos  = extra.get('Estimated_Receiver_Approx_Pos', None)
    esti_approx_pos_stats = extra.get('Estimated_Receiver_Approx_Pos_stats', None)
    tFirstObs             = extra['tFirstObs']
    tLastObs              = extra['tLastObs']
    tInterval             = extra['tInterval']
    nClockJumps           = extra['nClockJumps']
    stdClockJumpInterval  = extra['stdClockJumpInterval']
    meanClockJumpInterval = extra['meanClockJumpInterval']
    ionLimit              = extra['ionLimit']
    phaseCodeLimit        = extra['phaseCodeLimit']
    elevation_cutoff      = extra['elevation_cutoff']

    fid.write('GNSS_MultipathAnalysis\n')
    fid.write('Software version: %s\n' % __version__)
    fid.write('\n')
    fid.write('Software developed by Per Helge Aarnes (per.helge.aarnes@gmail.com) \n\n')
    fid.write('RINEX observation filename:\t\t %s\n' % (rinex_obs_filename))
    if sp3_filename is not None:
        fid.write('SP3 filename:\t\t\t\t\t %s\n' % (','.join(sp3_filename)))
    else:
        fid.write('Broadcast navigation filename:\t %s\n' % (','.join(broad_filename)))
    fid.write('RINEX version:\t\t\t\t\t %s\n' % (rinexVersion.strip()))
    fid.write('RINEX converting program:\t\t %s\n' % (rinexProgr))
    fid.write('Rec. approx position (RINEX):    %s\n' % (rinex_rec_approx_pos))

    if estimated_approx_pos:
        std_est_pos = (
            esti_approx_pos_stats["Standard Deviations"]["Sx"],
            esti_approx_pos_stats["Standard Deviations"]["Sy"],
            esti_approx_pos_stats["Standard Deviations"]["Sz"],
        )
        fid.write('Est. approx position: \t\t\t %s\n' % str(estimated_approx_pos))
        fid.write('St.dev of the est. position: \t %s\n' % ', '.join(map(str, std_est_pos)))

    tFirstObs_flat = tFirstObs.flatten()
    tLastObs_flat = tLastObs.flatten()

    fid.write('Marker name:\t\t\t\t\t %s\n' % (markerName))
    fid.write('Receiver type:\t\t\t\t\t %s\n' % (recType))
    fid.write('Date of observation start:\t\t %4d/%d/%d %d:%d:%.2f \n' % (tFirstObs_flat[0],tFirstObs_flat[1],tFirstObs_flat[2],tFirstObs_flat[3],tFirstObs_flat[4],tFirstObs_flat[5]))
    fid.write('Date of observation end:\t\t %4d/%d/%d %d:%d:%.2f \n'   % (tLastObs_flat[0],tLastObs_flat[1],tLastObs_flat[2],tLastObs_flat[3],tLastObs_flat[4],tLastObs_flat[5]))
    fid.write('Observation interval [seconds]:\t %d\n' % (tInterval))
    fid.write('Elevation angle cutoff [degree]: %d\n' % (elevation_cutoff))
    fid.write('Number of receiver clock jumps:\t %d\n' % (nClockJumps))
    fid.write('Average clock jumps interval:\t %s (std: %.2f seconds)\n\n' % (str(meanClockJumpInterval), stdClockJumpInterval))

    fid.write('Critical cycle slip limits [m/s]:\n')
    fid.write('- Ionospheric delay:\t\t\t%6.3f\n'% (ionLimit))
    fid.write('- Phase-code combination:\t\t%6.3f\n\n' % (phaseCodeLimit))
    fid.write('GNSS systems presents in RINEX observation file:\n')

    for i in range(0,nGNSSsystems):
        fid.write('- %s\n' % (analysisResults['GNSSsystems'][i]))

    if not LLI_Active:
        fid.write('\n\nNOTE: As there were no "Loss-of-Lock" indicators in RINEX observation file,\n. No information concerining "Loss-of-Lock" indicators is included in output file')

    fid.write('\n\nUser-specified contend included in output file\n')
    fid.write('- Include overview of observations for each satellite:\t\t\t%s\n' % (_YES_NO[includeObservationOverview]))
    fid.write('- Include compact summary of analysis estimates:\t\t\t\t%s\n' % (_YES_NO[includeCompactSummary]))
    fid.write('- Include detailed summary of analysis\n   estimates, including for each individual satellite:\t\t\t%s\n' % (_YES_NO[includeResultSummary]))
    fid.write('- Include information about "Loss-of-Lock"\n   indicators in detailed summary:\t\t\t\t\t\t\t\t%s\n' % (_YES_NO[includeLLIOverview]))

    fid.write('\n\n' + _SEP + '\n')
    fid.write(_SEP + '\n\n')
    fid.write('END OF HEADER\n\n\n')


def _write_observation_overview(fid, analysisResults, GNSSsystems, GNSS_Name2Code, nGNSSsystems, GLO_Slot2ChannelMap):
    """Write the observation completeness overview section."""
    fid.write('\n\n\n\n' + _SEP)
    fid.write('\n' + _SEP + '\n\n')
    fid.write('OBSERVATION COMPLETENESS OVERVIEW\n\n\n')

    for i in range(0, nGNSSsystems):
        sys_code = GNSS_Name2Code[analysisResults['GNSSsystems'][i]]

        if GNSSsystems[i] == 'GPS':
            obs = analysisResults['GPS']['observationOverview']
            fid.write('GPS Observation overview\n')
            fid.write(' ___________________________________________________________________________________________________\n')
            fid.write('|  PRN   |        L1 Observations          |             L2 Observations          | L5 Observations |\n')
            for PRN in range(1, len(obs) + 1):
                sat = obs['Sat_' + str(PRN)]
                if not all([sat['Band_1'], sat['Band_2'], sat['Band_5']]) == "":
                    fid.write('|________|_________________________________|______________________________________|_________________|\n')
                    fid.write('|%8s|%33s|%38s|%17s|\n' % (sys_code + str(PRN), sat['Band_1'], sat['Band_2'], sat['Band_5']))
            fid.write('|________|_________________________________|______________________________________|_________________|\n\n\n')

        elif GNSSsystems[i] == 'GLONASS':
            obs = analysisResults['GLONASS']['observationOverview']
            fid.write('GLONASS Observation overview\n')
            fid.write(' ________________________________________________________________________________________________________________________\n')
            fid.write('| Sat ID | Frequency Channel | G1 Observations | G2 Observations | G3 Observations | G1a Observations | G2a Observations |\n')
            for PRN in list(GLO_Slot2ChannelMap.keys()):
                sat = obs['Sat_' + str(PRN)]
                if not all([sat['Band_1'], sat['Band_2'], sat['Band_3'], sat['Band_4'], sat['Band_6']]) == "":
                    fid.write('|________|___________________|_________________|_________________|_________________|__________________|__________________|\n')
                    fid.write('|%8s|%19d|%17s|%17s|%17s|%18s|%18s|\n' % (
                        sys_code + str(PRN), GLO_Slot2ChannelMap[PRN],
                        sat['Band_1'], sat['Band_2'], sat['Band_3'], sat['Band_4'], sat['Band_1']))
            fid.write('|________|___________________|_________________|_________________|_________________|__________________|__________________|\n\n\n')

        elif GNSSsystems[i] == 'Galileo':
            obs = analysisResults['Galileo']['observationOverview']
            fid.write('Galileo Observation overview\n')
            fid.write(' _________________________________________________________________________________________________________\n')
            fid.write('|  PRN   | E1 Observations | E5a Observations | E6 Observations | E5b Observations | G5(a+b) Observations |\n')
            for PRN in range(1, len(obs.keys()) + 1):
                sat = obs['Sat_' + str(PRN)]
                if not all([sat['Band_1'], sat['Band_5'], sat['Band_6'], sat['Band_7'], sat['Band_8']]) == "":
                    fid.write('|________|_________________|__________________|_________________|__________________|______________________|\n')
                    fid.write('|%8s|%17s|%18s|%17s|%18s|%22s|\n' % (
                        sys_code + str(PRN), sat['Band_1'], sat['Band_5'], sat['Band_6'], sat['Band_7'], sat['Band_8']))
            fid.write('|________|_________________|__________________|_________________|__________________|______________________|\n\n\n')

        elif GNSSsystems[i] == 'BeiDou':
            obs = analysisResults['BeiDou']['observationOverview']
            fid.write('BeiDou Observation overview\n')
            fid.write(' ______________________________________________________________________________________________________________________________\n')
            fid.write('|  PRN   | B1 Observations | E1-2 Observations | B2a Observations | B3 Observations  | B2b Observations | B2(a+b) Observations |\n')
            for PRN in range(1, len(obs.keys()) + 1):
                sat = obs['Sat_' + str(PRN)]
                if not all([sat['Band_1'], sat['Band_2'], sat['Band_5'], sat['Band_6'], sat['Band_7'], sat['Band_8']]) == "":
                    fid.write('|________|_________________|___________________|__________________|__________________|__________________|______________________|\n')
                    fid.write('|%8s|%17s|%19s|%18s|%18s|%18s|%22s|\n' % (
                        sys_code + str(PRN), sat['Band_1'], sat['Band_2'], sat['Band_5'], sat['Band_6'], sat['Band_7'], sat['Band_8']))
            fid.write('|________|_________________|___________________|__________________|__________________|__________________|______________________|\n\n\n')

    fid.write(_SEP + '\n')
    fid.write(_SEP + '\n')
    fid.write('END OF OBSERVATION COMPLETENESS OVERVIEW\n\n\n\n\n')


def _write_compact_summary(fid, analysisResults, GNSSsystems, nGNSSsystems):
    """Write the compact analysis results summary section."""
    fid.write('\n\n\n\n' + _SEP)
    fid.write('\n' + _SEP + '\n\n')
    fid.write('ANALYSIS RESULTS SUMMARY (COMPACT)\n\n\n')

    for i in range(0, nGNSSsystems):
        curr_sys = GNSSsystems[i]
        sys_struct = analysisResults[analysisResults['GNSSsystems'][i]]

        # --- Build multipath / ambiguity-slip table ---
        headermsg            = '|                                             |'
        rmsMsg               = '|RMS multipath[meters]                        |'
        rmsWeightedMsg       = '|Weighted RMS multipath[meters]               |'
        nSlipsMsg            = '|N ambiguity slips periods                    |'
        slipRatioMsg         = '|Ratio of N slip periods/N obs epochs [%]     |'
        nSlipsOver10Msg      = '|N slip periods, elevation angle > 10 degrees |'
        nSlipsUnder10Msg     = '|N slip periods, elevation angle < 10 degrees |'
        nSlipsNaNMsg         = '|N slip periods, elevation angle not computed |'
        topline              = ' _____________________________________________'
        bottomline           = '|_____________________________________________|'

        fid.write('\n\n\n\n')
        fid.write('%s ANALYSIS SUMMARY\n\n' % (curr_sys))

        for j in range(sys_struct['nBands']):
            band_struct = sys_struct[sys_struct['Bands'][j]]
            for k in range(band_struct['nCodes']):
                codeName = band_struct['Codes'][k]
                try:
                    cs = band_struct[codeName]
                except:
                    logger.warning(f"INFO(GNSS_MultipathAnalysis): No estimates to put in report for {codeName} for {curr_sys}")
                    continue

                sd = cs['range1_slip_distribution']
                topline        += '_________'
                bottomline     += '________|'
                headermsg      += '%8s|' % codeName
                rmsMsg         += '%8.3f|' % cs['rms_multipath_range1_averaged']
                rmsWeightedMsg += '%8.3f|' % cs['elevation_weighted_average_rms_multipath_range1']
                slipRatioMsg   += '%8.3f|' % (100 * sd['n_slips_Tot'] / cs['nRange1Obs'])
                nSlipsMsg      += '%8d|' % sd['n_slips_Tot']
                nSlipsOver10Msg += '%8d|' % sum([sd['n_slips_10_20'], sd['n_slips_20_30'],
                                                  sd['n_slips_30_40'], sd['n_slips_40_50'], sd['n_slips_over50']])
                nSlipsUnder10Msg += '%8d|' % sd['n_slips_0_10']
                nSlipsNaNMsg     += '%8d|' % sd['n_slips_NaN']

        for line in [topline, headermsg, bottomline, rmsMsg, bottomline, rmsWeightedMsg, bottomline,
                     nSlipsMsg, bottomline, nSlipsOver10Msg, bottomline, nSlipsUnder10Msg, bottomline,
                     nSlipsNaNMsg, bottomline, slipRatioMsg, bottomline]:
            fid.write(line + '\n')

        # --- Build cycle-slip table ---
        fid.write('\n\n')
        headermsg            = '|                                             |'
        nSlipsMsg            = '|N detected cycle slips                       |'
        slipRatioMsg         = '|Ratio of N cycle slips/N obs epochs [%]      |'
        nSlipsUnder10Msg     = '|N cycle slip, elevation angle < 10 degrees   |'
        nSlips10_20Msg       = '|N cycle slip, elevation angle 10-20 degrees  |'
        nSlips20_30Msg       = '|N cycle slip, elevation angle 20-30 degrees  |'
        nSlips30_40Msg       = '|N cycle slip, elevation angle 30-40 degrees  |'
        nSlips40_50Msg       = '|N cycle slip, elevation angle 40-50 degrees  |'
        nSlipsOver50Msg      = '|N cycle slip, elevation angle > 50 degrees   |'
        nSlipsNaNMsg         = '|N cycle slip, elevation angle not computed   |'
        topline              = ' _____________________________________________'
        bottomline           = '|_____________________________________________|'

        fid.write('\n')
        fid.write('%s: DETECTED CYCLE SLIPS IN TOTAL FOR THE SIGNAL COMBINATION (IONOSPHERIC RESIDUALS & CODE-PHASE COMBINATION)\n' % (curr_sys))

        for j in range(sys_struct['nBands']):
            band_struct = sys_struct[sys_struct['Bands'][j]]
            for k in range(band_struct['nCodes']):
                codeName = band_struct['Codes'][k]
                try:
                    cs = band_struct[codeName]
                except:
                    continue

                sd = cs['cycle_slip_distribution']
                topline         += '_________'
                bottomline      += '________|'
                headermsg       += '%8s|' % codeName
                slipRatioMsg    += '%8.3f|' % (100 * sd['n_slips_Tot'] / cs['nRange1Obs'])
                nSlipsMsg       += '%8d|' % sd['n_slips_Tot']
                nSlipsUnder10Msg += '%8d|' % sd['n_slips_0_10']
                nSlips10_20Msg   += '%8d|' % sd['n_slips_10_20']
                nSlips20_30Msg   += '%8d|' % sd['n_slips_20_30']
                nSlips30_40Msg   += '%8d|' % sd['n_slips_30_40']
                nSlips40_50Msg   += '%8d|' % sd['n_slips_40_50']
                nSlipsOver50Msg  += '%8d|' % sd['n_slips_over50']

        for line in [topline, headermsg, bottomline, nSlipsMsg, bottomline,
                     nSlipsUnder10Msg, bottomline, nSlips10_20Msg, bottomline,
                     nSlips20_30Msg, bottomline, nSlips30_40Msg, bottomline,
                     nSlips40_50Msg, bottomline, nSlipsOver50Msg, bottomline,
                     slipRatioMsg, bottomline]:
            fid.write(line + '\n')

    fid.write('\n' + _SEP + '\n')
    fid.write('END OF ANALYSIS RESULTS SUMMARY (COMPACT)\n\n\n\n\n')


def _write_detailed_sat_row_with_lli(fid, code_struct, PRN, sys_code, glo_channel=None):
    """Write one per-satellite row in the detailed summary (with LLI columns)."""
    n_obs = _safe_int(code_struct['n_range1_obs_per_sat'][:, PRN])
    n_est = _safe_int(code_struct['nEstimates_per_sat'][PRN])
    rms = float(code_struct['rms_multipath_range1_satellitewise'][PRN])
    wrms = float(code_struct['elevation_weighted_rms_multipath_range1_satellitewise'][PRN])
    mean_elev = float(code_struct['mean_sat_elevation_angles'][PRN])

    r1 = code_struct['range1_slip_distribution_per_sat'][PRN]
    lli = code_struct['LLI_slip_distribution_per_sat'][PRN]
    both = code_struct['slip_distribution_per_sat_LLI_fusion'][PRN]

    tot_r1, tot_lli, tot_both = int(r1['n_slips_Tot']), int(lli['n_slips_Tot']), int(both['n_slips_Tot'])
    ratio_r1 = 100 * tot_r1 / n_obs
    ratio_lli = 100 * tot_lli / n_obs
    ratio_both = 100 * tot_both / n_obs
    bins = _interleaved_slip_bins(r1, lli, both)

    if glo_channel is not None:
        fid.write('|______|___________|____________|_______________|_________|______________|_______________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|\n')
        fid.write('|%6s|%11d|%12d|%15d|%9.3f|%14.3f|%15.3f|%10d|%7d|%8d|%10.3f|%7.3f|%8.3f|%10d|%7d|%8d|%10d|%7d|%8d|%10d|%7d|%8d|%10d|%7d|%8d|%10d|%7d|%8d|%10d|%7d|%8d|%10d|%7d|%8d|\n' % (
            sys_code + str(PRN), _safe_int(glo_channel), n_obs, n_est, rms, wrms, mean_elev,
            tot_r1, tot_lli, tot_both, ratio_r1, ratio_lli, ratio_both, *bins))
    else:
        fid.write('|___|____________|_______________|_________|______________|_______________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|\n')
        fid.write('|%3s|%12d|%15d|%9.3f|%14.3f|%15.3f|%10d|%7d|%8d|%10.3f|%7.3f|%8.3f|%10d|%7d|%8d|%10d|%7d|%8d|%10d|%7d|%8d|%10d|%7d|%8d|%10d|%7d|%8d|%10d|%7d|%8d|%10d|%7d|%8d|\n' % (
            sys_code + str(PRN), n_obs, n_est, rms, wrms, mean_elev,
            tot_r1, tot_lli, tot_both, ratio_r1, ratio_lli, ratio_both, *bins))


def _write_detailed_sat_row_no_lli(fid, code_struct, PRN, sys_code, glo_channel=None):
    """Write one per-satellite row in the detailed summary (without LLI columns)."""
    n_obs = _safe_int(code_struct['n_range1_obs_per_sat'][:, PRN])
    n_est = _safe_int(code_struct['nEstimates_per_sat'][PRN])
    rms = float(code_struct['rms_multipath_range1_satellitewise'][PRN])
    wrms = float(code_struct['elevation_weighted_rms_multipath_range1_satellitewise'][PRN])
    mean_elev = float(code_struct['mean_sat_elevation_angles'][PRN])

    sd = code_struct['range1_slip_distribution_per_sat'][PRN]
    slip_tot = int(sd['n_slips_Tot'])
    slip_ratio = 100 * slip_tot / n_obs if n_obs != 0 else float('nan')
    bins = _slip_bins(sd)

    if glo_channel is not None:
        fid.write('|______|___________|____________|_______________|_________|______________|_______________|_______________|__________|_________________|_________________|_________________|_________________|_________________|_________________|_________________|\n')
        fid.write('|%6s|%11d|%12d|%15d|%9.3f|%14.3f|%15.3f|%15d|%10.3f|%17d|%17d|%17d|%17d|%17d|%17d|%17d|\n' % (
            sys_code + str(PRN), glo_channel, n_obs, n_est, rms, wrms, mean_elev,
            slip_tot, slip_ratio, *bins))
    else:
        fid.write('|___|____________|_______________|_________|______________|_______________|_______________|__________|_________________|_________________|_________________|_________________|_________________|_________________|_________________|\n')
        fid.write('|%3s|%12d|%15d|%9.3f|%14.3f|%15.3f|%15d|%10.3f|%17d|%17d|%17d|%17d|%17d|%17d|%17d|\n' % (
            sys_code + str(PRN), n_obs, n_est, rms, wrms, mean_elev,
            slip_tot, slip_ratio, *bins))


def _write_detailed_summary(fid, analysisResults, GNSSsystems, GNSS_Name2Code,
                            nGNSSsystems, GLO_Slot2ChannelMap, includeLLIOverview):
    """Write the detailed per-satellite analysis summary section."""
    for i in range(0, nGNSSsystems):
        current_sys = GNSSsystems[i]
        sys_struct = analysisResults[GNSSsystems[i]]
        nBands = sys_struct['nBands']
        BandFreqMap = _GNSS_BAND_FREQ_MAP[GNSSsystems[i]]
        BandNameMap = _GNSS_BAND_NAME_MAP[GNSSsystems[i]]
        sys_code = GNSS_Name2Code[GNSSsystems[i]]
        is_glonass = (current_sys == 'GLONASS')

        fid.write('\n\n\n\n' + _SEP)
        fid.write('\n' + _SEP + '\n\n')
        fid.write('BEGINNING OF %s ANALYSIS\n\n' % (analysisResults['GNSSsystems'][i]))
        fid.write('Amount of carrier bands analysed: %d \n' % nBands)

        for j in range(nBands):
            bandName = sys_struct['Bands'][j]
            band_struct = sys_struct[bandName]
            nCodes = band_struct['nCodes']
            band_num = int(bandName[-1])

            fid.write('\n\n' + _SEP + '\n\n')
            if band_num in BandNameMap:
                fid.write('%s (%s)\n\n' % (bandName, BandNameMap[band_num]))
            else:
                continue
            fid.write('Frequency of carrier band [MHz]:\t\t\t\t\t %s\n' % (BandFreqMap[band_num]))
            fid.write('Amount of code signals analysed in current band:\t %d \n' % nCodes)

            for k in range(nCodes):
                try:
                    code_struct = band_struct[band_struct['Codes'][k]]
                except:
                    break

                range1Code = code_struct['range1_Code']
                nSat = len(code_struct['range1_slip_distribution_per_sat'])

                fid.write('\n------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n')
                fid.write('Code signal:\t\t\t\t\t\t\t\t\t %s\n\n' % range1Code)
                fid.write('Second code signal\n(Utilized for linear combinations):\t\t\t\t %s\n' % (code_struct['range2_Code']))
                fid.write('RMS multipath (All SVs) [meters]:\t\t\t\t%6.3f \n' % (code_struct['rms_multipath_range1_averaged']))
                fid.write('Weighted RMS multipath (All SVs) [meters]:\t\t%6.3f \n' % (code_struct['elevation_weighted_average_rms_multipath_range1']))
                fid.write('Number of %s observation epochs:\t\t\t\t %d \n' % (range1Code, code_struct['nRange1Obs']))
                fid.write('N epochs with multipath estimates:\t\t\t\t %d \n' % (code_struct['nEstimates']))
                fid.write('N ambiguity slips on %s signal:\t\t\t\t %d \n' % (
                    range1Code, code_struct['range1_slip_distribution']['n_slips_Tot']))
                fid.write('Ratio of N slip periods/N %s obs epochs [%%]:\t %.3f\n' % (
                    range1Code, 100 * code_struct['range1_slip_distribution']['n_slips_Tot'] / code_struct['nRange1Obs']))

                # --- Satellite overview table ---
                if includeLLIOverview:
                    _write_detailed_sat_table_with_lli(fid, code_struct, nSat, sys_code, is_glonass, GLO_Slot2ChannelMap, range1Code)
                else:
                    _write_detailed_sat_table_no_lli(fid, code_struct, nSat, sys_code, is_glonass, GLO_Slot2ChannelMap, range1Code)

        fid.write('\n' + _SEP + '\n')
        fid.write(_SEP + '\n')
        fid.write('END OF %s ANALYSIS\n\n\n\n' % (GNSSsystems[i]))


def _write_detailed_sat_table_with_lli(fid, code_struct, nSat, sys_code, is_glonass, GLO_Slot2ChannelMap, range1Code):
    """Write the per-satellite table with LLI columns."""
    if not is_glonass:
        fid.write('\nSatellite Overview\n')
        fid.write(' _____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________\n')
        fid.write('|   |    n %s   | n Epochs with |   RMS   | Weighted RMS |  Average Sat. |                           |     Slip Periods/Obs      |       n Slip Periods      |       n Slip Periods      |       n Slip Periods      |       n Slip Periods      |       n Slip Periods      |       n Slip Periods      |       n Slip Periods      |\n' % range1Code)
        fid.write('|PRN|Observations|   Multipath   |Multipath|  Multipath   |Elevation Angle|       n Slip Periods      |         Ratio             |      Elevation Angle      |      Elevation Angle      |      Elevation Angle      |      Elevation Angle      |      Elevation Angle      |      Elevation Angle      |      Elevation Angle      |\n')
        fid.write('|   |            |   Estimates   |[meters] |   [meters]   |   [degrees]   |                           |          [%]              |       0-10 degrees        |        10-20 degrees      |        20-30 degrees      |        30-40 degrees      |        40-50 degrees      |        >50 degrees        |        NaN degrees        |\n')
        fid.write('|   |            |               |         |              |               |___________________________|___________________________|___________________________|___________________________|___________________________|___________________________|___________________________|___________________________|___________________________|\n')
        fid.write('|   |            |               |         |              |               | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  |\n')
        for PRN in range(0, nSat):
            if code_struct['nEstimates_per_sat'][PRN] > 0:
                _write_detailed_sat_row_with_lli(fid, code_struct, PRN, sys_code)
            fid.write('|___|____________|_______________|_________|______________|_______________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|\n')
    else:
        fid.write('\nSatellite Overview\n')
        fid.write(' ____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________\n')
        fid.write('|      | Frequency |    n %s   | n Epochs with |   RMS   | Weighted RMS |  Average Sat. |                           |        Slip/Obs           |       n Slip Periods      |       n Slip Periods      |       n Slip Periods      |       n Slip Periods      |       n Slip Periods      |       n Slip Periods      |       n Slip Periods      |\n' % range1Code)
        fid.write('|Sat ID|  Channel  |Observations|   Multipath   |Multipath|  Multipath   |Elevation Angle|       n Slip Periods      |         Ratio             |      Elevation Angle      |      Elevation Angle      |      Elevation Angle      |      Elevation Angle      |      Elevation Angle      |      Elevation Angle      |      Elevation Angle      |\n')
        fid.write('|      |           |            |   Estimates   |[meters] |   [meters]   |   [degrees]   |                           |          [%]              |       0-10 degrees        |        10-20 degrees      |        20-30 degrees      |        30-40 degrees      |        40-50 degrees      |        >50 degrees        |        NaN degrees        |\n')
        fid.write('|      |           |            |               |         |              |               |___________________________|___________________________|___________________________|___________________________|___________________________|___________________________|___________________________|___________________________|___________________________|\n')
        fid.write('|      |           |            |               |         |              |               | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  | Analysed |  LLI  |  Both  |\n')
        for PRN in list(GLO_Slot2ChannelMap.keys()):
            if code_struct['nEstimates_per_sat'][PRN] > 0:
                _write_detailed_sat_row_with_lli(fid, code_struct, PRN, sys_code, glo_channel=GLO_Slot2ChannelMap[PRN])
        fid.write('|______|___________|____________|_______________|_________|______________|_______________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|__________|_______|________|\n')


def _write_detailed_sat_table_no_lli(fid, code_struct, nSat, sys_code, is_glonass, GLO_Slot2ChannelMap, range1Code):
    """Write the per-satellite table without LLI columns."""
    if not is_glonass:
        fid.write('\nSatellite Overview\n')
        fid.write(' __________________________________________________________________________________________________________________________________________________________________________________________________________________________________ \n')
        fid.write('|   |    n %s   | n Epochs with |   RMS   | Weighted RMS |  Average Sat. |               | Slip/Obs | n Slip Periods  | n Slip Periods  | n Slip Periods  | n Slip Periods  | n Slip Periods  | n Slip Periods  | n Slip Periods  |\n' % range1Code)
        fid.write('|PRN|Observations|   Multipath   |Multipath|  Multipath   |Elevation Angle|    n Slip     |  Ratio   | Elevation Angle | Elevation Angle | Elevation Angle | Elevation Angle | Elevation Angle | Elevation Angle | Elevation Angle |\n')
        fid.write('|   |            |   Estimates   |[meters] |   [meters]   |   [degrees]   |    Periods    |   [%]    |  0-10 degrees   |  10-20 degrees  |  20-30 degrees  |  30-40 degrees  |  40-50 degrees  |   >50 degrees   |   NaN degrees   |\n')
        for PRN in range(0, nSat):
            if code_struct['nEstimates_per_sat'][PRN] > 0:
                _write_detailed_sat_row_no_lli(fid, code_struct, PRN, sys_code)
        fid.write('|___|____________|_______________|_________|______________|_______________|_______________|__________|_________________|_________________|_________________|_________________|_________________|_________________|_________________|\n')
    else:
        fid.write('\nSatellite Overview\n')
        fid.write(' _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ \n')
        fid.write('|      | Frequency |    n %s   | n Epochs with |   RMS   | Weighted RMS |  Average Sat. |               | Slip/Obs | n Slip Periods  | n Slip Periods  | n Slip Periods  | n Slip Periods  | n Slip Periods  | n Slip Periods  | n Slip Periods  |\n' % range1Code)
        fid.write('|Sat ID|  Channel  |Observations|   Multipath   |Multipath|  Multipath   |Elevation Angle|    n Slip     |  Ratio   | Elevation Angle | Elevation Angle | Elevation Angle | Elevation Angle | Elevation Angle | Elevation Angle | Elevation Angle |\n')
        fid.write('|      |           |            |   Estimates   |[meters] |   [meters]   |   [degrees]   |    Periods    |   [%]    |  0-10 degrees   |  10-20 degrees  |  20-30 degrees  |  30-40 degrees  |  40-50 degrees  |   >50 degrees   |   NaN degrees   |\n')
        for PRN in list(GLO_Slot2ChannelMap.keys()):
            if code_struct['nEstimates_per_sat'][PRN] > 0:
                _write_detailed_sat_row_no_lli(fid, code_struct, PRN, sys_code, glo_channel=GLO_Slot2ChannelMap[PRN])
        fid.write('|______|___________|____________|_______________|_________|______________|_______________|_______________|__________|_________________|_________________|_________________|_________________|_________________|_________________|_________________|\n')


# ---------------------------------------------------------------------------
# Public API (unchanged signature)
# ---------------------------------------------------------------------------

def writeOutputFile(outputFilename, outputDir, analysisResults, includeResultSummary, includeCompactSummary,\
    includeObservationOverview, includeLLIOverview):

    """
    Function that write out the results of the analysis.

    """

    if outputDir is None or outputDir == "":
        outputDir = 'Outputs_Files'

    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)

    outputFilename = os.path.join(outputDir, outputFilename)

    ## -- Resolve GNSS system names
    GNSSsystems = analysisResults['GNSSsystems']
    GNSS_Name2Code = {key: val for key, val in _GNSS_NAME2CODE_FULL.items() if val in GNSSsystems}
    if 'G' in GNSSsystems:
        GNSSsystems[GNSSsystems.index('G')] = 'GPS'
    if 'R' in GNSSsystems:
        GNSSsystems[GNSSsystems.index('R')] = 'GLONASS'
    if 'E' in GNSSsystems:
        GNSSsystems[GNSSsystems.index('E')] = 'Galileo'
    if 'C' in GNSSsystems:
        GNSSsystems[GNSSsystems.index('C')] = 'BeiDou'

    nGNSSsystems = len(GNSSsystems)
    GLO_Slot2ChannelMap = analysisResults['ExtraOutputInfo']['GLO_Slot2ChannelMap']

    ## -- Check if any LLI indicators exist
    LLI_Active = _check_lli_active(analysisResults, GNSS_Name2Code, nGNSSsystems)
    if not LLI_Active:
        includeLLIOverview = 0

    ## -- Write sections
    fid = open(outputFilename, 'w+')

    _write_header(fid, analysisResults, GNSSsystems, nGNSSsystems,
                  LLI_Active, includeResultSummary, includeCompactSummary,
                  includeObservationOverview, includeLLIOverview)

    if includeObservationOverview:
        _write_observation_overview(fid, analysisResults, GNSSsystems, GNSS_Name2Code, nGNSSsystems, GLO_Slot2ChannelMap)

    if includeCompactSummary:
        _write_compact_summary(fid, analysisResults, GNSSsystems, nGNSSsystems)

    if includeResultSummary:
        _write_detailed_summary(fid, analysisResults, GNSSsystems, GNSS_Name2Code, nGNSSsystems, GLO_Slot2ChannelMap, includeLLIOverview)

    fid.write('\n\n\n\nEND OF OUTPUT FILE')
    fid.close()
    return
