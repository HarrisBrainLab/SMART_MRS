"""applied module (SMART_MRS, 2024, Bugler et al)

These functions allow the user to apply specific iterations of the artifact functions.

These functions primarily require fids [num_samples, spec_points], time[spec_points], and ppm[spec_points] data.

These functions require the artifacts module and the NumPy library.

This file can also be imported as a module and contains the following functions:

    * add_progressive_motion_artifact - returns FIDs and a list of (integers) transients affected by the linear frequency drift
    * add_subtle_motion_artifact - returns FIDs and a list of (integers) transients affected by the frequency and phase shifts [freq. list, phase list]
    * add_disruptive_motion_artifact - returns FIDs and a list of (integers) transients affected by the line broadening and baseline changes
    * add_lipid_artifact - returns FIDs and a list of (integers) transients affected by the nuisance peak
[last upd. 2025-01]
"""

# import Python packages
import numpy as np

# import functions from artifacts module
from .artifacts import add_freq_drift_linear, add_freq_shift, add_zero_order_phase_shift, add_linebroad, add_baseline, add_nuisance_peak

########################################################################################################################
# Applied functions
########################################################################################################################
def add_progressive_motion_artifact(fids, time, num_trans=None, echo=False):
    '''
    To add a frequency drift mimicking a participant's head moving in one direction over time
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing ppm values [spec_points] 
                num_trans (integer): number of transients affected by the frequency drift
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points]  ** with frequency drift**
                locs (list of integers): list of transient numbers affected by the frequency drift
    '''
    off_var=2.5                                                 # variance at each step within the drift
    slope_var=15                                                # overall frequency drift to accomplish between first and last transient
    start_trans=int(fids.shape[0]/4)                            # number of transient to start at

    if num_trans is None:                                       # number of affected transients
        num_trans = np.random.uniform(5, fids.shape[0]-start_trans)

    fids, locs = add_freq_drift_linear(fids=fids, time=time, freq_offset_var=off_var, freq_shift=slope_var, start_trans=start_trans, num_trans=num_trans, echo=echo)

    return fids, locs


def add_subtle_motion_artifact(fids, time, echo=False):
    '''
    To add a small frequency and phase shifts to certain transients to mimic some participant motion
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing ppm values [spec_points] 
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] **with subtle motion(s) artifact inserted**
                locs_fs (list of integers): list of transient numbers where frequency shift artifact(s) have been added
                locs_ps (list of integers): list of transient numbers where phase shift artifact(s) have been added
    '''   
    num_affected_trans= np.random.uniform(2, 6)                 # number of affected transients
    freq_shift_var=5                                            # +/- range of frequency shifts
    phase_shift_var=15                                          # +/- range of phase shifts

    fids, locs_fs = add_freq_shift(fids=fids, time=time, freq_var=freq_shift_var, cluster=True, num_trans=num_affected_trans, echo=echo)
    fids, locs_ps = add_zero_order_phase_shift(fids=fids, phase_var=phase_shift_var, cluster=True, num_trans=num_affected_trans, echo=echo)

    return fids, [locs_fs, locs_ps]


def add_disruptive_motion_artifact(fids, time, ppm, locs=None, num_trans=None, echo=False):
    '''
    To add a linebroadening and baseline changes to mimic large participant motion
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing time values [spec_points] 
                ppm (float): vector containing ppm values [spec_points] 
                locs (list of integers): list of transient numbers where disruptive artifact(s) have been added
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] **with disruptive motion(s) artifact inserted**
                locs (list of integers): list of transient numbers where disruptive artifact(s) have been added
    '''
    if locs is None:                                                            # number of affected transients
        num_trans = np.random.uniform(2, 6)
    else:
        num_trans = len(locs)

    damp= np.random.uniform(10, 18, size=1).repeat(num_trans).tolist()          # dampening factor for line broadening

    bvar = np.random.uniform(1.001, 1.01, size=14)
    motion_profile = { 
        "base_type": "SP",
        "x_vals": [[0.6*bvar[0]], [1.55*bvar[1]], [2.4*bvar[2]], [3.3*bvar[3]], [4.2*bvar[4]], [4.6*bvar[5]], [4.8*bvar[6]]],
        "y_vals": [[0.0033*bvar[7]], [0.0001*bvar[8]], [0.0024*bvar[9]], [0.0007*bvar[10]], [0.0032*bvar[11]], [0.0016*bvar[12]], [0.0005*bvar[13]]],
        "base_var": 0.02,
        "def_all_points": False}

    fids, locs = add_linebroad(fids=fids, time=time, damp=damp, cluster=True, locs=locs, num_trans=num_trans, echo=echo)
    fids, locs = add_baseline(fids=fids, ppm=ppm, base_profile=motion_profile, cluster=True, locs=locs, num_trans=num_trans, echo=echo)
    
    return fids, locs


def add_lipid_artifact(fids, time, edited=1, locs=None, num_trans=None, echo=False):
    '''
    To add lipid artifacts to a select number of transients
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing time values [spec_points] 
                edited (boolean): indicates whether scan is edited (True) or not (False)
                locs (list of integers): list of transient numbers where lipid artifact(s) have been added
                num_trans (integer): number of lipid artifacts in scan
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] **with lipid artifact(s) inserted**
                locs (list of integers): list of transient numbers where lipid artifact(s) have been added
    '''

    if nmb_lps is None:                                         # number of affected transients
        nmb_lps = np.random.uniform(1, fids.shape[0])
    else:
        nmb_lps = len(locs)
    
    if edited:                                                  # difference between edited scans
        edit_diff = np.random.uniform(0.85, 1.15, num_trans).tolist()
    else:
        edit_diff = np.ones(shape=(num_trans)).tolist()

    lipid_profile = {                                           # peak profile
    "peak_type": "G",
    "amp": np.random.uniform(0.01, 0.03, size=num_trans).tolist(),
    "width": np.random.uniform(0.4, 0.8, size=num_trans).tolist(),    
    "res_freq": np.random.uniform(1.4, 1.6, size=num_trans).tolist(),
    "phase": np.random.uniform(1.4, 1.6, size=num_trans).tolist(),
    "edited": edit_diff}
    
    fids, locs = add_nuisance_peak(fids=fids, time=time, peak_profile=lipid_profile, cluster=True, locs=locs, num_trans=num_trans, echo=echo)

    return fids, locs
