"""applied module (SMART_MRS, 2024, Bugler et al)

These functions allow the user to apply specific iterations of the artifact functions.

These functions primarily require fids [num_samples, spec_points], time[spec_points], and ppm[spec_points] data.

These functions require the artifacts module and the NumPy library.

This file can also be imported as a module and contains the following functions:

    * add_progressive_motion_artifact - returns FIDs and a list of (integers) transients affected by the linear frequency drift
    * add_subtle_motion_artifact - returns FIDs and a list of (integers) transients affected by the frequency and phase shifts [freq. list, phase list]
    * add_disruptive_motion_artifact - returns FIDs and a list of (integers) transients affected by the line broadening and baseline changes
    * add_lipid_artifact - returns FIDs and a list of (integers) transients affected by the nuisance peak
"""

# import Python packages
import numpy as np

# import functions from artifacts module
from .artifacts import add_freq_drift_linear, add_freq_shift, add_zero_order_phase_shift, add_linebroad, add_baseline, add_nuisance_peak

# applied module functions
def add_progressive_motion_artifact(fids, time, num_affected_trans=None, echo=False):
    '''
    To add a frequency drift mimicking a participant's head moving in one direction over time
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing ppm values [spec_points] 
                num_affected_trans (integer): number of transients affected by the frequency drift
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points]  ** with frequency drift**
                trans_affected (list of integers): list of transient numbers affected by the frequency drift
    '''
    off_var=2.5                                                 # variance at each step within the drift
    slope_var=15                                                # overall frequency drift to accomplish between first and last transient
    start_trans=int(fids.shape[0]/4)                            # number of transient to start at

    fids, trans_affected = add_freq_drift_linear(fids=fids, time=time, freq_offset_var=off_var, freq_shift=slope_var, start_trans=start_trans, num_trans=num_affected_trans, echo=echo)

    return fids, trans_affected


def add_subtle_motion_artifact(fids, time, echo=False):
    '''
    To add a small frequency and phase shifts to certain transients to mimic some participant motion
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing ppm values [spec_points] 
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] **with subtle motion(s) artifact inserted**
                trans_fs (list of integers): list of transient numbers where frequency shift artifact(s) have been added
                trans_ps (list of integers): list of transient numbers where phase shift artifact(s) have been added
    '''   
    num_affected_trans=1                                        # number of affected transients
    freq_shift_var=5                                            # +/- range of frequency shifts
    phase_shift_var=15                                          # +/- range of phase shifts

    fids, trans_fs = add_freq_shift(fids=fids, time=time, freq_var=freq_shift_var, cluster=False, num_trans=num_affected_trans, echo=echo)
    fids, trans_ps = add_zero_order_phase_shift(fids=fids, phase_var=phase_shift_var, cluster=False, num_trans=num_affected_trans, echo=echo)

    return fids, [trans_fs, trans_ps]


def add_disruptive_motion_artifact(fids, time, ppm, mot_locs=None, nmb_motion=1, echo=False):
    '''
    To add a linebroadening and baseline changes to mimic large participant motion
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing time values [spec_points] 
                ppm (float): vector containing ppm values [spec_points] 
                mot_locs (list of integers): list of transient numbers where disruptive artifact(s) have been added
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] **with disruptive motion(s) artifact inserted**
                mot_locs (list of integers): list of transient numbers where disruptive artifact(s) have been added
    '''
    var = np.random.uniform(-0.005, 0.005, size=2)
    amp=[1.75, 1.75*(1+var[0])]                                 # amplitude
    damp=[18, 18*(1+var[1])]                                    # line broadening
    
    motion_profile = {                                          # profile for baseline
    "base_type": "SN",
    "num_bases": 3,
    "amp_bases": [0.0005, 0.001, 0.008],
    "comp_bases": [0.3, 0.7, 0.20],
    "base_var": 0.001,
    "slope_bases": [-0.00001, 0.00001, -0.000015],
    "spline_fitted": True}

    fids, mot_locs = add_linebroad(fids=fids, time=time, amp=amp, damp=damp, mot_locs=mot_locs, nmb_motion=nmb_motion, echo=echo)
    fids, mot_locs = add_baseline(fids=fids, ppm=ppm, base_profile=motion_profile, cluster=False, nbase_locs=mot_locs, num_trans=nmb_motion, echo=echo)
    
    return fids, mot_locs


def add_lipid_artifact(fids, time, edited=1, cluster=False, lp_locs=None, nmb_lps=1, echo=False):
    '''
    To add lipid artifacts to a select number of transients
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing time values [spec_points]
                edited (float): indicates the percent difference from ON to OFF (between 0.01 - 1.99, where 1 indicates no difference) 
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                lp_locs (list of integers): list of transient numbers where lipid artifact(s) have been added
                nmb_lps (integer): number of lipid artifacts in scan
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] **with lipid artifact(s) inserted**
                lp_locs (list of integers): list of transient numbers where lipid artifact(s) have been added
    '''
    lipid_profile = {
    "peak_type": "G",
    "amp": [0.00012],
    "width": [2],    
    "res_freq": [1.5],
    "edited": edited}
    
    fids, lp_locs = add_nuisance_peak(fids=fids, time=time, peak_profile=lipid_profile, cluster=cluster, np_locs=lp_locs, num_trans=nmb_lps, echo=echo)

    return fids, lp_locs