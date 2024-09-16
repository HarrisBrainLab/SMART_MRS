"""artifact module (SMART_MRS, 2024, Bugler et al)

These functions allow the user to apply various artifact to their data.

These functions primarily require fids [num_samples, spec_points], time[spec_points], and ppm[spec_points] data.

These functions require the support module and the math, random, Scipy and NumPy libraries.

This file can also be imported as a module and contains the following functions:

    * add_time_domain_noise - returns FIDs with complex noise applied to the real and imaginary data independently 
    * add_spur_echo_artifact - returns FIDs with applied spurious echo(es) and a list of transient(s) affected
    * add_eddy_current_artifact - returns FIDs with applied eddy current(s) and a list of transient(s) affected
    * add_linebroad - returns FIDs with applied line broadening and a list of transient(s) affected
    * add_nuisance_peak - returns FIDs with applied nuisance peak(s) and a list of transient(s) affected
    * add_baseline - returns FIDs with applied baseline(s) and a list of transient(s) affected
    * add_freq_drift_linear - returns FIDs with applied linear frequency drift and a list of the first and last affected transient
    * add_freq_shift - returns FIDs with applied frequency shift(s) and a list of transient(s) affected
    * add_zero_order_phase_shift - returns FIDs with applied zero order phase shift(s) and a list of transient(s) affected
    * add_first_order_phase_shift - returns FIDs with applied first order phase shift(s) and a list of transient(s) affected

"""

# import Python packages
import math
import random
import numpy as np
from scipy.interpolate import splrep, BSpline

# import functions from support module
from .support import to_fids, to_specs

# artifact module functions
def add_time_domain_noise(fids, noise_level=10):
    '''
    Add complex time domain noise
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                noise_level (float): standard deviation of noise level (with zero-mean noise)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with Gaussian White Noise**
    '''
    fids = (fids.real + np.random.normal(0, noise_level, size=(fids.shape))) + \
           (fids.imag + np.random.normal(0, noise_level, size=(fids.shape))) * 1j
    
    return fids


def add_spur_echo_artifact(fids, time, amp=None, lf=127, cs=None, phase=None, tstart=None, tfinish=None, cluster=False, gs_locs=None, nmb_sps=1, echo=False):
    '''
    To add a spurious echo artifact to a select number of transients 
    (Adapted from Berrington et al. 2021)
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing ppm values [spec_points] 
                amp (list of floats): amplitude of spurious echo artifact
                cs (list of floats): chemical shift in ppm
                lf (int/float): larmor frequency (default 127)
                phase (list of floats): phase of artifact in radians
                tstart (list of floats): start time (ms) of artifact in time domain
                tfinish (list of floats): end time (ms) of artifact in time domain
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                gs_locs (list of integers): list of transient numbers affected by spurious echoes
                nmb_sps (integer): number of spurious echo artifacts in scan
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] **with spurious echo(es) inserted**
                gs_locs (list of integers): list of transient numbers affected by spurious echoes
    '''
    func_def = []
    lf, t_all = (2*np.pi*lf), np.max(time, axis=0)

    # check for user vs. default values
    if gs_locs is None or len(gs_locs)!=nmb_sps:
        if cluster is False:
            gs_locs = np.random.choice(range(0, fids.shape[0]), size=nmb_sps, replace=False)
        else:
            start = int(np.random.uniform(0, ((fids.shape[0]/2)-nmb_sps), size=1))
            gs_locs = range(start, start+nmb_sps)
        func_def.append(f'Locations: {gs_locs}')

    if tstart is None or len(tstart)!=nmb_sps:
        tstart = np.random.uniform(600, 900, size=nmb_sps)
        func_def.append(f'Start Times: {tstart}')

    if (tfinish is None) or (len(tfinish)!=nmb_sps) or (any(np.array(tstart) >= np.array(tfinish))): 
        tfinish = tstart + np.random.uniform(100, (np.max(time)*1e+03-(tstart)), size=nmb_sps)        # amplitude is also dependent on length of time of spurious echo
        func_def.append(f'End Times: {tfinish}')
  
    if phase is None or len(phase)!=nmb_sps:
        phase = np.random.uniform(0.1, 1.9, size=nmb_sps)*math.pi
        func_def.append(f'Phases: {phase}')

    if amp is None or len(amp)!=nmb_sps:
        amp = np.random.uniform(50, 150, size=nmb_sps)
        func_def.append(f'Amplitudes: {amp},')

    if cs is None or len(cs)!=nmb_sps:
        cs = np.random.uniform(-3.0, 3.0, size=nmb_sps)
        func_def.append(f'Chemical Shifts: {cs}')
    # if lf is kept the same, remap cs values user value (equivalent spectrum value at current lf) [3(6.75), 2(5.75), 1(4.75), 0(3.75), -1(2.75), -2(1.75), -3(0.75)]
    else:
        cs = [5.65-x  for x in cs]
    
    # calculate / expand params for spurious echo artifact(s)
    start_ind, finish_ind = np.empty(shape=nmb_sps), np.empty(shape=nmb_sps)
    for ind in range (nmb_sps):
        start_ind[ind] = np.array(np.min(np.where(time>=(tstart[ind]*1e-03))))
        finish_ind[ind] = np.array(np.min(np.where(time>=(tfinish[ind]*1e-03))))

    gs_locs = np.array(gs_locs)
    phase = np.array(phase)[:, np.newaxis].repeat(time.shape[0], axis=1)
    amp = np.array(amp)[:, np.newaxis].repeat(time.shape[0], axis=1)
    cs = np.array(cs)[:, np.newaxis].repeat(time.shape[0], axis=1)
    time = time[np.newaxis, :].repeat(nmb_sps, axis=0)

    # insert spurious echo artifact(s)
    for ii in range(0, nmb_sps):
        s, f = int(start_ind[ii]),int(finish_ind[ii])
        fids[gs_locs[ii], s:f] = fids[gs_locs[ii], s:f]*amp[ii, s:f]*np.exp(-abs(time[ii, s:f])/t_all)*np.exp(1j*(lf*(1-cs[ii, s:f])*time[ii, s:f]+phase[ii, s:f]))
        
    if echo is True:
        print(f'Non-user defined parameters for "add_spur_echo_artifact": {func_def}')

    return fids, np.sort(gs_locs)


def add_eddy_current_artifact(fids, time, amp=None, tc=None, cluster=False, ec_locs=None, nmb_ecs=1, echo=False):
    '''
    To add an Eddy Current artifact into specified number of transient 
    (Adapted from Eddy Current Artifact in FID-A (Simpson et al. 2017))
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing ppm values [spec_points] 
                amp (list of floats): amplitude of eddy current artifact
                tc (list of floats): time constant of eddy current artifact
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                ec_locs (list of integers): list of transient numbers affected by eddy currents
                nmb_ecs (integer): number of eddy current artifacts in scan
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] **with EC artifact(s) inserted**, 
                ec_locs (list of integers): list of transient numbers affected by eddy currents
    '''
    func_def = []

    # check for user vs. default values
    if ec_locs is None or len(ec_locs)!=nmb_ecs:
        if cluster is False:
            ec_locs = np.random.choice(range(0, fids.shape[0]), size=nmb_ecs, replace=False)
        else:
            start = int(np.random.uniform(0, ((fids.shape[0]/2)-nmb_ecs), size=1))
            ec_locs = np.array(range(start, start+nmb_ecs))
        func_def.append(f'Locations: {ec_locs}')

    if amp is None or len(amp)!=nmb_ecs:
        amp = np.random.uniform(0.10, 0.500, size=nmb_ecs)
        func_def.append(f'Amplitudes: {amp}')

    if tc is None or len(tc)!=nmb_ecs:      
        tc = np.random.uniform(0.09, 0.5, size=nmb_ecs)
        func_def.append(f'Time Constants: {tc}')

    # calculate / expand params for eddy current artifact(s)
    amp = np.array(amp)[:, np.newaxis].repeat(time.shape[0], axis=1)
    tc = np.array(tc)[:, np.newaxis].repeat(time.shape[0], axis=1)
    time = time[np.newaxis, :].repeat(nmb_ecs, axis=0)

    # insert eddy current artifact(s)
    fids[ec_locs, :] = fids[ec_locs, :] * (np.exp(-1j * time * (amp * np.exp(-time / tc)) * 2 * math.pi))
    
    if echo is True:
        print(f'Non-user defined parameters for "add_eddy_current_artifact": {func_def}')
    
    return fids, np.sort(ec_locs)


def add_linebroad(fids, time, amp=None, damp=None, cluster=False, mot_locs=None, nmb_motion=1, echo=False):
    '''
    Add line broadening artifact
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing ppm values [spec_points] 
                amp (list of floats): amplitude of decaying exponential used for line broadening
                damp (list of floats): degree of line broadening where larger values indicate faster decay (dampening coefficient)
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                mot_locs (list of integers): list of transient numbers affected by line broadening
                nmb_motion (integer): number of line broadening artifacts in scan
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with linebroadening artifact**
                mot_locs (list of integers): list of transient numbers affected by line broadening
    '''
    func_def = []

    # check for user vs. default values
    if mot_locs is None or len(mot_locs)!=nmb_motion:
        if cluster is False:
            mot_locs = np.random.choice(range(0, fids.shape[0]), size=nmb_motion, replace=False)
        else:
            start = int(np.random.uniform(0, ((fids.shape[0]/2)-nmb_motion), size=1))
            mot_locs = np.array(range(start, start+nmb_motion))
        func_def.append(f'Locations: {mot_locs}')

    if amp is None or len(amp)!=nmb_motion:
        amp = np.random.uniform(1.500, 2.250, size=nmb_motion)
        func_def.append(f'Amplitudes: {amp}')

    if damp is None or len(damp)!=nmb_motion:
        damp = np.random.uniform(5, 50, size=nmb_motion)
        func_def.append(f'Lineshape Variance: {damp}')

    # calculate / expand params for line broadening artifact(s)
    amp = np.array(amp)[:, np.newaxis].repeat(time.shape[0], axis=1)
    damp = np.array(damp)[:, np.newaxis].repeat(time.shape[0], axis=1)
    time = time[np.newaxis, :].repeat(nmb_motion, axis=0)

    # insert line broadening artifact(s)
    fids[mot_locs, :] = fids[mot_locs, :] * (amp * np.exp(-time * damp))
    
    if echo is True:
        print(f'Non-user defined parameters for "add_linebroad": {func_def}')

    return fids, np.sort(mot_locs)


def add_nuisance_peak(fids, time, peak_profile, cluster=None, np_locs=None, num_trans=1, echo=False):
    '''
    Design the shape of and add a nuisance peak (i.e. lipid peak) to the spectrum
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing ppm values [spec_points] 
                peak_profile (dictionnary): containing peak elements below
                    - peak_type (string): should be specified as "G" (Gaussian), "L" (Lorentzian), "V" (Voigt)
                    - amp (list of floats): amplitude for each multiplet (recommended to remain between 5e+06 and 2e+4 for normalized ground truth data)
                    - width (list of floats): FWHM of each multiplet in ppm (will follow same order as amp)
                    - res_freq (list of floats): location (center) of each multiplet in ppm (will follow same order as amp)
                    - edited (float): indicates the percent difference from ON to OFF (between 0.01 - 1.99, where 1 indicates no difference) 
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                np_locs (list of integers): list of transient numbers affected by nuisance peaks
                numTrans (integer): number of nuisance peaks artifacts in scan
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with nuisance peak added **
                np_locs (list of integers): list of transient numbers affected by nuisance peaks
    '''
    func_def = []

    # check for user vs. default values
    if np_locs is None or len(np_locs)!=num_trans:
        if cluster is False:
            np_locs = np.sort(np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False))
        else:
            start = random.randint(0, int((fids.shape[0]/2)-num_trans))
            np_locs = np.array(range(start, start+num_trans))
        func_def.append(f'Locations: {np_locs}')

    if peak_profile["peak_type"] is None:
        peak_type = "G"
        func_def.append(f'Peak Type: {peak_type}')
    else:
        peak_type = peak_profile["peak_type"]

    if peak_profile["amp"] is None or (len(peak_profile["amp"])!= num_trans and len(peak_profile["amp"])!=1):
        peak_profile["amp"] = np.random.uniform(0.00001, 0.0002, size=num_trans)
        func_def.append(f'Amplitude of Multiplets: {peak_profile["amp"]}')
    amp = np.array(peak_profile["amp"] ).repeat(repeats=num_trans)

    if (peak_profile["edited"] is not None and peak_profile["edited"] != 1):
        for ii in range(0, num_trans):
            if ii%2==0:
                amp[ii] = amp[ii]*peak_profile["edited"]                                                                        # difference between ON and OFF
    
    if peak_profile["width"] is None or (len(peak_profile["width"])!= num_trans and len(peak_profile["width"])!=1):
        peak_profile["width"] = np.random.uniform(0.001, 0.5, size=num_trans)
        func_def.append(f'Width of Multiplets: {peak_profile["width"]}')
    else:
        peak_profile["width"] = [ppm_width/5 for ppm_width in peak_profile["width"]]
    width = (1 / (1600 * np.array(peak_profile["width"]))).repeat(repeats=num_trans)                                             # 1/300*FWHM = sigma
    
    if peak_profile["res_freq"] is None or (len(peak_profile["res_freq"])!= num_trans and len(peak_profile["res_freq"])!=1):
        peak_profile["res_freq"] = np.repeat(np.random.uniform(0, 7), repeats=num_trans)
        func_def.append(f'Peak Locations: {peak_profile["res_freq"]}')
    res_freqs = (127.7 * np.array(peak_profile["res_freq"]) - 383.45).repeat(repeats=num_trans)                                 # 127.7*ppm-383.45 = mu
    
    trans = 0
    frac = np.random.uniform(0.1, 0.9)
    for trans_loc in np_locs:
        if peak_profile["peak_type"] == 'G':
            peak_shape = np.exp(2*math.pi*time*res_freqs[trans]*1j)*amp[trans]*(1/width[trans]*np.sqrt(2*math.pi))*np.exp(-np.power((time),2)/(2*np.power(width[trans],2)))
        elif peak_profile["peak_type"] == 'L':
            peak_shape = np.exp(2*math.pi*time*res_freqs[trans]*1j)*(amp[trans]/math.pi)*((0.5*width[trans])/(np.power(time,2)+np.power((0.5*width[trans]),2)))
        else:
            peak_time_gauss = np.exp(2*math.pi*time*res_freqs[trans]*1j)*amp[trans]*(1/width[trans]*np.sqrt(2*math.pi))*np.exp(-np.power((time),2)/(2*np.power(width[trans],2)))
            peak_time_lorentz = np.exp(2*math.pi*time*res_freqs[trans]*1j)*(amp[trans]/math.pi)*((0.5*width[trans])/(np.power(time,2)+np.power((0.5*width[trans]),2)))
            peak_shape = (peak_time_gauss*frac) + ((1-frac)*peak_time_lorentz)
        trans+=1
        fids[trans_loc, :] = fids[trans_loc, :] + peak_shape
    
    if echo is True:
        print(f'Non-user defined parameters for "add_nuisance_peak": {func_def}')

    return fids, np_locs


def add_baseline(fids, ppm, base_profile, cluster=None, nbase_locs=None, num_trans=1, echo=False):
    '''
    Design a wavering baseline to be added to the spectrum
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                ppm (float): vector containing ppm values [spec_points] 
                base_profile (dictionnary): containing baseline elements below
                    - base_type (list of strings): should be specified as "SN" (sinewave - default) or "SC" (sinc) ** Will get SPLINE Combined**
                    - num_bases (list of integers): number of baselines (default is 1)
                    - amp_bases (list of floats): amplitude for each baseline
                    - comp_bases (list of floats): width/compression of baseline (will follow same order as amp_bases)
                    - base_var (float): variance between bases
                    - slope_bases (list of floats): slope of mean of baseline to be added (will follow same order as amp_bases) (list - if empty, default is no slope)
                    - spline_fitted (boolean): spline fit of the combined bases
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                nbase_locs (list of integers): list of transient numbers affected by baseline changes
                numTrans (integer): number of baseline contamination artifacts in scan
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with baseline contamination**
                nbase_locs (list of integers): list of transient numbers affected by baseline changes
    '''
    func_def = []
    specs = to_specs(fids)

    # check for user vs. default values
    if nbase_locs is None or len(nbase_locs)!=num_trans:
        if cluster is False:
            nbase_locs = np.sort(np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False))
        else:
            start = random.randint(0, int((fids.shape[0]/2)-num_trans))
            nbase_locs = np.array(range(start, start+num_trans))
        func_def.append(f'Transient Locations: {nbase_locs}')

    if base_profile["base_type"] is None:
        base_type = "SN"
        func_def.append(f'Base Type: {base_type}')
    else:
        base_type = base_profile["base_type"]

    if base_profile["num_bases"] is None:
        num_bases = 2
        func_def.append(f'Number of Bases: {num_bases}')
    else:
        num_bases = int(base_profile["num_bases"])

    if base_profile["amp_bases"] is None or len(base_profile["amp_bases"])!= num_bases:
        base_profile["amp_bases"] = np.random.uniform(0, 1, size=num_bases)
        func_def.append(f'Amplitude of Bases: {base_profile["amp_bases"]}')

    amp_bases = np.array(base_profile["amp_bases"]) 
    amp_bases = amp_bases[np.newaxis, :].repeat(num_trans, axis=0)

    if base_profile["base_var"] is None:
        base_profile["base_var"] = 0.0001
        func_def.append(f'Base Variation: {base_profile["base_var"]}')

    base_var = np.random.normal(-1*base_profile["base_var"], base_profile["base_var"], size=(num_trans, num_bases))
    amp_bases = amp_bases*(1+base_var)

    if base_profile["comp_bases"] is None or len(base_profile["comp_bases"])!= num_bases:
        base_profile["comp_bases"] = np.random.uniform(0, 0.8, size=num_bases)
        func_def.append(f'Compression of Bases: {base_profile["comp_bases"]}')
    
    comp_bases = np.array(base_profile["comp_bases"])
    comp_bases = comp_bases[np.newaxis, :]
    comp_bases = comp_bases*(1+base_var)

    if base_profile["slope_bases"] is None:
        slope_bases = np.array(np.repeat([0], num_bases))
        slope_bases = np.repeat(slope_bases[np.newaxis, :], num_trans, axis=0)
    else:
        if len(base_profile["slope_bases"])!= num_bases:
            base_profile["slope_bases"] = np.random.uniform(0, 0.8, size=num_bases)
            func_def.append(f'Slope of Bases: {base_profile["slope_bases"]}')
        slope_bases = np.array(base_profile["slope_bases"])
        slope_bases = slope_bases[np.newaxis, :]
        slope_bases = slope_bases*(1+base_var)

    # calculate / expand params for baseline contamination artifact(s)
    trans_nbs = 0
    for trans in nbase_locs:
        base_shapes = np.zeros(shape=(num_bases, len(ppm)))
        for bases in range(0, base_shapes.shape[0]):
            phase = random.uniform(0, 2)*math.pi
            if base_type == "SC":   # Sinc
                base_shapes[bases, :] = (amp_bases[trans_nbs, bases]*((np.sin(comp_bases[trans_nbs, bases]*ppm-phase))/(comp_bases[trans_nbs, bases]*ppm-phase))+slope_bases[trans_nbs, bases]*ppm)
            ## if undefined at 0, use piece-wise below without phase term
                # ppm_conds = [ppm >= 0.1, (ppm < 0.1) & (ppm > -0.1), ppm <= -0.1]
                # sinc_funcs = [lambda ppm: (amp_bases[trans_nbs, bases] * np.sin(comp_bases[trans_nbs, bases]*ppm)/(math.pi*ppm) + (slope_bases[trans_nbs, bases]*ppm)), lambda ppm: 1, lambda ppm: (amp_bases[trans_nbs, bases] * np.sin(comp_bases[trans_nbs, bases]*ppm)/(math.pi*ppm) + (slope_bases[trans_nbs, bases]*ppm))]
                # base_shapes[bases, :] = np.piecewise(ppm, ppm_conds, sinc_funcs)
            else:                   # Sine
                base_shapes[bases, :] = amp_bases[trans_nbs, bases] * np.sin(comp_bases[trans_nbs, bases]*ppm-phase)+(slope_bases[trans_nbs, bases]*ppm)
        trans_nbs+=1

        if base_profile["spline_fitted"] is True:
            coeffs = splrep(x=ppm, y=np.sum(base_shapes, axis=0))
            spline = BSpline(coeffs[0], coeffs[1], coeffs[2])
            all_bases = spline(ppm)
        else:
            all_bases = np.sum(base_shapes, axis=0)

        # insert baseline contamination artifact(s)
        specs[trans, :] = specs[trans, :] + all_bases

    if echo is True:
        print(f'Non-user defined parameters for "add_baseline": {func_def}')

    return to_fids(specs), nbase_locs


def add_freq_drift_linear(fids, time, freq_offset_var=None, freq_shift=None, start_trans=None, num_trans=1, echo=False):
    '''
    Add linear frequency drift
    Default values used were based on (DOI: 10.1002/mrm.25009, Harris et al. (2014) Impact of frequency drift on gamma-aminobutyric acid-edited MR spectroscopy)
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing ppm values [spec_points] 
                freq_offset_var (integer): variance (Hz) at each step within the drift
                freq_shift (integer): overall frequency shift (Hz) to accomplish between first and last transient
                start_trans (integer): number of the first transient affected by the drift
                numTrans (integer): number of transients affected by the frequency drift
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with frequency drift**
                start_trans (integer): number of the first transient affected by the drift
                numTrans (integer): number of transients affected by the frequency drift
    '''
    func_def = []

    # check for user vs. default values
    if num_trans is None:
        num_trans = int(np.random.uniform(2, (fids.shape[0]-1)))
        func_def.append(f'Number of Transients Affected: {num_trans}')

    if start_trans is None or not(isinstance(start_trans, int)):
        start_trans = int(np.random.uniform(0, (fids.shape[0]-num_trans)))
        func_def.append(f'First Transient: {start_trans}')

    if freq_offset_var is None:
        freq_offset_var = 0
        func_def.append(f'Offset Variation: {freq_offset_var}')

    if freq_shift is None:
        # shift -15 to +15 per 200 transients (0.075 Hz/trans)
        freq_shift = np.random.uniform(-0.075*num_trans, 0.075*num_trans)
        func_def.append(f'Overall Frequency Shift: {freq_shift}')

    # calculate / expand params for frequency drift
    end_trans = start_trans+num_trans
    slope = np.linspace(start=freq_shift/num_trans, stop=freq_shift, num=num_trans)
    f_shift_linear = np.random.normal(0, freq_offset_var, size=num_trans) + slope
    f_shift_linear = f_shift_linear[:, np.newaxis].repeat(2048, axis=1)
    time = time[np.newaxis, :].repeat(num_trans, axis=0)

    # insert frequency drift
    fids[start_trans:end_trans, :] = fids[start_trans:end_trans, :]*np.exp(-1j*time*f_shift_linear*2*math.pi)
    
    if echo is True:
        print(f'Non-user defined parameters for "add_freq_drift_linear": {func_def}')
    
    return fids, [start_trans, num_trans]


def add_freq_shift(fids, time, freq_var=None, cluster=False, shift_locs=None, num_trans=1, echo=False):
    '''
    Add frequency shifts
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing ppm values [spec_points] 
                freq_var (integer): +/- range of frequency shifts 
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                shift_locs (list of integers): list of transient numbers affected by frequency shifts
                numTrans (integer): number of frequency shift artifacts in scan
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with random frequency shift**
                numTrans(number of transients affected): integer 
    '''
    func_def = []

    # check for user vs. default values
    if shift_locs is None or len(shift_locs)!=num_trans:
        if cluster is False:
            shift_locs = np.sort(np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False))
        else:
            start = int(np.random.uniform(0, ((fids.shape[0]/2)-num_trans), size=1)) 
            shift_locs = np.array(range(start, start+num_trans))
        func_def.append(f'Transient Locations: {shift_locs}')

    if freq_var is None:
        freq_var = np.random.uniform(2, 20, size=1)
        func_def.append(f'Frequency Shift Variance: {freq_var}')

    # calculate / expand params for frequency shifts
    f_shift = np.random.uniform(low=-abs(freq_var), high=freq_var, size=(num_trans, 1)).repeat(fids.shape[1], axis=1)
    time = time[np.newaxis, :].repeat(num_trans, axis=0)

    # insert frequency shifts
    fids[shift_locs, :] = fids[shift_locs, :] * np.exp(-1j * time * f_shift * 2 * math.pi)

    if echo is True:
        print(f'Non-user defined parameters for "add_freq_shift_random": {func_def}')

    return fids, np.sort(shift_locs)


def add_zero_order_phase_shift(fids, phase_var=None, cluster=False, shift_locs=None, num_trans=1, echo=False):
    '''
    Add zero order phase shifts
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                phase_var (integer): +/- range of phase shifts 
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                shift_locs (list of integers): list of transient numbers affected by phase shifts
                numTrans (integer): number of phase shift artifacts in scan
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with random phase shifts**
                shift_locs (list of integers): list of transient numbers affected by phase shifts
    '''
    func_def = []

    # check for user vs. default values
    if shift_locs is None or len(shift_locs)!=num_trans:
        if cluster is False:
            shift_locs = np.sort(np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False))
        else:
            start = int(np.random.uniform(0, ((fids.shape[0]/2)-num_trans), size=1)) 
            shift_locs = np.array(range(start, start+num_trans))
        func_def.append(f'Transient Locations: {shift_locs}')

    if phase_var is None:
        phase_var = np.random.uniform(5, 90, size=1)
        func_def.append(f'Phase Shift Variance: {phase_var}')

    # calculate / expand params for phase shifts
    p_noise = np.random.uniform(low=-abs(phase_var), high=phase_var, size=(num_trans, 1)).repeat(fids.shape[1], axis=1)

    # insert phase shifts
    fids[shift_locs, :] = fids[shift_locs, :] * np.exp(-1j * p_noise * math.pi / 180)

    if echo is True:
        print(f'Non-user defined parameters for "add_zero_order_phase_shift": {func_def}')

    return fids, np.sort(shift_locs)


def add_first_order_phase_shift(fids, ppm, shift=None, cluster=False, shift_locs=None, num_trans=1, echo=False):
    '''
    Add first order phase shifts
    (Adapted from phase1 in FID-A (Simpson et al. 2017))
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                ppm (float): vector containing ppm values [spec_points] 
                shift (float): time constant in ms used to calculate first order shifts
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                shift_locs (list of integers): list of transient numbers affected by phase shifts
                numTrans (integer): number of phase shift artifacts in scan
                echo (boolean): indicates whether to print which default function values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with random phase shifts**
                shift_locs (list of integers): list of transient numbers affected by phase shifts
    '''
    func_def = []

    # check for user vs. default values
    if shift_locs is None or len(shift_locs)!=num_trans:
        if cluster is False:
            shift_locs = np.sort(np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False))
        else:
            start = int(np.random.uniform(0, ((fids.shape[0]/2)-num_trans), size=1)) 
            shift_locs = np.array(range(start, start+num_trans))
        func_def.append(f'Transient Locations: {shift_locs}')

    if shift is None:
        shift = np.random.uniform(0.001, 1, size=1)
        func_def.append(f'Time shift: {shift}')

    # calculate frequency from ppm (assumed to be for B0 of 3T)
    freq = (ppm-np.median(ppm))*42.577*3

    # insert phase shifts
    for ii in shift_locs:
        p_noise = freq*shift*2*math.pi
        fids[ii, :] = fids[ii, :] * np.exp(-1j * p_noise * math.pi / 180)

    if echo is True:
        print(f'Non-user defined parameters for "add_first_order_phase_shift": {func_def}')

    return fids, np.sort(shift_locs)