"""
SMART_MRS Artifact Simulation Toolbox by Hanna Bugler, Amirmohammad Shamaei, Roberto Souza and Ashley Harris (2025)
To generate transients with commonly occuring artifacts from simulated ground truths
Default function parameter values are written for normalized frequency domain spectra
Artifact Functions: L18-L696, Applied Functions: L697-L811, IO Functions: L812-L1241, Support Functions: L1242-eod
[last upd. 2025-01-24]
"""

import math
import random
import json
import numpy as np
import scipy.io as sci_io
from scipy.interpolate import splrep, BSpline, splev
from scipy.fftpack import fft, ifft, fftshift
import nibabel as nib

########################################################################################################################
# Artifact functions
########################################################################################################################
def add_time_domain_noise(fids, noise_level=0.00005):
    '''
    Add complex time domain noise
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                noise_level (float): standard deviation of noise level (with zero-mean noise)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with Gaussian White Noise**
    '''
    fids = (fids.real + np.random.normal(0, noise_level, size=(fids.shape))) + \
           (fids.imag + np.random.normal(0, noise_level, size=(fids.shape))) * 1j
    
    return fids


def add_spur_echo_artifact(fids, time, amp=None, cs=None, phase=None, t_echo=None, cf_ppm=4.65, lf=127, locs=None, num_trans=None, cluster=False, echo=False):  
    '''
    To add a spurious echo artifact to a select number of transients
    (Adapted from Berrington et al. 2021)
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing time values [spec_points] 
                amp (list of floats): amplitude of spurious echo artifact (earlier start time will increase amp, longer echo will create wider echo)
                cs (list of floats): chemical shift in ppm
                phase (list of floats): phase of artifact in radians
                t_echo (list of floats): fraction of time of the total FID in ms
                cf_ppm (integer/float): center frequency in ppm (default is 4.65 ppm)
                lf (int/float): larmor frequency in MHz (default is 127 MHz)
                locs (list of integers): list of transient numbers affected by spurious echoes
                num_trans (integer): number of spurious echo artifacts in scan
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                echo (boolean): indicates whether to print which values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] **with spurious echo(es) inserted**
                locs (list of integers): list of transient numbers affected by spurious echoes
    '''
    func_def = []
    t_all = np.max(time, axis=0)

    # check for user vs. default values
    if locs is None:
        if num_trans is None:
            # default to all transients
            num_trans = fids.shape[0]
            locs = range(0, fids.shape[0])
        else:
            # default to user number of transients
            if cluster is True:
                start = int(np.random.uniform(0, ((fids.shape[0]/2)-num_trans), size=1))
                locs = range(start, start+num_trans)
            else:
                locs = np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False)
    else:
        # default to user locations
        num_trans = len(locs)
    func_def.append(f'Number of Transients: {num_trans}')
    func_def.append(f'Locations: {locs}')
    
    # other params
    if phase is None or len(phase)!=num_trans:
        phase = np.random.uniform(0.1, 1.9, size=num_trans)*math.pi
    func_def.append(f'Phases: {phase}')

    if amp is None or len(amp)!=num_trans:
        amp = np.random.uniform(50, 150, size=num_trans)
    func_def.append(f'Amplitudes: {amp},')
    
    if t_echo is None or len(t_echo)!=num_trans:
        t_echo = np.random.uniform(0.1, 0.9, size=num_trans)
    func_def.append(f'Time Fraction: {t_echo},')

    if cs is None or len(cs)!=num_trans:
        cs = np.random.uniform(0.0, 6.0, size=num_trans) 
        func_def.append(f'Chemical Shifts: {cs}')
    cs = [((cf_ppm - x)*2*math.pi*lf)  for x in cs]

    # insert spurious echo artifact(s)
    for ii in range(0, num_trans):
        echo_artif = amp[ii] * np.exp(-abs(time-t_echo[ii])/t_all) * np.exp(1j*((1-cs[ii])*time+phase[ii]))
        fids[locs[ii]] = fids[locs[ii]] + echo_artif
        
    if echo is True:
        print(f'Non-user defined parameters for "add_spur_echo_artifact": {func_def}')

    return fids, np.sort(locs)


def add_eddy_current_artifact(fids, time, amp=None, tc=None, locs=None, num_trans=None, cluster=False, echo=False):
    '''
    To add an Eddy Current artifact into specified number of transient 
    (Adapted from Eddy Current Artifact in FID-A (Simpson et al. 2017))
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing time values [spec_points] 
                amp (list of floats): amplitude of eddy current artifact
                tc (list of floats): time constant of the eddy current artifact
                locs (list of integers): list of transient numbers affected by eddy currents
                num_trans (integer): number of eddy current artifacts in scan
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                echo (boolean): indicates whether to print which values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] **with EC artifact(s) inserted**, 
                locs (list of integers): list of transient numbers affected by eddy currents
    '''
    func_def = []

    # check for user vs. default values
    if locs is None:
        if num_trans is None:
            # default to all transients
            num_trans = fids.shape[0]
            locs = range(0, fids.shape[0])
        else:
            # default to user number of transients
            if cluster is True:
                start = int(np.random.uniform(0, ((fids.shape[0]/2)-num_trans), size=1))
                locs = range(start, start+num_trans)
            else:
                locs = np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False)
    else:
        # default to user locations
        num_trans = len(locs)
    func_def.append(f'Number of Transients: {num_trans}')
    func_def.append(f'Locations: {locs}')

    # other params
    if amp is None or len(amp)!=num_trans:
        amp = np.random.uniform(1, 12, size=num_trans)
    func_def.append(f'Amplitudes: {amp}')

    if tc is None or len(tc)!=num_trans:      
        tc = np.random.uniform(0.001, 0.30, size=num_trans)
    func_def.append(f'Time Constants: {tc}')

    # calculate / expand params for eddy current artifact(s)
    amp = np.array(amp)[:, np.newaxis].repeat(time.shape[0], axis=1)
    tc = np.array(tc)[:, np.newaxis].repeat(time.shape[0], axis=1)
    time = time[np.newaxis, :].repeat(num_trans, axis=0)

    # insert eddy current artifact(s)
    fids[locs, :] = fids[locs, :] * (np.exp(-1j * time * (amp * np.exp(-time / tc)) * 2 * math.pi))
    
    if echo is True:
        print(f'Non-user defined parameters for "add_eddy_current_artifact": {func_def}')
    
    return fids, np.sort(locs)


def add_nuisance_peak(fids, time, peak_profile, cf_ppm=4.65, lf=127, locs=None, num_trans=None, cluster=False, echo=False):
    '''
    Design the shape of and add a nuisance peak (i.e. lipid peak) to the spectrum
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing time values [spec_points] 
                peak_profile (dictionnary): containing peak elements below
                    - peak_type (string): should be specified as "G" (Gaussian), "L" (Lorentzian), "V" (Voigt)
                    - amp (list of floats): amplitude for each multiplet
                    - phase (list of floats): phase of the peak in radians (will follow same order as amp)
                    - width (list of floats): FWHM of each multiplet in ppm (will follow same order as amp)
                    - res_freq (list of floats): location (center) of each multiplet in ppm (will follow same order as amp)
                    - edited (float): indicates the percent difference of the amplitude from ON to OFF (between 0.01 - 1.99, where 1 indicates no difference) 
                cf_ppm (integer/float): center frequency in ppm (default is 4.65 ppm)
                lf (integer/float): larmor frequency in Mhz (default is 127 MHz)
                locs (list of integers): list of transient numbers affected by nuisance peaks
                num_trans (integer): number of nuisance peaks artifacts in scan
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                echo (boolean): indicates whether to print which values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with nuisance peak added **
                locs (list of integers): list of transient numbers affected by nuisance peaks
    '''
    func_def = []

    number_of_points = len(time)
    dt = time[1]/np.arange(number_of_points)[1]

     # check for user vs. default values
    if locs is None:
        if num_trans is None:
            # default to all transients
            num_trans = fids.shape[0]
            locs = range(0, fids.shape[0])
        else:
            # default to user number of transients
            if cluster is True:
                start = int(np.random.uniform(0, ((fids.shape[0]/2)-num_trans), size=1))
                locs = range(start, start+num_trans)
            else:
                locs = np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False)
    else:
        # default to user locations
        num_trans = len(locs)
    func_def.append(f'Number of Transients: {num_trans}')
    func_def.append(f'Locations: {locs}')

    # peak type
    if peak_profile["peak_type"] is None:
        peak_profile["peak_type"] = "G"
    func_def.append('Peak Type: "G"')

    # amplitude of the peak
    if peak_profile["amp"] is None:
        amp = np.random.uniform(0.000005, 0.0002, size=num_trans)
    elif len(peak_profile["amp"]) == 1:
        amp = np.repeat(peak_profile["amp"], repeats=num_trans)
    else:
        amp = np.array(peak_profile["amp"])

    if peak_profile["edited"] is not None:
        for ii in range(0, num_trans):
            if ii%2==0:
                amp[ii] = amp[ii]*peak_profile["edited"] 
    func_def.append(f'Amplitude of Multiplets: {amp}')

    # linewidth in ppm
    if peak_profile["width"] is None:
        width = np.random.uniform(0.01, 2, size=num_trans)
    elif len(peak_profile["width"]) == 1:
        width = np.repeat(peak_profile["width"], repeats=num_trans)
    else:
        width = np.array(peak_profile["width"])
    func_def.append(f'Width of Multiplets: {width}')

    # peak frequency
    if peak_profile["res_freq"] is None:
        peak_profile["res_freq"] = np.random.uniform(0, 7, size=num_trans)
    elif len(peak_profile["res_freq"]) == 1:
        peak_profile["res_freq"] = np.repeat(peak_profile["res_freq"], repeats=num_trans).tolist()
    res_freqs = np.array([(x-cf_ppm)*lf for x in peak_profile["res_freq"]])
    func_def.append(f'Peak Locations: {res_freqs}')

    # Peak Phase      
    if peak_profile["phase"] is None:
        phase = np.random.uniform(0, 2*math.pi, size=num_trans)
    elif len(peak_profile["phase"]) == 1:
        phase = np.repeat(peak_profile["phase"], repeats=num_trans)
    else:
        phase = np.array(peak_profile["phase"])
    func_def.append(f'Peak Phases: {phase}')

    # calculate / expand params for nuisance peak artifact(s) and insert into scan
    trans = 0
    for trans_loc in locs:
        if peak_profile["peak_type"] == 'G':
            T_2 = 2 * (1 / (width[trans] * lf * math.pi))
            M_0 = (dt * amp[trans]) / (T_2 * np.sqrt(math.pi/4))
            peak_shape = M_0 * np.exp(1j*(2 * math.pi * res_freqs[trans] * time + phase[trans])) * np.exp(-(time**2)/(T_2**2))
        elif peak_profile["peak_type"] == 'L':
            T_2 = 1 / (width[trans] * lf * math.pi)
            M_0 = dt * amp[trans] / T_2
            peak_shape = M_0 * np.exp(1j*(2 * math.pi * res_freqs[trans] * time + phase[trans])) * np.exp(-time/T_2)
        else:
            T_2_G = 2 * (1 / (width[trans] * lf * math.pi))
            M_0_G = (dt * amp[trans]) / (T_2_G * np.sqrt(math.pi/4))
            T_2_L = 1 / (width[trans] * lf * math.pi)
            M_0_L = dt * amp[trans] / T_2_L
            frac = np.random.uniform(0.1, 0.9)          # fraction attributed to "G" vs. "L" (variable eta)
            peak_time_gauss = M_0_G * np.exp(1j*(2 * math.pi * res_freqs[trans] * time + phase[trans])) * np.exp(-(time**2)/(T_2_G**2))
            peak_time_lorentz = M_0_L * np.exp(1j*(2 * math.pi * res_freqs[trans] * time + phase[trans])) * np.exp(-time/T_2_L)
            peak_shape = (peak_time_gauss*frac) + ((1-frac)*peak_time_lorentz)
        trans+=1
        fids[trans_loc, :] = fids[trans_loc, :] + peak_shape
    
    if echo is True:
        print(f'Non-user defined parameters for "add_nuisance_peak": {func_def}')

    return fids, locs


def add_baseline(fids, ppm, base_profile, locs=None, num_trans=None, cluster=None, echo=False):
    '''
    Design a wavering baseline to be added to the spectrum
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                ppm (float): vector containing ppm values [spec_points] 
                base_profile (dictionnary): containing baseline elements below
                    - base_type (list of strings): should be specified as "SN" (sinewave - default) or "SC" (sinc)
                    - num_bases (list of integers): number of baselines (default is 1)
                    - amp_bases (list of floats): amplitude for each baseline
                    - comp_bases (list of floats): width/compression of baseline (will follow same order as amp_bases)
                    - base_var (float): variance between bases
                    - slope_bases (list of floats): slope of mean of baseline to be added (will follow same order as amp_bases) (list - if None, default is no slope)
                    - spline_fitted (boolean): spline fit of the combined bases
                    [Spline Only]
                    - x_vals (list of floats): frequency values for spline interpolation
                    - y_vals (list of floats): amplitude values for spline interpolation
                    - def_all_points (boolean): whether non-central points ]-5, 5[ are provided in addition to central points
                locs (list of integers): list of transient numbers affected by baseline changes
                num_trans (integer): number of baseline contamination artifacts in scan
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                echo (boolean): indicates whether to print which values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with baseline contamination**
                locs (list of integers): list of transient numbers affected by baseline changes
    '''
    func_def = []
    specs = to_specs(fids)

    # check for user vs. default values
    if locs is None:
        if num_trans is None:
            # default to all transients
            num_trans = fids.shape[0]
            locs = range(0, fids.shape[0])
        else:
            # default to user number of transients
            if cluster is True:
                start = int(np.random.uniform(0, ((fids.shape[0]/2)-num_trans), size=1))
                locs = range(start, start+num_trans)
            else:
                locs = np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False)
    else:
        # default to user locations
        num_trans = len(locs)
    func_def.append(f'Number of Transients: {num_trans}')
    func_def.append(f'Locations: {locs}')

    # base type
    if base_profile["base_type"] != "SP":       # if not SPLINE (SP), is assumed to be a SINUSOIDAL (SN) OR SINC (SC)
        base_type = base_profile["base_type"]

        # number of combined functions for single baseline
        if base_profile["num_bases"] is None:
            num_bases = 2
        else:
            num_bases = int(base_profile["num_bases"])
        func_def.append(f'Number of Bases: {num_bases}')

        # amplitude of each function in the baseline
        if base_profile["amp_bases"] is None or len(base_profile["amp_bases"])!= num_bases:
            base_profile["amp_bases"] = np.random.uniform(0, 1, size=num_bases)
        amp_bases = np.array(base_profile["amp_bases"]) 
        amp_bases = amp_bases[np.newaxis, :].repeat(num_trans, axis=0)

        # variation of each baseline within the set
        if base_profile["base_var"] is None:
            base_profile["base_var"] = 0.0001
        base_var = np.random.normal(-1*base_profile["base_var"], base_profile["base_var"], size=(num_trans, num_bases))
        amp_bases = amp_bases*(1+base_var)
        func_def.append(f'Amplitude of Bases: {amp_bases}')
        func_def.append(f'Base Variation: {base_var}')

        # number of period of each function in the baseline
        if base_profile["comp_bases"] is None or len(base_profile["comp_bases"])!= num_bases:
            base_profile["comp_bases"] = np.random.uniform(0, 0.8, size=num_bases)
        comp_bases = np.array(base_profile["comp_bases"])
        comp_bases = comp_bases[np.newaxis, :]
        comp_bases = comp_bases*(1+base_var)
        func_def.append(f'Compression of Bases: {comp_bases}')

        # slopes of each function in the baseline
        if base_profile["slope_bases"] is None:
            slope_bases = np.array(np.repeat([0], num_bases))
            slope_bases = np.repeat(slope_bases[np.newaxis, :], num_trans, axis=0)
        else:
            if len(base_profile["slope_bases"])!= num_bases:
                base_profile["slope_bases"] = np.random.uniform(0, 0.8, size=num_bases)
            slope_bases = np.array(base_profile["slope_bases"])
            slope_bases = slope_bases[np.newaxis, :]
            slope_bases = slope_bases*(1+base_var)
        func_def.append(f'Slope of Bases: {slope_bases}')

        # calculate / expand params for baseline contamination artifact(s)
        trans_nbs = 0
        for trans in locs:
            base_shapes = np.zeros(shape=(num_bases, len(ppm)))
            for bases in range(0, base_shapes.shape[0]):
                phase = random.uniform(0, 2)*math.pi
                if base_type == "SC":   # Sinc
                    base_shapes[bases, :] = (amp_bases[trans_nbs, bases]*((np.sin(comp_bases[trans_nbs, bases]*ppm-phase))/(comp_bases[trans_nbs, bases]*ppm-phase))+slope_bases[trans_nbs, bases]*ppm)
                else:                   # Sine
                    base_shapes[bases, :] = amp_bases[trans_nbs, bases] * np.sin(comp_bases[trans_nbs, bases]*ppm-phase)+(slope_bases[trans_nbs, bases]*ppm)
            trans_nbs+=1
            
            # additional if spline fitted was selected (separate from spline baseline)
            if base_profile["spline_fitted"] is True:
                coeffs = splrep(x=ppm, y=abs(np.sum(base_shapes, axis=0)))
                spline = BSpline(coeffs[0], coeffs[1], coeffs[2])
                all_bases = spline(ppm)
            else:
                all_bases = np.sum(base_shapes, axis=0)

            # insert baseline contamination artifact(s)
            specs[trans, :] = (abs(specs[trans, :]) + all_bases) * np.exp(1j*np.angle(specs[trans, :]))

    else: # SPLINE
        num_bases = 0
        for trans in locs:
            
            if base_profile["def_all_points"]!=True:
                # define points outside of window for stability
                x_vals = np.array([[-20], [-15], [-10], [-5]] + base_profile["x_vals"] + [[10], [15], [20], [25]])
                y_vals = np.array([[0.0001], [0.0001], [0.0001], [0.0001]] + base_profile["y_vals"] + [[0.0001], [0.0001], [0.0001], [0.0001]])
            else:
                # all points provided
                x_vals = np.array(base_profile["x_vals"])
                y_vals = abs(np.array(base_profile["y_vals"]))

            # if not first transient in list
            if trans != locs[0]:
                base_var = np.random.uniform(0.01*base_profile["base_var"], base_profile["base_var"], size=(2, x_vals.shape[0]))
                x_vals = np.multiply(np.squeeze(x_vals), (1+base_var[0, :]))
                y_vals = np.multiply(np.squeeze(y_vals), (1+base_var[1, :]))

            func_def.append(f'Spline X-values: {x_vals}')
            func_def.append(f'Spline Y-values: {y_vals}')

            # interpolate piece-wise spline, apply to ppm data, and insert into scan
            tck_vals = splrep(x_vals, y_vals)
            spline_base = (splev(ppm, tck_vals))
            specs[trans, :] = (abs(specs[trans, :]) + spline_base) * np.exp(1j*np.angle(specs[trans, :]))
            num_bases += 1

    if echo is True:
        print(f'Non-user defined parameters for "add_baseline": {func_def}')

    return to_fids(specs), locs


def add_linebroad(fids, time, damp=None, locs=None, num_trans=None, cluster=False, echo=False):
    '''
    Add line broadening artifact
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing time values [spec_points] 
                damp (list of floats): dampening coefficient representing the desired increase of the FWHM in Hz
                locs (list of integers): list of transient numbers affected by line broadening
                num_trans (integer): number of line broadening artifacts in scan
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                echo (boolean): indicates whether to print which values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with linebroadening artifact**
                locs (list of integers): list of transient numbers affected by line broadening
    '''
    func_def = []

    # check for user vs. default values
    if locs is None:
        if num_trans is None:
            # default to all transients
            num_trans = fids.shape[0]
            locs = range(0, fids.shape[0])
        else:
            # default to user number of transients
            if cluster is True:
                start = int(np.random.uniform(0, ((fids.shape[0]/2)-num_trans), size=1))
                locs = range(start, start+num_trans)
            else:
                locs = np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False)
    else:
        # default to user locations
        num_trans = len(locs)
    func_def.append(f'Number of Transients: {num_trans}')
    func_def.append(f'Locations: {locs}')

    # other params
    if damp is None or len(damp)!=num_trans:
        damp = np.random.uniform(5, 50, size=num_trans)
    func_def.append(f'Lineshape Variance: {damp}')

    # calculate / expand params for line broadening artifact(s)
    damp = np.array(damp)[:, np.newaxis].repeat(time.shape[0], axis=1)
    time = time[np.newaxis, :].repeat(num_trans, axis=0)

    # insert line broadening artifact(s)
    fids[locs, :] = fids[locs, :] * (np.exp(-time * damp * np.pi))
    
    if echo is True:
        print(f'Non-user defined parameters for "add_linebroad": {func_def}')

    return fids, np.sort(locs)

    
def add_freq_drift_linear(fids, time, freq_offset_var=None, freq_shift=None, start_trans=None, num_trans=None, echo=False):
    '''
    Add linear frequency drift
    Default values used were based on (DOI: 10.1002/mrm.25009, Harris et al. (2014) Impact of frequency drift on gamma-aminobutyric acid-edited MR spectroscopy)
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing time values [spec_points] 
                freq_offset_var (integer): variance (Hz) at each step within the drift
                freq_shift (integer): overall frequency shift (Hz) to accomplish between first and last transient
                start_trans (integer): number of the first transient affected by the drift
                num_trans (integer): number of transients affected by the frequency drift
                echo (boolean): indicates whether to print which values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with frequency drift**
                start_trans (integer): number of the first transient affected by the drift
                numTrans (integer): number of transients affected by the frequency drift
    '''
    func_def = []

    # check for user vs. default values
    if num_trans is None:
        num_trans = fids.shape[0]
    func_def.append(f'Number of Transients Affected: {num_trans}')

    if start_trans is None or not(isinstance(start_trans, int)):
        start_trans = int(np.random.uniform(0, (fids.shape[0]-num_trans)))
    func_def.append(f'First Transient: {start_trans}')

    if freq_offset_var is None:
        freq_offset_var = 0.001
    func_def.append(f'Offset Variation: {freq_offset_var}')

    if freq_shift is None:
        # shift -15 to +15 per 200 transients (0.075 Hz/trans)
        freq_shift = np.random.uniform(-0.075*num_trans, 0.075*num_trans)
    func_def.append(f'Overall Frequency Shift: {freq_shift}')

    # calculate / expand params for frequency drift
    end_trans = start_trans+num_trans
    slope = np.linspace(start=freq_shift/num_trans, stop=freq_shift, num=num_trans)
    f_shift_linear = np.random.normal(0, freq_offset_var, size=num_trans) + slope
    f_shift_linear = f_shift_linear[:, np.newaxis].repeat(fids.shape[1], axis=1)
    time = time[np.newaxis, :].repeat(num_trans, axis=0)

    # insert frequency drift
    fids[start_trans:end_trans, :] = fids[start_trans:end_trans, :]*np.exp(-1j*time*f_shift_linear*2*math.pi)
    
    if echo is True:
        print(f'Non-user defined parameters for "add_freq_drift_linear": {func_def}')
    
    return fids, [start_trans, num_trans]


def add_freq_shift(fids, time, freq_var=None, dist="N", locs=None, num_trans=None, cluster=False, echo=False):
    '''
    Add frequency shifts
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing time values [spec_points] 
                freq_var (integer): +/- range of frequency shifts
                dist (string): "N" indicates normal distribution (default) while "U" indicates a uniform distribution
                locs (list of integers): list of transient numbers affected by frequency shifts
                numTrans (integer): number of frequency shift artifacts in scan
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                echo (boolean): indicates whether to print which values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with random frequency shift**
                numTrans(number of transients affected): integer 
    '''
    func_def = []

    # check for user vs. default values
    if locs is None:
        if num_trans is None:
            # default to all transients
            num_trans = fids.shape[0]
            locs = range(0, fids.shape[0])
        else:
            # default to user number of transients
            if cluster is True:
                start = int(np.random.uniform(0, ((fids.shape[0]/2)-num_trans), size=1))
                locs = range(start, start+num_trans)
            else:
                locs = np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False)
    else:
        # default to user locations
        num_trans = len(locs)
    func_def.append(f'Number of Transients: {num_trans}')
    func_def.append(f'Locations: {locs}')

    if freq_var is None:
        freq_var = np.random.uniform(2, 20, size=1)
    func_def.append(f'Frequency Shift Variance: {freq_var}')

    # calculate / expand params for frequency shifts
    if dist == "N":
        f_shift = np.random.normal(loc=0.0, scale=freq_var, size=(num_trans, 1)).repeat(fids.shape[1], axis=1)
    else:
        f_shift = np.random.uniform(low=-abs(freq_var), high=freq_var, size=(num_trans, 1)).repeat(fids.shape[1], axis=1)

    time = time[np.newaxis, :].repeat(num_trans, axis=0)

    # insert frequency shifts
    fids[locs, :] = fids[locs, :] * np.exp(-1j * time * f_shift * 2 * math.pi)

    if echo is True:
        print(f'Non-user defined parameters for "add_freq_shift_random": {func_def}')

    return fids, np.sort(locs)


def add_zero_order_phase_shift(fids, phase_var=None, dist="N", locs=None, num_trans=None, cluster=False, echo=False):
    '''
    Add zero order phase shifts
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                phase_var (integer): +/- range of phase shifts 
                dist (string): "N" indicates normal distribution (default) while "U" indicates a uniform distribution
                locs (list of integers): list of transient numbers affected by phase shifts
                num_trans (integer): number of phase shift artifacts in scan
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                echo (boolean): indicates whether to print which values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with random phase shifts**
                locs (list of integers): list of transient numbers affected by phase shifts
    '''
    func_def = []

    # check for user vs. default values
    if locs is None:
        if num_trans is None:
            # default to all transients
            num_trans = fids.shape[0]
            locs = range(0, fids.shape[0])
        else:
            # default to user number of transients
            if cluster is True:
                start = int(np.random.uniform(0, ((fids.shape[0]/2)-num_trans), size=1))
                locs = range(start, start+num_trans)
            else:
                locs = np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False)
    else:
        # default to user locations
        num_trans = len(locs)
    func_def.append(f'Number of Transients: {num_trans}')
    func_def.append(f'Locations: {locs}')

    if phase_var is None:
        phase_var = np.random.uniform(5, 90, size=1)
    func_def.append(f'Phase Shift Variance: {phase_var}')

    # calculate / expand params for phase shifts
    if dist == "N":
        p_noise = np.random.normal(loc=0.0, scale=phase_var, size=(num_trans, 1)).repeat(fids.shape[1], axis=1)
    else:
        p_noise = np.random.uniform(low=-abs(phase_var), high=phase_var, size=(num_trans, 1)).repeat(fids.shape[1], axis=1)

    # insert phase shifts
    fids[locs, :] = fids[locs, :] * np.exp(-1j * p_noise * math.pi / 180)

    if echo is True:
        print(f'Non-user defined parameters for "add_zero_order_phase_shift": {func_def}')

    return fids, np.sort(locs)


def add_first_order_phase_shift(fids, ppm, shift=None, dist="N", lf=127, cluster=False, locs=None, num_trans=None, echo=False):
    '''
    Add first order phase shifts
    (Adapted from phase1 in FID-A (Simpson et al. 2017))
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                ppm (float): vector containing ppm values [spec_points] 
                shift (float): time constant in ms used to calculate first order shifts
                dist (string): "N" indicates normal distribution (default) while "U" indicates a uniform distribution
                lf (integer/float): larmor frequency in MHz (default 127 MHz)
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                locs (list of integers): list of transient numbers affected by phase shifts
                num_trans (integer): number of phase shift artifacts in scan
                echo (boolean): indicates whether to print which values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with random phase shifts**
                locs (list of integers): list of transient numbers affected by phase shifts
    '''
    func_def = []

    # check for user vs. default values
    if locs is None:
        if num_trans is None:
            # default to all transients
            func_def.append(f'Number of Transients: {num_trans}')
            num_trans = fids.shape[0]
            locs = range(0, fids.shape[0])
        else:
            # default to user number of transients
            if cluster is True:
                start = int(np.random.uniform(0, ((fids.shape[0]/2)-num_trans), size=1))
                locs = range(start, start+num_trans)
            else:
                locs = np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False)
        func_def.append(f'Locations: {locs}')
    else:
        # default to user locations
        num_trans = len(locs)

    if shift is None:
        if dist == "N":
            shift = np.random.normal(0, 1, size=1)
        else:
            shift = np.random.uniform(0.001, 1, size=1)
        func_def.append(f'Time shift: {shift}')

    # calculate frequency from ppm
    freq = (ppm-np.median(ppm))*lf

    # insert phase shifts
    for ii in locs:
        p_noise = freq*shift*2*math.pi
        fids[ii, :] = fids[ii, :] * np.exp(-1j * p_noise * math.pi / 180)

    if echo is True:
        print(f'Non-user defined parameters for "add_first_order_phase_shift": {func_def}')

    return fids, np.sort(locs)


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

    if num_trans is None:                              # number of affected transients
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


########################################################################################################################
# IO Functions
########################################################################################################################
def get_FIDA_mat_data(dir_mat, struct_name):
    '''
    Load FIDs from an existing MATLAB .mat FID-A struct (will need to perform for x number of subspectra)
    :param:     dir_mat (string): directory and .mat filename (i.e. "C:/Users/FIDA/MyMatFile.mat")
                struct_name (string): name of struct contained within .mat file
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing time values [spec_points]
                ppm (float): vector containing ppm values [spec_points] 
    '''
    # from MATLAB, open .mat data
    sims = sci_io.loadmat(dir_mat)[struct_name]
    fids = []

    # obtain fids, time, and ppm
    for ii in range (0, sims.shape[1]):
        fids.append(sims[0, ii][0])
    time = np.squeeze(sims[0,0][1])
    ppm = np.squeeze(sims[0,0][2])[::-1]

    return np.squeeze(np.array(fids)), time, ppm


def return_FIDA_mat_data(dir_mat, struct_name, fids):
    '''
    Return changed FIDs to an existing MATLAB .mat FID-A struct (will need to perform for x number of subspectra)
    :param:     dir_mat (string): directory and .mat filename (i.e. "C:/Users/FIDA/MyMatFile.mat")
                struct_name (string): name of struct contained within .mat file
                fids (complex floats): free induction decay values of shape [num_samples, spec_points] *WITH ARTIFACTS*
    '''
     # from MATLAB, open .mat data
    sims = sci_io.loadmat(dir_mat)[struct_name]
    fids = fids[:, :, np.newaxis]

    # save existing .mat file with new fids
    for ii in range (0, sims.shape[1]):
        sims[0, ii][0] = fids[ii, :, :]
    
    og_dir = dir_mat.split(".")
    sci_io.savemat(f"{og_dir[0]}_SMART.mat", {struct_name: sims})


def get_nifti_mrs_data(dir_nifti, cf=4.65):
    '''
    Load single voxel FIDs from an existing nifti-mrs file (assumes standard dimension naming conventions as per Clarke et al. 2022
    where data is already coil combined)(dim_6 (DIM_DYN), dim_7 (DIM_EDIT))
    :param:     dir_nifti (string): directory and nifti-mrs filename (i.e. "C:/Users/FIDA/MyNiftiFile.nii.gz")
                cf (float): center frequency in ppm (assumed 4.65)
    :return:    fids (dictionary of complex floats): free induction decay values of shape [num_samples, spec_points] 
                time (float): vector containing time values [spec_points]
                ppm (float): vector containing ppm values [spec_points] 
    '''
    fids, fids_on, fids_off = [], [], []
    dim5, dim6, dim7 = 'NOT_ASSIGNED', 'NOT_ASSIGNED', 'NOT_ASSIGNED'
    spec_points_dim = 1                         # existing spec points dim
    unused_dims = 3                             # voxel location dims (assumed single voxel, unsused)
    

    # open nifti-mrs file
    nifti_load = nib.load(dir_nifti)
    nifti = nifti_load.get_fdata(dtype=np.complex64)
    og_num_dims = nifti.ndim
    nifti_header = nifti_load.header
    json_header = json.loads(nifti_load.header.extensions[nifti_load.header.extensions.get_codes().index(44)].get_content())

    # check if single voxel
    if nifti.shape[0] > 1 or nifti.shape[1] > 1 or nifti.shape[2] > 1:
        print('Data is not from single voxel.')
        return False

    # obtained by going from +ve to -ve bandwidth (of length = number of spec points) divided by the central frequency
    spec_points = nifti.shape[3]
    dwtime = nifti_header['pixdim'][4]
    freq = np.linspace(start=(-(1/dwtime)/2) + ((1/dwtime)/(2*spec_points)), stop=((1/dwtime)/2) - ((1/dwtime)/(2*spec_points)), num=spec_points)
    ppm = (-freq / json_header['SpectrometerFrequency'][0])+ cf
    time = np.linspace(start=0, stop=(spec_points-1)*dwtime, num=spec_points)

    # check for existence of dimensions 5-7
    if 'dim_5' in json_header:
        dim5 = json_header['dim_5']
        if og_num_dims == 5:
            nifti = nifti[0, 0, 0, :, :]
    if 'dim_6' in json_header:
        dim6 = json_header['dim_6']
        if og_num_dims == 6:
            nifti = nifti[0, 0, 0, :, :, :]
    if 'dim_7' in json_header:
        dim7 = json_header['dim_7']
        if og_num_dims == 7:
            nifti = nifti[0, 0, 0, :, :, :, :]
        
    all_dims = ["DIM_SPEC_POINTS", dim5, dim6, dim7, "NOT_ASSIGNED_EXTRA"]

    # check which dimension has coils
    if 'DIM_COIL' in all_dims:
        coil_ind = all_dims.index("DIM_COIL")

        # make sure coil dimension is 1 and remove from list
        if nifti.shape[int(coil_ind)]>1 or coil_ind>1:                    # coil index is not first (after spec points)
            print('Coils need to be combined prior to using this function and must be DIM_5.')
            return False
    else:
        coil_ind = len(all_dims)-1

    # check which dimension has averages
    if 'DIM_DYN' in all_dims:
        avgs_ind = all_dims.index("DIM_DYN")
    else:
        avgs_ind = len(all_dims)-1
    
    # check which dimension has editing
    if 'DIM_EDIT' in all_dims:
        edit_ind = all_dims.index("DIM_EDIT")
        edit_cond = json_header[f'dim_{int(unused_dims+spec_points_dim+edit_ind)}_header']['EditCondition'][0]
    else:
        edit_ind = len(all_dims)-1
        edit_cond = len(all_dims)-1

    if len(list(set(all_dims) - {"DIM_COIL", "DIM_DYN", "DIM_EDIT", "DIM_SPEC_POINTS", "NOT_ASSIGNED", "NOT_ASSIGNED_EXTRA"}))>0:
        print("Custom coil dimensions are not permitted with this function. Dimensions must be named 'DIM_COIL', 'DIM_EDIT', and/or 'DIM_DYN'.")
        return False

    # assign correct order of values to FIDs
    # spectral points only
    if og_num_dims==4:
        fids = nifti[0, 0, 0, :][np.newaxis, :]
        fids_on, fids_off = [], []
    
    # five total dimensions
    elif og_num_dims==5:

        # num averages or coil combined only
        if dim5 == all_dims[int(avgs_ind)] or dim5 == all_dims[int(coil_ind)]:
            fids = np.transpose(nifti[:, :])
            fids_on, fids_off = [], []
        
        # subspectra only (ON first)
        elif dim5 == all_dims[int(edit_ind)] and edit_cond=='ON':
            fids_on = np.transpose(nifti[:, 0])
            fids_off = np.transpose(nifti[:, 1])

        # subspectra only (OFF first)
        else:
            fids_on = np.transpose(nifti[:, 1])
            fids_off = np.transpose(nifti[:, 0])
    
    # six total dimensions
    elif og_num_dims==6:

        # coil combined first in sequence
        if dim5 == all_dims[int(coil_ind)]:
            # num averages second in sequence
            if dim6 == all_dims[int(avgs_ind)]:
                fids = np.transpose(nifti[:, 0, :])
                fids_on, fids_off = [], []
            
            # subspectra second in sequence
            # subspectra (ON first)
            elif dim6 == all_dims[int(edit_ind)] and edit_cond=='ON':
                fids_on = np.transpose(nifti[:, 0, 0])
                fids_off = np.transpose(nifti[:, 0, 1])

            # subspectra (OFF first)
            else:
                fids_on = np.transpose(nifti[:, 0, 1])
                fids_off = np.transpose(nifti[:, 0, 0])

        # num averages first and subspectra second in sequence
        elif dim5 == all_dims[int(avgs_ind)]:

            # subspectra (ON first)
            if edit_cond=='ON':
                fids_on = np.transpose(nifti[:, :, 0])
                fids_off = np.transpose(nifti[:, :, 1])
            
            # subspectra (OFF first)
            else:
                fids_on = np.transpose(nifti[:, :, 1])
                fids_off = np.transpose(nifti[:, :, 0])
        
        # subspectra first and num averages second in sequence
        # subspectra (ON first)
        elif dim5 == all_dims[int(edit_ind)] and edit_cond=='ON':
            fids_on = np.transpose(nifti[:, 0, :])
            fids_off = np.transpose(nifti[:, 1, :])
        
        # subspectra (OFF first)
        else:
            fids_on = np.transpose(nifti[:, 1, :])
            fids_off = np.transpose(nifti[:, 0, :])
    
    # seven total dimensions (coil combined first dimension)
    elif og_num_dims==7:

        # num averages second and subspectra third in sequence
        if dim6 == all_dims[int(avgs_ind)]:

            # subspectra (ON first)
            if dim7 == all_dims[int(edit_ind)] and edit_cond=='ON':
                fids_on = np.transpose(nifti[:, 0, :, 0])
                fids_off = np.transpose(nifti[:, 0, :, 1])
            
            # subspectra (OFF first)
            else:
                fids_on = np.transpose(nifti[:, 0, :, 1])
                fids_off = np.transpose(nifti[:, 0, :, 0])
        
        # subspectra second and num averages third in sequence
        # subspectra (ON first)
        elif dim6 == all_dims[int(edit_ind)] and edit_cond=='ON':
            fids_on = np.transpose(nifti[:, 0, 0, :])
            fids_off = np.transpose(nifti[:, 0, 1, :])
        
        # subspectra (OFF first)
        else:
            fids_on = np.transpose(nifti[:, 0, 1, :])
            fids_off = np.transpose(nifti[:, 0, 0, :])

    else:
        print('Dimensions do not follow one of the following expected conventions:')
        print('dims 1-4;')
        print('dim_5 (DIM_COIL)')
        print('dim_5 (DIM_DYN)')
        print('dim_5 (DIM_EDIT)')
        print('dim_5 (DIM_DYN), dim_6 (DIM_EDIT);')
        print('dim_5 (DIM_EDIT), dim_6 (DIM_DYN);')
        print('dim_5 (DIM_COIL) [SIZE = 1], dim_6 (DIM_DYN);')
        print('dim_5 (DIM_COIL) [SIZE = 1], dim_6 (DIM_EDIT);')
        print('dim_5 (DIM_COIL) [SIZE = 1], dim_6 (DIM_DYN), dim_7 (DIM_EDIT);')
        print('dim_5 (DIM_COIL) [SIZE = 1], dim_6 (DIM_EDIT), dim_7 (DIM_DYN);')

    # interleave fids for ease of use
    if list(fids_on):                                               # check if fids_on is not empty and that samples exist
        if fids_on.ndim==1:                                         # single pair of trans
            fids = interleave(fids_off=fids_off[np.newaxis, :], fids_on=fids_on[np.newaxis, :])
            print('Nifti accepted - Interleaving subspectra...')
        if fids_on.ndim==2:
            fids = interleave(fids_off=fids_off, fids_on=fids_on)
            print('Nifti accepted - Interleaving subspectra...')

    return  fids, time, ppm


def return_nifti_mrs_data(dir_nifti, fids, edited=True):
    '''
    Returns single voxel FIDs to a new nifti-mrs file (with existing header) 
    (assumes standard dimension naming conventions as per Clarke et al. 2022 where data is already coil combined)
    (dim_6 (DIM_DYN), dim_7 (DIM_EDIT))
    :param:     dir_nifti (string): directory and nifti-mrs filename (i.e. "C:/Users/FIDA/MyNiftiFile.nii.gz")
                fids (complex floats): free induction decay values of shape [num_samples, spec_points] (assumes interleaved when 'edited' is TRUE)
                edited (boolean): indicates whether the scan is edited (designated by True)
    '''
    dim5, dim6, dim7 = 'NOT_ASSIGNED', 'NOT_ASSIGNED', 'NOT_ASSIGNED'
    spec_points_dim = 1                         # existing spec points dim
    unused_dims = 3                             # voxel location dims (assumed single voxel, unsused)
    final_fids = []

    # check fid dimension
    if fids.ndim > 2:
        print('Fids are larger than expected ndim of 1 or 2.')
        return False

    # open nifti-mrs file
    nifti = nib.load(dir_nifti)
    nifti_data = nifti.get_fdata(dtype=np.complex64)
    og_num_dims = nifti_data.ndim
    json_header = json.loads(nifti.header.extensions[nifti.header.extensions.get_codes().index(44)].get_content())

    # check for existence of dimensions 5-7
    if 'dim_5' in json_header:
        dim5 = json_header['dim_5']
    if 'dim_6' in json_header:
        dim6 = json_header['dim_6']
    if 'dim_7' in json_header:
        dim7 = json_header['dim_7']
        
    all_dims = ["DIM_SPEC_POINTS", dim5, dim6, dim7, "NOT_ASSIGNED_EXTRA"]

    # check which dimension has coils
    if 'DIM_COIL' in all_dims:
        coil_ind = all_dims.index("DIM_COIL")

        # make sure coil dimension is 1 and remove from list
        if nifti.shape[int(coil_ind)]>1 or coil_ind>1:                    # coil index is not first (after spec points)
            print('Coils need to be combined prior to using this function and must be DIM_5.')
            return False
    else:
        coil_ind = len(all_dims)-1

    # check which dimension has averages
    if 'DIM_DYN' in all_dims:
        avgs_ind = all_dims.index("DIM_DYN")
    else:
        avgs_ind = len(all_dims)-1
    
    # check which dimension has editing
    if 'DIM_EDIT' in all_dims:
        edit_ind = all_dims.index("DIM_EDIT")
        edit_cond = json_header[f'dim_{int(unused_dims+spec_points_dim+edit_ind)}_header']['EditCondition'][0]
    else:
        edit_ind = len(all_dims)-1
        edit_cond = len(all_dims)-1

    # assign values to correct dimension
    # acquisition is edited
    if edited:
        # if edited, undo interleaving and transpose fids for nifti format
        print('Undoing subspectra interleave...')
        fids_on, fids_off = undo_interleave(fids=fids)
        fids_on, fids_off = np.transpose(fids_on), np.transpose(fids_off)

        # five total dimensions
        if og_num_dims==5:
            final_fids = np.zeros(shape=(fids_on.shape[0], 2), dtype=complex)

            # subspectra dimension (ON first) (Assumes single ON and OFF transient)
            if dim5 == all_dims[int(edit_ind)] and edit_cond=='ON':    
                fids_on, fids_off = np.squeeze(fids_on), np.squeeze(fids_off)       
                final_fids[:, 0], final_fids[:, 1] = fids_on, fids_off
            
            # subspectra (OFF first) (Assumes single ON and OFF transient)
            else:
                fids_on, fids_off = np.squeeze(fids_on), np.squeeze(fids_off)     
                final_fids[:, 0], final_fids[:, 1] = fids_off, fids_on

            final_fids = final_fids[np.newaxis, np.newaxis, np.newaxis, :, :]

        # six total dimensions
        elif og_num_dims==6:
    
            # coil dimension present first in sequence and subspectra second in sequence
            if dim5 == all_dims[int(coil_ind)]:     
                final_fids = np.zeros(shape=(fids_on.shape[0], 1, 2), dtype=complex)

                # subspectra (ON first)
                if dim6 == all_dims[int(edit_ind)] and edit_cond=='ON':
                    fids_on, fids_off = np.squeeze(fids_on), np.squeeze(fids_off)     
                    final_fids[:, 0, 0], final_fids[:, 0, 1] = fids_on, fids_off
                
                # subspectra (OFF first)
                else:
                    fids_on, fids_off = np.squeeze(fids_on), np.squeeze(fids_off)     
                    final_fids[:, 0, 0], final_fids[:, 0, 1] = fids_off, fids_on
            
            # num averages first in sequence and subspectra second in sequence
            elif dim5 == all_dims[int(avgs_ind)]:
                final_fids = np.zeros(shape=(fids_on.shape[0], fids_on.shape[1], 2), dtype=complex)

                # subspectra (ON first) second in sequence
                if edit_cond=='ON':
                    final_fids[:, :, 0], final_fids[:, :, 1] = fids_on, fids_off
                
                # subspectra (OFF first) second in sequence
                else:
                    final_fids[:, :, 0], final_fids[:, :, 1] = fids_off, fids_on

            # subspectra first in sequence and num averages second in sequence
            else:
                final_fids = np.zeros(shape=(fids_on.shape[0], 2, fids_on.shape[1]), dtype=complex)

                # subspectra (ON first)
                if dim5 == all_dims[int(edit_ind)] and edit_cond=='ON':
                    final_fids[:, 0, :], final_fids[:, 1, :] = fids_on, fids_off
            
                # subspectra (OFF first)
                else:
                    final_fids[:, 0, :], final_fids[:, 1, :] = fids_off, fids_on

            final_fids = final_fids[np.newaxis, np.newaxis, np.newaxis, :, :, :]

        # seven total dimensions (coil dimension must be present)
        elif og_num_dims==7:

            # coil dimension first, num averages second, and subspectra third in sequence
            if dim6 == all_dims[int(avgs_ind)]: 
                final_fids = np.zeros(shape=(fids_on.shape[0], 1, fids_on.shape[1], 2), dtype=complex)
            
                # subspectra (ON first)
                if dim7 == all_dims[int(edit_ind)] and edit_cond=='ON':   
                    final_fids[:, 0, :, 0], final_fids[:, 0, :, 1] = fids_on, fids_off

                # subspectra (OFF first)
                else: 
                    final_fids[:, 0, :, 0], final_fids[:, 0, :, 1] = fids_off, fids_on


            # coil dimension first, subspectra second, and num averages third in sequence
            else:
                final_fids = np.zeros(shape=(fids_on.shape[0], 1, 2, fids_on.shape[1]), dtype=complex)

                # subspectra (ON first)
                if dim6 == all_dims[int(edit_ind)] and edit_cond=='ON':   
                    final_fids[:, 0, 0, :], final_fids[:, 0, 1, :] = fids_on, fids_off

                # subspectra (OFF first)
                else:     
                    final_fids[:, 0, 0, :], final_fids[:, 0, 1, :] = fids_off, fids_on        

            final_fids = final_fids[np.newaxis, np.newaxis, np.newaxis, :, :, :, :]

    # acquisition is NOT edited
    else:
        # num averages only or coil combined only
        if og_num_dims==5 and (dim5 == all_dims[int(avgs_ind)] or dim5 == all_dims[int(coil_ind)]):
            final_fids = np.transpose(fids)
            final_fids = final_fids[np.newaxis, np.newaxis, np.newaxis, :, :]

        # coil combined (dim5) and num averages (dim6)
        elif og_num_dims==6 and dim5 == all_dims[int(coil_ind)] and dim6 == all_dims[int(avgs_ind)]:
            final_fids = np.zeros(shape=(fids.shape[1], 1, fids.shape[0]), dtype=complex)
            final_fids[:, 0, :] = np.transpose(fids)
            final_fids = final_fids[np.newaxis, np.newaxis, np.newaxis, :, :, :]

        # assumes spectral points only
        else:
            final_fids = fids
            final_fids = final_fids[np.newaxis, np.newaxis, np.newaxis, :]

    # save new data into modified nifti file
    if not list(final_fids):
        print('Nifti NOT accepted...')
    else:
        print('Nifti accepted...')
        new_nifti = nib.Nifti2Image(final_fids, nifti.affine, nifti.header)
        og_dir = dir_nifti.split(".")
        nib.save(new_nifti, f"{og_dir[0]}_SMART.nii.gz")


########################################################################################################################
# Support Functions
########################################################################################################################
def to_fids(specs, axis=1):
    '''
    Convert to Fids (time domain)
    :param:     specs (complex floats): spectrum values of shape [num_samples, spec_points]
                axis: *provided in case SPEC axes are swapped*
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points]
    '''
    return ifft(fftshift(specs, axes=axis), axis=axis)


def to_specs(fids, axis=1):
    '''
    Convert to Specs (frequency domain)
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                axis: *provided in case FID axes are swapped*
    :return:    specs (complex floats): spectrum values of shape [num_samples, spec_points]
    '''
    return fftshift(fft(fids, axis=axis), axes=axis)


def interleave(fids_on, fids_off):
    '''
    Interleave edited Fids so they appear in order of collection (assumes ON first and a 1,0,1,0,1... interleaving)
    :param:     fids_on (complex floats): free induction decay values of shape [num_samples/2, spec_points]
                fids_off (complex floats): free induction decay values of shape [num_samples/2, spec_points]
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] **interleaved**
    '''
    fids = np.zeros((fids_on.shape[0]*2, fids_on.shape[1]), dtype=complex)
    on, off = 0, 0

    for ii in range(0, fids_on.shape[0]*2):
        if ii%2==0:
            fids[ii, :] = fids_on[on, :]
            on+=1
        else:
            fids[ii, :] = fids_off[off, :]
            off+=1

    return fids


def undo_interleave(fids):
    '''
    Reverses interleaving of edited Fids to obtain both subspectra groups separately (assumes ON first and a 1,0,1,0,1... interleaving)
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
    :return:    fids_on (complex floats): free induction decay values of shape [num_samples/2, spec_points]
                fids_off (complex floats): free induction decay values of shape [num_samples/2, spec_points]
    '''
    fids_on = np.zeros((int(fids.shape[0]/2), fids.shape[1]), dtype=complex)
    fids_off = np.zeros((int(fids.shape[0]/2), fids.shape[1]), dtype=complex)
    on, off = 0, 0
    
    for ii in range(0, fids.shape[0]):
        if ii%2==0:
            fids_on[on, :] = fids[ii, :]
            on+=1
        else:
            fids_off[off, :] = fids[ii, :]
            off+=1

    return fids_on, fids_off


def scale(fids):
    '''
    Scale data by the max of the absolute value of the complex data to apply artifact functions
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] **normalized**
                scaleFact (float): scale factor
    '''
    scale_fact = np.max(abs(to_specs(fids)))
    norm_specs = to_specs(fids)/scale_fact
    return to_fids(norm_specs), scale_fact


def undo_scale(fids, scale_fact):
    '''
    Undo scaling
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points] **normalized**
                scaleFact (float): scale factor
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points]
    '''
    norm_specs = to_specs(fids)*scale_fact
    return to_fids(norm_specs)
    