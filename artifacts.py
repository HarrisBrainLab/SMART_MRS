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

[last upd. 2025-03-13]
"""

# import Python packages
import random
import numpy as np
from scipy.interpolate import splrep, BSpline, splev

# import functions from support module
from .support import to_fids, to_specs

########################################################################################################################
# Artifact functions
########################################################################################################################
def add_time_domain_noise(fids, noise_level=0.00005):
    '''
    Add indepedently sampled complex time domain noise.
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                noise_level (float): standard deviation of noise level (with zero-mean noise)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with Gaussian White Noise**
    '''
    fids = (fids.real + np.random.normal(0, noise_level, size=(fids.shape))) + \
           (fids.imag + np.random.normal(0, noise_level, size=(fids.shape))) * 1j
    
    return fids


def add_spur_echo_artifact(fids, time, amp=None, cs=None, phase=None, t_echo=None, t2=None, cf_ppm=4.65, lf=127.7*10**6, locs=None, num_trans=None, cluster=False, echo=False):  
    '''
    To add a spurious echo artifact to a select number of transients
    (Adapted from Berrington et al. 2021)
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing time values [spec_points] 
                amp (list of floats): amplitude of spurious echo artifact (earlier start time will increase amp, longer echo will create wider echo)
                cs (list of floats): chemical shift in ppm
                phase (list of floats): phase of artifact in radians
                t_echo (list of floats): is the time where the echo reaches its maximum value
                t2 (list of floats): is the transverse decay time of the echo
                cf_ppm (integer/float): center frequency in ppm (default is 4.65 ppm)
                lf (int/float): larmor frequency in Hz (default is 127.7 MHz)
                locs (list of integers): list of transient numbers affected by spurious echoes
                num_trans (integer): number of spurious echo artifacts in scan
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                echo (boolean): indicates whether to print which values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] **with spurious echo(es) inserted**
                locs (list of integers): list of transient numbers affected by spurious echoes
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
    if phase is None or len(phase)!=num_trans:
        phase = np.random.uniform(0.1, 1.9, size=num_trans)*np.pi
    func_def.append(f'Phases: {phase}')

    if amp is None or len(amp)!=num_trans:
        amp = np.random.uniform((5*10**-7), (25*10**-6), size=num_trans)
    func_def.append(f'Amplitudes: {amp},')
    
    if t_echo is None or len(t_echo)!=num_trans:
        t_echo = np.random.uniform(0.1, 0.9, size=num_trans) * np.max(time, axis=0)
    else:
        t_echo = t_echo * np.max(time, axis=0).repeat(len(t_echo))
    func_def.append(f'Echo Times: {t_echo},')

    if t2 is None or len(t2)!=num_trans:
        t2 = np.random.uniform(10, 50, size=num_trans)
    func_def.append(f'Transverse Relaxation Times: {t_echo},')

    if cs is None or len(cs)!=num_trans:
        cs = np.random.uniform(0, 8, size=num_trans) 
    func_def.append(f'Chemical Shifts: {cs}')

    # insert spurious echo artifact(s)
    for ii in range(0, num_trans):
        echo_artif = amp[ii] * np.exp(-abs(time-t_echo[ii])/t2[ii]) * np.exp(1j*((cs[ii]-cf_ppm)*(10**-6)*2*np.pi*time*lf)+phase[ii])
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
        amp = np.random.uniform(8, 20, size=num_trans)
    func_def.append(f'Amplitudes: {amp}')

    if tc is None or len(tc)!=num_trans:      
        tc = np.random.uniform((1*10**-4), 0.03, size=num_trans)
    func_def.append(f'Time Constants: {tc}')

    # calculate / expand params for eddy current artifact(s)
    amp = np.array(amp)[:, np.newaxis].repeat(time.shape[0], axis=1)
    tc = np.array(tc)[:, np.newaxis].repeat(time.shape[0], axis=1)
    time = time[np.newaxis, :].repeat(num_trans, axis=0)

    # insert eddy current artifact(s)
    fids[locs, :] = fids[locs, :] * (np.exp(-1j * time * (amp * np.exp(-time / tc)) * 2 * np.pi))
    
    if echo is True:
        print(f'Non-user defined parameters for "add_eddy_current_artifact": {func_def}')
    
    return fids, np.sort(locs)


def add_nuisance_peak(fids, time, peak_profile, cf_ppm=4.65, lf=127.7*10**6, locs=None, num_trans=None, cluster=False, echo=False):
    '''
    Design the shape of and add a nuisance peak (i.e. lipid peak) to the spectrum.
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
                lf (integer/float): larmor frequency in Hz (default is 127.7 MHz)
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
        amp = np.random.uniform((5*10**-6), (2*10**-4), size=num_trans)
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
        peak_profile["res_freq"] = np.random.uniform(0, 8, size=num_trans)
    elif len(peak_profile["res_freq"]) == 1:
        peak_profile["res_freq"] = np.repeat(peak_profile["res_freq"], repeats=num_trans).tolist()
    cs = np.array(peak_profile["res_freq"])
    func_def.append(f'Peak Locations: {cs}')

    # Peak Phase      
    if peak_profile["phase"] is None:
        phase = np.random.uniform(0, 2 * np.pi, size=num_trans)
    elif len(peak_profile["phase"]) == 1:
        phase = np.repeat(peak_profile["phase"], repeats=num_trans)
    else:
        phase = np.array(peak_profile["phase"])
    func_def.append(f'Peak Phases: {phase}')

    # calculate / expand params for nuisance peak artifact(s) and insert into scan
    trans = 0
    for trans_loc in locs:
        if peak_profile["peak_type"] == 'G':
            T_2 = 2 * (1 / (width[trans] * (10**-6) * lf * np.pi))
            M_0 = (dt * amp[trans]) / (T_2 * np.sqrt(np.pi/4))
            peak_shape = M_0 * np.exp(1j*(((cs[trans] - cf_ppm) * (10**-6) * lf * 2 * np.pi * time) + phase[trans])) * np.exp(-(time**2)/(T_2**2))
        elif peak_profile["peak_type"] == 'L':
            T_2 = 1 / (width[trans] * (10**-6) * lf * np.pi)
            M_0 = dt * amp[trans] / T_2
            peak_shape = M_0 * np.exp(1j*(((cs[trans] - cf_ppm) * (10**-6) * lf * 2 * np.pi * time) + phase[trans])) * np.exp(-time/T_2)
        else:
            T_2_G = 2 * (1 / (width[trans] * (10**-6) * lf * np.pi))
            M_0_G = (dt * amp[trans]) / (T_2_G * np.sqrt(np.pi/4))
            T_2_L = 1 / (width[trans] * (10**-6) * lf * np.pi)
            M_0_L = dt * amp[trans] / T_2_L
            frac = np.random.uniform(0.1, 0.9)          # fraction attributed to "G" vs. "L" (variable eta)
            peak_time_gauss = M_0_G * np.exp(1j*(((cs[trans] - cf_ppm) * (10**-6) * lf * 2 * np.pi * time) + phase[trans])) * np.exp(-(time**2)/(T_2_G**2))
            peak_time_lorentz = M_0_L * np.exp(1j*(((cs[trans] - cf_ppm) * (10**-6) * lf * 2 * np.pi * time) + phase[trans])) * np.exp(-time/T_2_L)
            peak_shape = (peak_time_gauss*frac) + ((1-frac)*peak_time_lorentz)
        trans+=1
        fids[trans_loc, :] = fids[trans_loc, :] + peak_shape
    
    if echo is True:
        print(f'Non-user defined parameters for "add_nuisance_peak": {func_def}')

    return fids, locs


def add_baseline(fids, ppm, base_profile, locs=None, num_trans=None, cluster=None, echo=False):
    '''
    Design a wavering baseline to be added to the spectrum using Sine, Sinc and/or Spline functions.
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
            base_profile["base_var"] = 1*10**-4
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
                phase = random.uniform(0, 2) * np.pi
                if base_type == "SC":   # Sinc
                    base_shapes[bases, :] = (amp_bases[trans_nbs, bases] * ((np.sin(comp_bases[trans_nbs, bases]*ppm-phase)) / (comp_bases[trans_nbs, bases]*ppm-phase)) + slope_bases[trans_nbs, bases]*ppm)
                else:                   # Sine
                    base_shapes[bases, :] = amp_bases[trans_nbs, bases] * np.sin(comp_bases[trans_nbs, bases]*ppm-phase) + (slope_bases[trans_nbs, bases]*ppm)
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
            specs[trans, :] = (abs(specs[trans, :]) + spline_base) * np.exp(1j * np.angle(specs[trans, :]))
            num_bases += 1

    if echo is True:
        print(f'Non-user defined parameters for "add_baseline": {func_def}')

    return to_fids(specs), locs


def add_linebroad(fids, time, damp=None, locs=None, num_trans=None, cluster=False, echo=False):
    '''
    Add line broadening artifact.
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                time (float): vector containing time values [spec_points] 
                damp (list of floats): line broadening factor in Hz (represents the desired increase of the FWHM in Hz)
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
        damp = np.random.uniform(5, 20, size=num_trans)
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
    Add linear frequency drift which is presented as a linear function of frequency shifts which increase or decrease over the course of numerous transients.
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
    fids[start_trans:end_trans, :] = fids[start_trans:end_trans, :] * np.exp(-1j * time * f_shift_linear * 2 * np.pi)
    
    if echo is True:
        print(f'Non-user defined parameters for "add_freq_drift_linear": {func_def}')
    
    return fids, [start_trans, num_trans]


def add_freq_shift(fids, time, freq_var=None, dist="N", locs=None, num_trans=None, cluster=False, echo=False):
    '''
    Add frequency shifts in Hz that follow either a normal or uniform distribution.
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
        freq_var = np.random.uniform(1, 15, size=1)
    func_def.append(f'Frequency Shift Variance: {freq_var}')

    # calculate / expand params for frequency shifts
    if dist == "N":
        f_shift = np.random.normal(loc=0.0, scale=freq_var, size=(num_trans, 1)).repeat(fids.shape[1], axis=1)
    else:
        f_shift = np.random.uniform(low=-abs(freq_var), high=freq_var, size=(num_trans, 1)).repeat(fids.shape[1], axis=1)

    time = time[np.newaxis, :].repeat(num_trans, axis=0)

    # insert frequency shifts
    fids[locs, :] = fids[locs, :] * np.exp(-1j * time * f_shift * 2 * np.pi)

    if echo is True:
        print(f'Non-user defined parameters for "add_freq_shift_random": {func_def}')

    return fids, np.sort(locs)


def add_zero_order_phase_shift(fids, phase_var=None, dist="N", locs=None, num_trans=None, cluster=False, echo=False):
    '''
    Add zero order phase shifts in degrees that follow either a normal or uniform distribution.
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
        phase_var = np.random.uniform(1, 90, size=1)
    func_def.append(f'Phase Shift Variance: {phase_var}')

    # calculate / expand params for phase shifts
    if dist == "N":
        p_noise = np.random.normal(loc=0.0, scale=phase_var, size=(num_trans, 1)).repeat(fids.shape[1], axis=1)
    else:
        p_noise = np.random.uniform(low=-abs(phase_var), high=phase_var, size=(num_trans, 1)).repeat(fids.shape[1], axis=1)

    # insert phase shifts
    fids[locs, :] = fids[locs, :] * np.exp(-1j * p_noise * np.pi / 180)

    if echo is True:
        print(f'Non-user defined parameters for "add_zero_order_phase_shift": {func_def}')

    return fids, np.sort(locs)


def add_first_order_phase_shift(fids, ppm, tshift=None, dist="N", lf=127.7*10**6, cluster=False, locs=None, num_trans=None, echo=False):
    '''
    Add first order phase shifts (to frequency domain data) using a user provided time shift.
    (Adapted from phase1 in FID-A (Simpson et al. 2017))
    :param:     fids (complex floats): free induction decay values of shape [num_samples, spec_points]
                ppm (float): vector containing ppm values [spec_points] 
                tshift (float): time constant in ms used to calculate first order shifts
                dist (string): "N" indicates normal distribution (default) while "U" indicates a uniform distribution
                lf (integer/float): larmor frequency in Hz (default 127.7 MHz)
                cluster (boolean): indicates whether affected transients will be consecutive (designated by True)
                locs (list of integers): list of transient numbers affected by phase shifts
                num_trans (integer): number of phase shift artifacts in scan
                echo (boolean): indicates whether to print which values were used (designated by True)
    :return:    fids (complex floats): free induction decay values of shape [num_samples, spec_points] ** with random phase shifts**
                locs (list of integers): list of transient numbers affected by phase shifts
    '''
    func_def = []
    specs = to_specs(fids)

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

    if tshift is None:
        if dist == "N":
            tshift = np.random.normal(0, 1, size=1)
        else:
            tshift = np.random.uniform((1*10**-3), 1, size=1)
        func_def.append(f'Time shift: {tshift}')

    # calculate frequency from ppm
    freq = (ppm-np.median(ppm)) * lf

    # insert phase shifts
    for ii in locs:
        specs[ii, :] = specs[ii, :] * np.exp(-1j * freq * tshift * 2 * np.pi)

    if echo is True:
        print(f'Non-user defined parameters for "add_first_order_phase_shift": {func_def}')

    return to_fids(specs), np.sort(locs)
