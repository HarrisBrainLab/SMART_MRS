"""support module (SMART_MRS, 2024, Bugler et al)

These functions support data manipulation for the use of functions in other modules.

These functions primarily require fids data.

These functions require the NumPy and Scipy libraries.

This file can also be imported as a module and contains the following functions:

    * to_fids - gets FIDs from Specs data
    * to_specs - get Specs from FIDs data
    * interleave - interleave a two subspectrum acquisition ('ON' and 'OFF') into a single FIDs matrix
    * undo_interleave - return two subspectra ('ON' and 'OFF') from a single FIDs matrix
    * scale - normalize the Specs data (return as FIDs)
    * undo_scale - return the non-normalized Specs data (return as FIDs)
"""

# import Python packages
import numpy as np
from scipy.fftpack import fft, ifft, fftshift

# support module functions
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
