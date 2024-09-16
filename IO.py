"""IO module (SMART_MRS, 2024, Bugler et al)

These functions allow the user to import and export from specific data formats (FID-A .mat structure and nifti-MRS).

These functions primarily require local directory and fids [num_samples, spec_points] data.

These functions require the support module, NumPy, JSON, Scipy.io, and Nibabel libraries.

This file can also be imported as a module and contains the following functions:

    * get_FIDA_mat_data - gets FIDs, time and ppm from a FID-A .mat structure format
    * return_FIDA_mat_data - returns FIDs to an existing FID-A .mat structure format
    * get_nifti_mrs_data - gets FIDs, time and ppm from a nifti-MRS format
    * return_nifti_mrs_data - returns FIDs to an nifti-MRS format
"""

# import Python packages
import numpy as np
import json
import scipy.io as sci_io
import nibabel as nib

# import functions from support module
from .support import interleave, undo_interleave

# IO module functions
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
    ppm = np.squeeze(sims[0,0][2])

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
    sci_io.savemat(dir_mat, {struct_name: sims})


def get_nifti_mrs_data(dir_nifti):
    '''
    Load single voxel FIDs from an existing nifti-mrs file (assumes standard dimension naming conventions as per Clarke et al. 2022
    where data is already coil combined)(dim_6 (DIM_DYN), dim_7 (DIM_EDIT))
    :param:     dir_nifti (string): directory and nifti-mrs filename (i.e. "C:/Users/FIDA/MyNiftiFile.nii.gz")
    :return:    fids (dictionary of complex floats): free induction decay values of shape [num_samples, spec_points] 
                time (float): vector containing time values [spec_points]
                ppm (float): vector containing ppm values [spec_points] 
    '''
    dim5, dim6, dim7 = 'NOT_ASSIGNED', 'NOT_ASSIGNED', 'NOT_ASSIGNED'
    spec_points_dim = 1  # existing spec points dim
    unused_dims = 3  # voxel location dims (assumed single voxel, unsused)

    # open nifti-mrs file
    nifti_load = nib.load(dir_nifti)
    nifti = nifti_load.get_fdata(dtype=np.complex64)
    og_num_dims = nifti.ndim
    nifti_header = nifti_load.header
    json_header = json.loads(
        nifti_load.header.extensions[nifti_load.header.extensions.get_codes().index(44)].get_content())

    # check if single voxel
    if nifti.shape[0] > 1 or nifti.shape[1] > 1 or nifti.shape[2] > 1:
        print('Data is not from single voxel.')
        return False

    # obtained by going from +ve to -ve bandwidth (of length = number of spec points) divided by the central frequency
    freq = np.linspace((1 / nifti_header['pixdim'][4]) / 2, -(1 / nifti_header['pixdim'][4]) / 2, nifti.shape[3])
    ppm = 4.65 + freq / json_header['SpectrometerFrequency'][0]  # change if center frequency is not 4.65
    time = 1 / freq

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
        if nifti.shape[int(coil_ind)] > 1 or coil_ind > 1:  # coil index is not first (after spec points)
            print('Coils need to be combined prior to using this function and must be DIM_5.')
            return False
    else:
        coil_ind = len(all_dims) - 1

    # check which dimension has averages
    if 'DIM_DYN' in all_dims:
        avgs_ind = all_dims.index("DIM_DYN")
    else:
        avgs_ind = len(all_dims) - 1

    # check which dimension has editing
    if 'DIM_EDIT' in all_dims:
        edit_ind = all_dims.index("DIM_EDIT")
        edit_cond = json_header[f'dim_{int(unused_dims + spec_points_dim + edit_ind)}_header']['EditCondition'][0]
    else:
        edit_ind = len(all_dims) - 1
        edit_cond = len(all_dims) - 1

    if len(list(set(all_dims) - {"DIM_COIL", "DIM_DYN", "DIM_EDIT", "DIM_SPEC_POINTS", "NOT_ASSIGNED",
                                 "NOT_ASSIGNED_EXTRA"})) > 0:
        print(
            "Custom coil dimensions are not permitted with this function. Dimensions must be named 'DIM_COIL', 'DIM_EDIT', and/or 'DIM_DYN'.")
        return False

    # assign correct order of values to FIDs
    # spectral points only
    if og_num_dims == 4:
        fids = nifti[0, 0, 0, :][np.newaxis, :]
        fids_on, fids_off = [], []

    # five total dimensions
    elif og_num_dims == 5:

        # num averages or coil combined only
        if dim5 == all_dims[int(avgs_ind)] or dim5 == all_dims[int(coil_ind)]:
            fids = np.transpose(nifti[:, :])
            fids_on, fids_off = [], []

        # subspectra only (ON first)
        elif dim5 == all_dims[int(edit_ind)] and edit_cond == 'ON':
            fids_on = np.transpose(nifti[:, 0])
            fids_off = np.transpose(nifti[:, 1])

        # subspectra only (OFF first)
        else:
            fids_on = np.transpose(nifti[:, 1])
            fids_off = np.transpose(nifti[:, 0])

    # six total dimensions
    elif og_num_dims == 6:

        # coil combined first in sequence
        if dim5 == all_dims[int(coil_ind)]:
            # num averages second in sequence
            if dim6 == all_dims[int(avgs_ind)]:
                fids = np.transpose(nifti[:, 0, :])
                fids_on, fids_off = [], []

            # subspectra second in sequence
            # subspectra (ON first)
            elif dim6 == all_dims[int(edit_ind)] and edit_cond == 'ON':
                fids_on = np.transpose(nifti[:, 0, 0])
                fids_off = np.transpose(nifti[:, 0, 1])

            # subspectra (OFF first)
            else:
                fids_on = np.transpose(nifti[:, 0, 1])
                fids_off = np.transpose(nifti[:, 0, 0])

        # num averages first and subspectra second in sequence
        elif dim5 == all_dims[int(avgs_ind)]:

            # subspectra (ON first)
            if edit_cond == 'ON':
                fids_on = np.transpose(nifti[:, :, 0])
                fids_off = np.transpose(nifti[:, :, 1])

            # subspectra (OFF first)
            else:
                fids_on = np.transpose(nifti[:, :, 1])
                fids_off = np.transpose(nifti[:, :, 0])

        # subspectra first and num averages second in sequence
        # subspectra (ON first)
        elif dim5 == all_dims[int(edit_ind)] and edit_cond == 'ON':
            fids_on = np.transpose(nifti[:, 0, :])
            fids_off = np.transpose(nifti[:, 1, :])

        # subspectra (OFF first)
        else:
            fids_on = np.transpose(nifti[:, 1, :])
            fids_off = np.transpose(nifti[:, 0, :])

    # seven total dimensions (coil combined first dimension)
    elif og_num_dims == 7:

        # num averages second and subspectra third in sequence
        if dim6 == all_dims[int(avgs_ind)]:

            # subspectra (ON first)
            if dim7 == all_dims[int(edit_ind)] and edit_cond == 'ON':
                fids_on = np.transpose(nifti[:, 0, :, 0])
                fids_off = np.transpose(nifti[:, 0, :, 1])

            # subspectra (OFF first)
            else:
                fids_on = np.transpose(nifti[:, 0, :, 1])
                fids_off = np.transpose(nifti[:, 0, :, 0])

        # subspectra second and num averages third in sequence
        # subspectra (ON first)
        elif dim6 == all_dims[int(edit_ind)] and edit_cond == 'ON':
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
    if list(fids_on):  # check if fids_on is empty and that samples exist
        if fids_on.ndim == 1:  # single pair of trans
            fids = interleave(fids_off=fids_off[np.newaxis, :], fids_on=fids_on[np.newaxis, :])
            print('Nifti accepted - Interleaving subspectra...')
        if fids_on.ndim == 2:
            fids = interleave(fids_off=fids_off, fids_on=fids_on)
            print('Nifti accepted - Interleaving subspectra...')

    return fids, time, ppm


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
    spec_points_dim = 1  # existing spec points dim
    unused_dims = 3  # voxel location dims (assumed single voxel, unsused)
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
        if nifti.shape[int(coil_ind)] > 1 or coil_ind > 1:  # coil index is not first (after spec points)
            print('Coils need to be combined prior to using this function and must be DIM_5.')
            return False
    else:
        coil_ind = len(all_dims) - 1

    # check which dimension has averages
    if 'DIM_DYN' in all_dims:
        avgs_ind = all_dims.index("DIM_DYN")
    else:
        avgs_ind = len(all_dims) - 1

    # check which dimension has editing
    if 'DIM_EDIT' in all_dims:
        edit_ind = all_dims.index("DIM_EDIT")
        edit_cond = json_header[f'dim_{int(unused_dims + spec_points_dim + edit_ind)}_header']['EditCondition'][0]
    else:
        edit_ind = len(all_dims) - 1
        edit_cond = len(all_dims) - 1

    # assign values to correct dimension
    # acquisition is edited
    if edited:
        # if edited, undo interleaving and transpose fids for nifti format
        print('Undoing subspectra interleave...')
        fids_on, fids_off = undo_interleave(fids=fids)
        fids_on, fids_off = np.transpose(fids_on), np.transpose(fids_off)

        # five total dimensions
        if og_num_dims == 5:
            final_fids = np.zeros(shape=(fids_on.shape[0], 2), dtype=complex)

            # subspectra dimension (ON first) (Assumes single ON and OFF transient)
            if dim5 == all_dims[int(edit_ind)] and edit_cond == 'ON':
                fids_on, fids_off = np.squeeze(fids_on), np.squeeze(fids_off)
                final_fids[:, 0], final_fids[:, 1] = fids_on, fids_off

            # subspectra (OFF first) (Assumes single ON and OFF transient)
            else:
                fids_on, fids_off = np.squeeze(fids_on), np.squeeze(fids_off)
                final_fids[:, 0], final_fids[:, 1] = fids_off, fids_on

            final_fids = final_fids[np.newaxis, np.newaxis, np.newaxis, :, :]

        # six total dimensions
        elif og_num_dims == 6:

            # coil dimension present first in sequence and subspectra second in sequence
            if dim5 == all_dims[int(coil_ind)]:
                final_fids = np.zeros(shape=(fids_on.shape[0], 1, 2), dtype=complex)

                # subspectra (ON first)
                if dim6 == all_dims[int(edit_ind)] and edit_cond == 'ON':
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
                if edit_cond == 'ON':
                    final_fids[:, :, 0], final_fids[:, :, 1] = fids_on, fids_off

                # subspectra (OFF first) second in sequence
                else:
                    final_fids[:, :, 0], final_fids[:, :, 1] = fids_off, fids_on

            # subspectra first in sequence and num averages second in sequence
            else:
                final_fids = np.zeros(shape=(fids_on.shape[0], 2, fids_on.shape[1]), dtype=complex)

                # subspectra (ON first)
                if dim5 == all_dims[int(edit_ind)] and edit_cond == 'ON':
                    final_fids[:, 0, :], final_fids[:, 1, :] = fids_on, fids_off

                # subspectra (OFF first)
                else:
                    final_fids[:, 0, :], final_fids[:, 1, :] = fids_off, fids_on

            final_fids = final_fids[np.newaxis, np.newaxis, np.newaxis, :, :, :]

        # seven total dimensions (coil dimension must be present)
        elif og_num_dims == 7:

            # coil dimension first, num averages second, and subspectra third in sequence
            if dim6 == all_dims[int(avgs_ind)]:
                final_fids = np.zeros(shape=(fids_on.shape[0], 1, fids_on.shape[1], 2), dtype=complex)

                # subspectra (ON first)
                if dim7 == all_dims[int(edit_ind)] and edit_cond == 'ON':
                    final_fids[:, 0, :, 0], final_fids[:, 0, :, 1] = fids_on, fids_off

                # subspectra (OFF first)
                else:
                    final_fids[:, 0, :, 0], final_fids[:, 0, :, 1] = fids_off, fids_on


            # coil dimension first, subspectra second, and num averages third in sequence
            else:
                final_fids = np.zeros(shape=(fids_on.shape[0], 1, 2, fids_on.shape[1]), dtype=complex)

                # subspectra (ON first)
                if dim6 == all_dims[int(edit_ind)] and edit_cond == 'ON':
                    final_fids[:, 0, 0, :], final_fids[:, 0, 1, :] = fids_on, fids_off

                # subspectra (OFF first)
                else:
                    final_fids[:, 0, 0, :], final_fids[:, 0, 1, :] = fids_off, fids_on

            final_fids = final_fids[np.newaxis, np.newaxis, np.newaxis, :, :, :, :]

    # acquisition is NOT edited
    else:
        # num averages only or coil combined only
        if og_num_dims == 5 and (dim5 == all_dims[int(avgs_ind)] or dim5 == all_dims[int(coil_ind)]):
            final_fids = np.transpose(fids)
            final_fids = final_fids[np.newaxis, np.newaxis, np.newaxis, :, :]

        # coil combined (dim5) and num averages (dim6)
        elif og_num_dims == 6 and dim5 == all_dims[int(coil_ind)] and dim6 == all_dims[int(avgs_ind)]:
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
