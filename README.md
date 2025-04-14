# SMART_MRS
**SMART_MRS** is a Python based library (toolbox) for applying 
simulated artifacts to edited Magnetic Resonance Spectroscopy (MRS) data.

MIT License, Copyright (c) 2024 HarrisBrainLab.

Use of SMART_MRS requires citation. Please see either CITATION.cff or GitHub's "Cite this repository".

For further information on the toolbox, please see [SMART_MRS Preprint](https://www.biorxiv.org/content/10.1101/2024.09.19.612894v1)

## Updates
Current version is 2.1. Previous version 2.0.
See CHANGES.md for more details.


## Module Descriptions
The **SMART_MRS** package contains 4 modules:
* **IO.py** (allows for the import and export of specific data formats such as FID-A and nifti-mrs):
    * get_FIDA_mat_data()
    * return_FIDA_mat_data()
    * get_nifti_mrs_data()
    * return_nifti_mrs_data()

* **support.py** (supports data manipulation for the use of other module functions):
    * to_fids()
    * to_specs()
    * interleave()
    * undo_interleave()
    * scale()
    * undo_scale()

* **artifacts.py** (for the application of various artifacts):
    * add_time_domain_noise()
    * add_spur_echo_artifact()
    * add_eddy_current_artifact()
    * add_linebroad()
    * add_nuisance_peak()
    * add_baseline()
    * add_freq_drift_linear()
    * add_freq_shift_random()
    * add_zero_order_phase_shift()
    * add_first_order_phase_shift()

* **applied.py** (allows for specific iterations of the artifacts):
    * add_progressive_motion_artifact()
    * add_subtle_motion_artifact()
    * add_disruptive_motion_artifact()
    * add_lipid_artifact()


## Dependencies
Each of the 4 modules have relative dependencies in addition to the following dependencies:
* Nibabel (v.5.2.1)
* NumPy (v.1.25.2)
* SciPy (v.1.11.4)


## SMART_MRS Installation Guide
Create an empty Conda environement for installation:
```bash
conda create -n my_smart_mrs_env python=3.11.9 
```

Activate the Conda environement:
```bash
conda activate my_smart_mrs_env 
```

Install pip into your Conda environement:
```bash
conda install pip
```

Use pip package manager to install the **SMART_MRS** library as below:
```bash
python3 -m pip install SMART_MRS
```

For more information on the versions of dependencies, please consult the smart_env.yml file.


## Usage
Below are example uses of functions from each module in **SMART_MRS**.
For further information on specific functions, please consult SupplementaryMaterial.pdf

```python
import SMART_MRS
import matplotlib.pyplot as plt

# IO Functions example get_nifti_mrs_data() - returns FIDs, time, and ppm
DIR = "C:/Users/"
fids, time, ppm = SMART_MRS.IO.get_nifti_mrs_data(dir_nifti=f"{DIR}jdifference_nifti_SMART_MRS_EX.nii.gz")
print(f'Data has {fids.shape[0]} FIDs with {fids.shape[1]} spectral points.')

# Support Functions example scale() - returns scaled FIDs and scale factor
raw_scaled_fids, nifti_scale = SMART_MRS.support.scale(fids)
fids = np.copy(raw_scaled_fids)

# Artifacts Functions example add_nuisance_peak() - returns FIDs and artifact locations within dataset
# Apply specific user values
gaussian_peak_profile = {
    "peak_type": "G",
    "amp": [0.010, 0.010],
    "width": [0.85, 0.85],    
    "res_freq": [1.5, 1.5],
    "phase": [0, 0],
    "edited": 1.02
}

# When echo is True, will print non-user specified values used (in this case, the locations of the artifacts)
fids, np_locations = SMART_MRS.artifacts.add_nuisance_peak(fids=fids, time=time, peak_profile=gaussian_peak_profile, locs=[0,1], cf_ppm=3, echo=True)

# Plot nuisance peak
plt.figure(figsize=(10,5))
plt.suptitle('GABA-Edited Specs Before and After Nuisance Peak Addition')

plt.subplot(121)
plt.title('Before', fontsize=10)
plt.plot(ppm[::-1], (SMART_MRS.support.to_specs(raw_scaled_fids)[0, :]-SMART_MRS.support.to_specs(raw_scaled_fids)[1, :]).real, 'black')
plt.xlabel('ppm')
plt.xlim(0, 6)
plt.gca().invert_xaxis()

plt.subplot(122)
plt.title('After', fontsize=10)
plt.plot(ppm[::-1], (SMART_MRS.support.to_specs(fids)[0, :]-SMART_MRS.support.to_specs(raw_scaled_fids)[1, :]).real, 'green')
plt.xlabel('ppm')
plt.xlim(0, 6)
plt.gca().invert_xaxis()
plt.show()

# Applied Functions example add_disruptive_motion_artifact() - returns FIDs and artifact locations within dataset
# use function specific values
fids, motion_locations = SMART_MRS.applied.add_disruptive_motion_artifact(fids=fids, time=time, ppm=ppm, locs=[2,3])

# Plot disruptive motion
plt.figure(figsize=(10,5))
plt.suptitle('GABA-Edited Specs Before and After Disruptive Artifact')

plt.subplot(121)
plt.title('Before', fontsize=10)
plt.plot(ppm[::-1], (SMART_MRS.support.to_specs(raw_scaled_fids)[2, :]-SMART_MRS.support.to_specs(raw_scaled_fids)[3, :]).real, 'black')
plt.xlabel('ppm')
plt.xlim(0, 6)
plt.gca().invert_xaxis()

plt.subplot(122)
plt.title('After', fontsize=10)
plt.plot(ppm[::-1], (SMART_MRS.support.to_specs(fids)[2, :]-SMART_MRS.support.to_specs(fids)[3, :]).real, 'blue')
plt.xlabel('ppm')
plt.xlim(0, 6)
plt.gca().invert_xaxis()
plt.show()

# Save Fids with Artifacts as original data type
# New nifti should be saved under same name "_SMART.niigz" at same location
fids = SMART_MRS.support.undo_scale(fids=fids, scale_fact=nifti_scale)
SMART_MRS.IO.return_nifti_mrs_data(dir_nifti=f"{DIR}jdifference_nifti_SMART_MRS_EX.nii.gz", fids=fids, edited=True)
```
The above code generates the following plots:
__Nuisance Peak Example__
![alt text](https://github.com/HarrisBrainLab/SMART_MRS/NuisancePeakExample.png?raw=true)


__Disruptive Motion Example__
![alt text](https://github.com/HarrisBrainLab/SMART_MRS/DisruptiveMotionExample.png?raw=true)