# SMART_MRS
**SMART_MRS** is a Python based library (toolbox) for applying 
simulated artifacts to edited Magnetic Resonance Spectroscopy (MRS) data.

For further information on the toolbox, please see LINK_TO_PREPRINT


## Updates
Current version is 1.0 with no preceding versions.


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
* JSON
* Math
* Nibabel (v.5.2.1)
* NumPy (v.1.25.2)
* Random
* SciPy (v.1.11.4)


## Installation
Use pip package manager to install the **SMART_MRS** library as below:

```bash
pip install SMART_MRS
```


## Usage
Below are example uses of functions from each module in **SMART_MRS**.
For further information on specific functions, please consult: LINK_TO_READ_THE_DOCS

```python
from SMART_MRS

# IO Functions example get_nifti_mrs_data() - returns FIDs, time, and ppm
dir = "C:/Users/"
fids, time, ppm = SMART_MRS.IO.get_nifti_mrs_data(dir_nifti=dir+"jdifference_nifti_SMART_MRS_EX.nii.gz")

# Artifacts Functions example add_nuisance_peak() 
# - returns FIDs and artifact locations within dataset

# Apply specific user values
gaussian_peak_profile = {
    "peak_type": "G",
    "amp": [0.00015],
    "width": [0.8],    
    "res_freq": [4],
    "edited": 1.4}

# When echo is True, will print non-user specified values used (in this case, the locations of the artifacts)
fids, np_locations = SMART_MRS.artifacts.add_nuisance_peak(fids=fids, time=time, peak_profile=gaussian_peak_profile, num_trans=3, echo=True)

# Applied Functions example add_disruptive_motion_artifact() - returns FIDs and artifact locations within dataset
# use function specific values
fids, motion_locations = SMART_MRS.applied.add_disruptive_motion_artifact(fids=fids, time=time, ppm=ppm, mot_locs=[3, 9], nmb_motion=2)

# Save Fids with Artifacts as original data type
# New nifti should be saved under same name "_SMART.niigz" at same location
SMART_MRS.IO.return_nifti_mrs_data(dir_nifti=dir+"jdifference_nifti_SMART_MRS_EX.nii.gz", fids=fids, edited=True)
```
