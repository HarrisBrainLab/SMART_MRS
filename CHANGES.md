# SMART_MRS (2025-04-14)
**Release Highlights**
SMART_MRS (PyPi version 2.1) has the following updates:
- Minor improvements to artifact functions
    - Spurious echo updated to T2 vs. T_all
    - Larmor frequency now in Hz for ALL relevant equations
    - Corrected inconsistencies in phase component implementation in relevant equations

# SMART_MRS (2025-01-21)
**Release Highlights**
SMART_MRS (PyPi version 2) has the following updates:
- Improvements to artifact functions
    - Spurious echo updated to two-sided implementation
    - Linebroadening dampening factor definition updated
    - Nuisance peak function updated for time domain implementation
    - Additional 'spline' baseline option added with complex-valued implementation
    - Frequency and phase shift functions updated with additional sampling distribution type parameter added
- Default number of transients changed from 1 to all for artifact functions
- Nifti import function frequency vector updated to follow established convention