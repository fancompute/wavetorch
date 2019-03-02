# wavetorch

## Overview

This package is for solving and optimizing (learning) on the two-dimensional (2D) scalar wave equation. Vowel data from [Prof. James Hillenbrand](https://homepages.wmich.edu/~hillenbr/voweldata.html) is in `data/`, study scripts are in `study/`, and the package itself is in `wavetorch/`. This package uses `pytorch` to perform the optimization and gradient calculations.

The best entry points to this package are the study scripts which are described below.

## Implementation

 - [ ] Describe the finite difference formulations
 - [ ] Describe how convolutions are used to implement the spatial FDs
 - [ ] Describe the adiabatic absorber formulation
 - [ ] Describe `WaveCell()`
 - [ ] Describe data loading and batching

## Studies

 * `study/train.py` - Primary script for training on vowel recognition
 * `study/inference.py` - Script for performing inference from saved trained models and for plotting fields

## Requirements

This package has the following dependencies.

* `pytorch`
* `sklearn`
* `librosa`
* `seaborn`
* `matplotlib`
* `numpy`