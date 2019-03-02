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

## Usage

The following scripts represent the primary entry points to this package: 

* `study/train.py` - script for training on vowel recognition
* `study/inference.py` - script for performing inference from saved models and for plotting field patterns; this script can also be used to simulate propagating waves on a "blank" model (if no model is specified for loading)

These scripts have a lot of options which can be summarized by passing the `--help` flag.

As an example, the following command (issued via ipython) can be used to train the model for 5 epochs:
```
%run ./study/train.py --N_epochs 5 --batch_size 3 --train_size 12 --test_size 12
```
**WARNING:** depending on the batch size and the sample rate for the vowel data, determined by the `--sr` option, the gradient computation may require significant amounts of memory. Using too large of a value for either of these parameters may cause your computer to lock up.

**Note:** The model trained in this example will not perform very well because we used very few training examples.

After issuing the above command, the model will be optimized and the progress will be printed to the screen. After training, the model will be saved to a file, along with the training history and all of the input arguments.

The following command can be issued to load a previously saved model file:
```
%run ./study/inference.py --model <PATH_TO_MODEL>
```
Additionally, several options can be passed to this script to view various results.

The `--show` option will display the distribution of the wave speed, like so:
![](../master/img/c.png)

The `--hist` option will display the training history, if it was saved with the model, like so:
![](../master/img/hist.png)

The `--cm` option will display a confusion matrix, computed over the entire dataset, like so:
![](../master/img/cm.png)

The `--fields` option will display an integrated field distribution for the vowel classes, along with spectral information for the time series data, like so:
![](../master/img/fields.png)

## Requirements

This package has the following dependencies:

* `pytorch`
* `sklearn`
* `librosa`
* `seaborn`
* `matplotlib`
* `numpy`
