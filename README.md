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

This package has been reorganized from the previous study scripts to use a common entry point. Now, the module has a training and inference mode. 

### Training
Issuing the following command via ipython will train the model using the configuration specified by the file [study/example.yml](study/example.yml):
```
%run -m wavetorch train --config ./study/example.yml
```
Alternatively, training can be performed directly from the command line by issuing the command
```
python -m wavetorch train --config ./study/example.yml
```

Please see [study/example.yml](study/example.yml) for an example of how to configure the training process.

**WARNING:** depending on the batch size and the sample rate for the vowel data, determined by the `sr` option, the gradient computation may require significant amounts of memory. Using too large of a value for either of these parameters may cause your computer to lock up.
**Note:** The model trained in this example will not perform very well because we used very few training examples.

After issuing the above command, the model will be optimized and the progress will be printed to the screen. After training, the model will be saved to a file, along with the training history and all of the input arguments.

### Inference
The following command can be issued to load a previously saved model file:
```
%run -m wavetorch inference --name <PATH_TO_MODEL>
```
Alternatively, training can be performed directly from the command line by issuing the command
```
python -i -m wavetorch inference --name <PATH_TO_MODEL>
```

Adding the `--summary` option will display a summary of the trained model.

Adding the `--stft` option will display a matrix of short time Fourier transforms of the received signal, where the row corresponds to an input vowel and the column corresponds to a particular probe (matching the confusion matrix distribution), like so:

![](../master/img/stft.png)

Adding the `--fields` option will display an integrated field distribution for the vowel classes, like so:

![](../master/img/fields.png)

## Requirements

This package has the following dependencies:

* `pytorch`
* `sklearn`
* `librosa`
* `seaborn`
* `matplotlib`
* `numpy`
* `yaml`
