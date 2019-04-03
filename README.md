# wavetorch

## Introduction

This python package computes solutions to the [scalar wave equation](https://en.wikipedia.org/wiki/Wave_equation). Using backpropagation, `wavetorch` also computes gradients of those solutions with respect to the spatial distribution of the wave speed. In practice, the wave speed could be related to a material density distribution in an acoustic setting, or to a distribution of materials with different refractive indices in an optical setting. 

In this package, the wave equation is discretized using centered finite differences in both space and time, which are implemented in a custom RNN cell subclassing the pytorch `torch.nn.Module`. The optimizers provided by pytorch (e.g. ADAM, SGD, LBFGS, etc) are used to optimize the physical system described by the scalar wave equation.

## Application: Vowel recognition

This package is designed around the application of vowel recognition. A dataset [1] of 12 vowel classes recorded from 45 male and 48 female speakers is located in the `data` subfolder.

## Usage

### Training

Issuing the following command via ipython will train the model using the configuration specified by the file [study/example.yml](study/example.yml):
```
%run -m wavetorch train --config ./study/example.yml
```

Alternatively, training can be performed directly from the command line by issuing the command
```
python -m wavetorch train --config ./study/example.yml
```

After issuing the above command either via the command line or ipython, the model will be optimized and the progress will be printed to the screen. After training has completed, the model will be saved to a file, along with the training history and the configuration.

The example configuration file is heavily commented. Please see [study/example.yml](study/example.yml) for more details on how the problem, physical geometry, and training process are defined. 

**WARNING:** depending on the batch size, the window length, and the sample rate for the vowel data (all of which are specified in the YAML configuration file) the gradient computation may require a significant amount of memory. It is recommended to start small with the batch size and work your way up gradually, depending on what your machine can handle.

### Summary of results

A summary of a trained model which was previously saved to disk can be generated with the following command:
```
python -m wavetorch summary <PATH_TO_MODEL>
```
The resulting figure will look like this:

![](../master/img/summary.png)

### Display field snapshots

The command
```
python -m wavetorch fields <PATH_TO_MODEL> 1500 2500 3500 ...
```
will display snapshots in time of the field distribution, like so:

![](../master/img/fields.png)

### Display short-time Fourier transform (STFT) of signals

The command
```
python -m wavetorch stft <PATH_TO_MODEL>
```
will display a matrix of short time Fourier transforms of the received signal, where the row corresponds to an input vowel and the column corresponds to a particular probe (matching the confusion matrix distribution), like so:

![](../master/img/stft.png)

## Pacakage dependencies

* `pytorch`
* `sklearn`
* `librosa`
* `seaborn`
* `matplotlib`
* `numpy`
* `yaml`
* `pandas`

## References

1. James  Hillenbrand,  Laura  A.  Getty,  Michael  J.  Clark, and  Kimberlee  Wheeler,  "[Acoustic  characteristics  of
American English vowels](http://dx.doi.org/%2010.1121/1.411872)," The Journal of the Acoustical Society of America **97**, 3099–3111 (1995). *The associated dataset is available for download from [here](https://homepages.wmich.edu/~hillenbr/voweldata.html).*
