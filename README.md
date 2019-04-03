# wavetorch

## Introduction

This python package computes solutions to the [scalar wave equation](https://en.wikipedia.org/wiki/Wave_equation). Using backpropagation, `wavetorch` also computes gradients of those solutions with respect to the spatial distribution of the wave speed. In practice, the wave speed could be related to a material density distribution in an acoustic setting, or to a distribution of materials with different refractive indices in an optical setting. 

In this package, the wave equation is discretized using centered finite differences in both space and time. This discretization is implemented in a custom RNN cell which subclasses `torch.nn.Module` from pytorch. The optimizers provided by pytorch (e.g. ADAM, SGD, LBFGS, etc) are used to optimize the physical system described by the scalar wave equation.

## Application: Vowel recognition

This package is designed around the application of vowel recognition. A dataset [1] of 12 vowel classes recorded from 45 male and 48 female speakers is located in the `data/` subfolder.

## Usage

### Training

To train the model using the configuration specified by the file [study/example.yml](study/example.yml), issue the following command from the top-level of the repository:
```
python -m wavetorch train ./study/example.yml
```
The configuration file, [study/example.yml](study/example.yml), is commented to provide information on how the vowel data is processed, how the physics of the problem is specified, and how the training process is configured.

During training, the progress of the optimization will be printed to the screen. At the end of each epoch, the current state of the model, along with a history of the model state and performance at all previous epochs and cross validation folds, is saved to a file.

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
