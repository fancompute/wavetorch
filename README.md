# wavetorch

![](../master/img/optimization.png)

## Introduction

This python package computes solutions of the [scalar wave equation](https://en.wikipedia.org/wiki/Wave_equation) in the time domain. It also computes gradients of these solutions, using pytorch, with respect to the spatial distribution of  material density within a domain. The wave equation is discretized with finite differences and implemented in a recurrent neural network (RNN) cell, `WaveCell`, which subclasses `torch.nn.Module`. All of the standard pytorch optimization modules can be used for training. An example of the evolution of the structure during training, described by its spatial wave speed distribution, is shown above.

This package is designed around vowel recognition, using the the dataset available from James Hillenbrand's [website](https://homepages.wmich.edu/~hillenbr/voweldata.html). However, the core components provided by this package, namely the `WaveCell` module and the training routines, could be applied to many different time series learning tasks. 

Note that, in principle, you could adapt this code to be used as a component in a larger neural network stack. However, our focus is on training numerical models of physical systems to learn features of data with temporal dynamics.

## Usage

Below we describe how to use the package for vowel recognition. Note that all of the raw audio files are included in this repository.

### Training

To train the model using the configuration specified by the file [study/example.yml](study/example.yml), issue the following command from the top-level of the repository:
```
python -m wavetorch train ./study/example.yml
```
The configuration file, [study/example.yml](study/example.yml), is commented to provide information on how the vowel data is processed, how the physics of the problem is specified, and how the training process is configured.

During training, the progress of the optimization will be printed to the screen. At the end of each epoch, the current state of the model, along with a history of the model state and performance at all previous epochs and cross validation folds, is saved to a file.

**WARNING:** depending on the batch size, the window length, and the sample rate for the vowel data (all of which are specified in the YAML configuration file) the gradient computation may require a significant amount of memory. It is recommended to start small with the batch size and work your way up gradually, depending on what your machine can handle.

### Summary of results

A summary of a trained model which was previously saved to disk can be generated like so:
```
python -m wavetorch summary <PATH_TO_MODEL>
```

![](../master/img/summary.png)

### Display field snapshots

Snapshots of the scalar field distribution for randomly selected vowels samples can be generated like so:
```
python -m wavetorch fields <PATH_TO_MODEL> 1500 2500 3500 ...
```

![](../master/img/fields.png)

### Display short-time Fourier transform (STFT) of signals

A matrix of short time Fourier transforms of the received signal, where the row corresponds to an input vowel and the column corresponds to a particular probe (matching the confusion matrix distribution) can be generated like so:
```
python -m wavetorch stft <PATH_TO_MODEL>
```

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
