# wavetorch

## Introduction

This python package computes solutions to the [scalar wave equation](https://en.wikipedia.org/wiki/Wave_equation). Using backpropagation, `wavetorch` also computes gradients of those solutions with respect to the spatial distribution of the wave speed. In practice, the wave speed could be related to a material density in an acoustic setting, or to a refractive index in an optical setting. 

In this package, the wave equation is discretized using centered finite differences in both space and time which are implemented in a custom RNN cell subclassing pytorch's `torch.nn.Module`.

## Learning task: vowel recognition

This package is designed around the physically-motivated application of vowel recognition. A dataset of recorded vowels from male and female speakers (sourced from Prof. James Hillenbrand's [website]](https://homepages.wmich.edu/~hillenbr/voweldata.html)) is located in `data/`.

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

The configuration file is heavily commented. Please see [study/example.yml](study/example.yml) for an example of how to configure the training process. After issuing the above command, the model will be optimized and the progress will be printed to the screen. After training, the model will be saved to a file, along with the training history and the problem configuration.

**WARNING:** depending on the batch size, the window length, and the sample rate for the vowel data (all of which are specified in the YAML configuration file) the gradient computation may require a significant amount of memory. It is recommended to start small with the batch size and work your way up gradually, depending on what your machine can handle.

### Results

#### Summary

A summary figure of a trained model can be created with the following command:
```
python -m wavetorch summary <PATH_TO_MODEL>
```

The output will look something like the following:

![](../master/img/summary.png)

#### Fields

The command
```
python -m wavetorch fields <PATH_TO_MODEL> 1500 2500 3500 ...
```
will display snapshots in time of the field distribution, like so:

![](../master/img/fields.png)

#### STFT (short-time Fourier transform)

The command
```
python -m wavetorch stft <PATH_TO_MODEL>
```
will display a matrix of short time Fourier transforms of the received signal, where the row corresponds to an input vowel and the column corresponds to a particular probe (matching the confusion matrix distribution), like so:

![](../master/img/stft.png)

## Dependencies

* `pytorch`
* `sklearn`
* `librosa`
* `seaborn`
* `matplotlib`
* `numpy`
* `yaml`
* `pandas`
