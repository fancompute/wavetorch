# wavetorch

![](../master/img/optimization.png)

## Introduction

This python package computes time-domain solutions of the [scalar wave equation](https://en.wikipedia.org/wiki/Wave_equation). It also computes gradients of these solutions, using pytorch, with respect to the spatial distribution of material density inside a user-defined region of a larger domain. Here, the wave equation is discretized with finite differences and implemented in a recurrent neural network (RNN) cell, `WaveCell`, which subclasses `torch.nn.Module`. This allows all of the standard pytorch optimization modules to be used for training/optimization. An example of a structure's evolution during the training procedure is shown in the image above.

This package is designed to perform vowel recognition, using the the dataset of raw audio recordings available from Prof James Hillenbrand's [website](https://homepages.wmich.edu/~hillenbr/voweldata.html). However, the core components provided by this package, namely the `WaveCell` module and the training routines, may be easily applied to other learning tasks involving time-series data. 

If you find this package useful in your research, please consider citing our paper:

 * T. W. Hughes, I. A. D. Williamson, M. Minkov, and S. Fan, "[Wave Physics as an Analog Recurrent Neural Network](https://arxiv.org/abs/1904.12831)" ArXiv:1904.12831 [physics.app-ph], 2019.

## Usage

To use this package, simply clone and/or download the repository:
```
git clone https://github.com/fancompute/wavetorch.git
```
All interactions with this package can be carried out from the top-level directory of the repository, as described below. It's also helpful to add the top-level of the repository to the `PYTHONPATH` environment variable. This can be achieved (on a Unix-like system) by executing the following command from the top-level directory of the repository:
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Propagating waves

A simple example of modeling a monochromatic excitation is provided in [study/propagate.py](study/propagate.py). 

![](../master/img/propagate.png)

### Training on vowel recognition

To train the model using the configuration specified by the file [study/example.yml](study/example.yml), issue the following command from the top-level directory of the repository:
```
python ./study/vowel_train.py ./study/example.yml
```
The configuration file, [study/example.yml](study/example.yml), is commented to provide information on how the vowel data is processed, how the physics of the problem is specified, and how the training process is configured.

During training, the progress of the optimization will be printed to the screen. At the end of each epoch, the current state of the model, along with a history of the model state and performance at all previous epochs and cross validation folds, is saved to a file.

**WARNING:** depending on the batch size, the window length, and the sample rate for the vowel data (all of which are specified in the YAML configuration file) the gradient computation may require a significant amount of memory. It is recommended to start small with the batch size and work your way up gradually, depending on what your machine can handle.

### Summary of vowel recognition results

A summary of a trained model which was previously saved to disk can be generated like so:
```
python ./study/vowel_summary.py <PATH_TO_MODEL>
```

![](../master/img/summary.png)

### Display field snapshots during vowel recognition

Snapshots of the scalar field distribution for randomly selected vowels samples can be generated like so:
```
python ./study/vowel_analyze.py fields <PATH_TO_MODEL> --times 1500 2500 3500 ...
```

![](../master/img/fields.png)

### Display short-time Fourier transform (STFT) of vowel waveforms

A matrix of short time Fourier transforms of the received signal, where the row corresponds to an input vowel and the column corresponds to a particular probe (matching the confusion matrix distribution) can be generated like so:
```
python ./study/vowel_analyze.py stft <PATH_TO_MODEL>
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
