# wavetorch

In this repository I am exploring the solution and optimization (learning) on the two-dimensional scalar wave equation with pytorch.

## Contents

The following study scripts represent different entry points to the code.

 * `train/train.py` - Primary script for training on vowel recognition
 * `train/inference.py` - Script for performing inference from saved trained models
 * `train/vanilla_rnn.py` - Testing a vanilla RNN on the vowel recognition task (not working and very sensitive to weight initialization)
 * `train/propagation.py` - For performing basic solves of propagating waves

## Example

Below is an example of running `train/train.py` from `ipython`.

```
In [1]: %run ./study/train.py --N_epochs 10 --num_of_each 36 --learning_rate 0.01 --sr 5000 --ratio_train 0.5 --num_threads 16 --batch_size 9
```

```
Using CPU...
Using a sample rate of 5000 Hz (sequence length of 7663)
Using 108 total samples (36 of each vowel)
   54 for training
   54 for validation
 ---
Now begining training for 10 epochs ...
Epoch:  1/10  564.0 sec  |  L = 1.039e+00 ,  train_acc = 0.704 ,  val_acc = 0.796
Epoch:  2/10  519.0 sec  |  L = 8.419e-01 ,  train_acc = 0.944 ,  val_acc = 0.907
Epoch:  3/10  517.1 sec  |  L = 7.817e-01 ,  train_acc = 0.926 ,  val_acc = 0.889
Epoch:  4/10  516.2 sec  |  L = 7.677e-01 ,  train_acc = 0.944 ,  val_acc = 0.926
Epoch:  5/10  510.0 sec  |  L = 7.297e-01 ,  train_acc = 0.981 ,  val_acc = 0.926
Epoch:  6/10  518.1 sec  |  L = 7.217e-01 ,  train_acc = 0.963 ,  val_acc = 0.926
Epoch:  7/10  532.0 sec  |  L = 7.032e-01 ,  train_acc = 1.000 ,  val_acc = 0.907

...

```
