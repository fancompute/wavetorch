import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.nn.functional import pad

from wavetorch import *

import argparse
import time
import os

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default=None)
argparser.add_argument('--Nx', type=int, default=140)
argparser.add_argument('--Ny', type=int, default=140)
argparser.add_argument('--dt', type=float, default=0.707)
argparser.add_argument('--probe_space', type=int, default=30)
argparser.add_argument('--probe_x', type=int, default=100)
argparser.add_argument('--probe_y', type=int, default=40)
argparser.add_argument('--src_x', type=int, default=40)
argparser.add_argument('--src_y', type=int, default=70)
argparser.add_argument('--sr', type=int, default=5000)
argparser.add_argument('--num_threads', type=int, default=4)
argparser.add_argument('--pad_factor', type=float, default=1.0)
args = argparser.parse_args()

torch.set_num_threads(args.num_threads)

# Each dir corresponds to a distinct class and is automatically given a corresponding one hot
directories_str = ("./data/vowels/a/",
                   "./data/vowels/e/",
                   "./data/vowels/o/")

N_classes = len(directories_str)

x_train, x_test, y_train, y_test = load_all_vowels(directories_str, sr=args.sr, normalize=True, train_size=3, test_size=3, pad_factor=args.pad_factor)

# Load the model
if args.model is not None:
    model, _, _, _, _ = load_model(args.model)
else:
    probe_x = args.probe_x
    probe_y = torch.arange(args.probe_y, args.probe_y + N_classes*args.probe_space, args.probe_space)
    model = WaveCell(args.dt, args.Nx, args.Ny, args.src_x, args.src_y, probe_x, probe_y, binarized=False)

plot_c(model)

# Not much point in plotting the total E for uniform media
if args.model is not None:
	for xb, yb in DataLoader(TensorDataset(x_test, y_test), batch_size=3):
	    with torch.no_grad():
	        plot_total_field(model(xb, probe_output=False))

model_animate(model, x_test, block=True, batch_ind=0, filename=None, interval=1, fps=30, bitrate=768)
