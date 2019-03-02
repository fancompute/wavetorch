import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.nn.functional import pad

from wavetorch import *
from wavetorch.wave import train

import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # Training options
    argparser.add_argument('--name', type=str, default=None,
                                help='Name to add to saved model file. If unspecified a date/time stamp is used')
    argparser.add_argument('--N_epochs', type=int, default=5, 
                                help='Number of training epochs')
    argparser.add_argument('--learning_rate', type=float, default=0.01, 
                                help='Optimizer learning rate')
    argparser.add_argument('--batch_size', type=int, default=3, 
                                help='Batch size used during training and testing')
    argparser.add_argument('--train_size', type=int, default=3,
                                help='Size of randomly selected training set')
    argparser.add_argument('--test_size', type=int, default=3,
                                help='Size of randomly selected testing set')
    argparser.add_argument('--num_threads', type=int, default=4,
                                help='Number of threads')
    argparser.add_argument('--use-cuda', action='store_true',
                                help='Use CUDA to perform computations')

    # Data options
    argparser.add_argument('--sr', type=int, default=10000,
                                help='Sampling rate to use for vowel data')
    argparser.add_argument('--pad_factor', type=float, default=0.0,
                                help='Amount of zero-padding applied to vowels in units of original length. For example, a value of 1 would double the sample length')

    # Simulation options
    argparser.add_argument('--c0', type=float, default=1.0,
                                help='Background wave speed')
    argparser.add_argument('--c1', type=float, default=0.9,
                                help='Second wave speed value used with --c0 when --binarized')
    argparser.add_argument('--Nx', type=int, default=140,
                                help='Number of grid cells in x-dimension of simulation domain')
    argparser.add_argument('--Ny', type=int, default=140,
                                help='Number of grid cells in y-dimension of simulation domain')
    argparser.add_argument('--dt', type=float, default=0.707,
                                help='Time step (spatial step size is determined automatically)')
    argparser.add_argument('--px', type=int, nargs='*',
                                help='Probe x-coordinates in grid cells')
    argparser.add_argument('--py', type=int, nargs='*',
                                help='Probe y-coordinates in grid cells')
    argparser.add_argument('--pd', type=int, default=30,
                                help='Spacing, in number grid cells, between probe points')
    argparser.add_argument('--src_x', type=int, default=None,
                                help='Source x-coordinate in grid cells')
    argparser.add_argument('--src_y', type=int, default=None,
                                help='Source y-coordinate in grid cells')
    argparser.add_argument('--binarized', action='store_true',
                                help='Binarize the distribution of wave speed between --c0 and --c1')
    argparser.add_argument('--pml_N', type=int, default=20,
                                help='PML thickness in grid cells')
    argparser.add_argument('--pml_p', type=float, default=4.0,
                                help='PML polynomial order')
    argparser.add_argument('--pml_max', type=float, default=3.0,
                                help='PML max dampening')
    
    args = argparser.parse_args()

    if args.use_cuda and torch.cuda.is_available():
        args.dev = torch.device('cuda')
    else:
        args.dev = torch.device('cpu')

    torch.set_num_threads(args.num_threads)

    ### Print args summary
    for i in vars(args):
        print('%16s = %s' % (i, vars(args)[i]))
    print('\n')

    ### Load data
    directories_str = ("./data/vowels/a/",
                       "./data/vowels/e/",
                       "./data/vowels/o/")

    N_classes = len(directories_str)

    x_train, x_test, y_train, y_test = load_all_vowels(directories_str, sr=args.sr,
                                                                        normalize=True, 
                                                                        train_size=args.train_size, 
                                                                        test_size=args.test_size, 
                                                                        pad_factor=args.pad_factor)
    
    x_train = x_train.to(args.dev)
    x_test  = x_test.to(args.dev)
    y_train = y_train.to(args.dev)
    y_test  = y_test.to(args.dev)

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    ### Define model
    px, py = setup_probe_coords(N_classes, args.px, args.py, args.pd, args.Nx, args.Ny, args.pml_N)
    src_x, src_y = setup_src_coords(args.src_x, args.src_y, args.Nx, args.Ny, args.pml_N)

    model = WaveCell(args.dt, args.Nx, args.Ny, src_x, src_y, px, py, pml_N=args.pml_N, pml_p=args.pml_p, pml_max=args.pml_max, c0=args.c0, c1=args.c1, binarized=args.binarized)
    model.to(args.dev)

    ### Train
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    history = train(model, optimizer, criterion, train_dl, test_dl, args.N_epochs, args.batch_size)
    
    ### Save
    save_model(model, args.name, history, args)
    
    ### Print confusion matrix
    cm_test = calc_cm(model, test_dl)
    cm_train = calc_cm(model, train_dl)

    print("Confusion matrix for training dataset:")
    print(cm_train)
    print('\n')

    print("Confusion matrix for testing dataset:")
    print(cm_test)
    print('\n')
