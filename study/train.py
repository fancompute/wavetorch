import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.nn.functional import pad

from wavetorch import *

import argparse
import time

if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name', type=str, default=None)
    argparser.add_argument('--N_epochs', type=int, default=5)
    argparser.add_argument('--Nx', type=int, default=140)
    argparser.add_argument('--Ny', type=int, default=140)
    argparser.add_argument('--dt', type=float, default=0.707)
    argparser.add_argument('--probe_space', type=int, default=30)
    argparser.add_argument('--probe_x', type=int, default=100)
    argparser.add_argument('--probe_y', type=int, default=40)
    argparser.add_argument('--src_x', type=int, default=40)
    argparser.add_argument('--src_y', type=int, default=70)
    argparser.add_argument('--sr', type=int, default=5000)
    argparser.add_argument('--learning_rate', type=float, default=0.01)
    argparser.add_argument('--batch_size', type=int, default=3)
    argparser.add_argument('--train_size', type=int, default=3)
    argparser.add_argument('--test_size', type=int, default=3)
    argparser.add_argument('--num_threads', type=int, default=4)
    argparser.add_argument('--pad_factor', type=float, default=1.0)
    argparser.add_argument('--c0', type=float, default=1.0)
    argparser.add_argument('--c1', type=float, default=0.9)
    argparser.add_argument('--use-cuda', action='store_true')
    argparser.add_argument('--binarized', action='store_true')
    args = argparser.parse_args()

    torch.set_num_threads(args.num_threads)

    # figure out which device we're on
    if args.use_cuda and torch.cuda.is_available():
        print("Using GPU \n")
        args.dev = torch.device('cuda')
    else:
        print("Using CPU \n")
        args.dev = torch.device('cpu')

    # Each dir corresponds to a distinct class and is automatically given a corresponding one hot
    directories_str = ("./data/vowels/a/",
                       "./data/vowels/e/",
                       "./data/vowels/o/")

    N_classes = len(directories_str)

    x_train, x_test, y_train, y_test = load_all_vowels(directories_str, sr=args.sr, normalize=True, train_size=args.train_size, test_size=args.test_size, pad_factor=args.pad_factor)
    
    x_train = x_train.to(args.dev)
    x_test  = x_test.to(args.dev)
    y_train = y_train.to(args.dev)
    y_test  = y_test.to(args.dev)

    # Put tensors into Datasets and then Dataloaders to let pytorch manage batching
    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    # Setup probe coords and loss func
    probe_x = args.probe_x
    probe_y = torch.arange(args.probe_y, args.probe_y + N_classes*args.probe_space, args.probe_space)

    criterion = torch.nn.CrossEntropyLoss()

    # Define model
    model = WaveCell(args.dt, args.Nx, args.Ny, args.src_x, args.src_y, probe_x, probe_y, c0=args.c0, c1=args.c1, binarized=args.binarized)
    model.to(args.dev)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Run training
    for i in vars(args):
        print('%16s = %s' % (i, vars(args)[i]))
    print('\n')
    t_start = time.time()

    history = {"loss": [],
               "loss_avg": [],
               "acc_train": [],
               "acc_test": []}

    for epoch in range(1, args.N_epochs + 1):
        t_epoch = time.time()
        print('Epoch: %2d/%2d' % (epoch, args.N_epochs))

        num = 1
        for xb, yb in train_dl:
            # Needed to define this for LBFGS.
            # Technically, Adam doesn't require this but we can be flexible this way
            def closure():
                optimizer.zero_grad()
                loss = criterion(model(xb), yb.argmax(dim=1))
                loss.backward()
                return loss

            # Track loss
            loss = optimizer.step(closure)
            history["loss"].append(loss.item())

            model.clip_pml_rho()
            
            print(" ... Training batch   %2d/%2d   |   loss = %.3e" % (num, len(train_dl), history["loss"][-1]))
            num += 1


        print(" ... Computing accuracies ")
        with torch.no_grad():
            acc_tmp = []
            num = 1
            for xb, yb in train_dl:
                acc_tmp.append( accuracy(model(xb), yb.argmax(dim=1)) )
                print(" ... Training %2d/%2d " % (num, len(test_dl)))
                num += 1

            history["acc_train"].append( np.mean(acc_tmp) )

            acc_tmp = []
            num = 1
            for xb, yb in test_dl:
                acc_tmp.append( accuracy(model(xb), yb.argmax(dim=1)) )
                print(" ... Testing  %2d/%2d " % (num, len(test_dl)))
                num += 1

            history["acc_test"].append( np.mean(acc_tmp) )

        # Log metrics
        
        history["loss_avg"].append( np.mean(history["loss"][-args.batch_size:]) )

        print(" ... ")
        print(' ... elapsed time: %4.1f sec   |   loss = %.3e   accuracy = %.4f (train) / %.4f (test) \n' % 
                (time.time()-t_epoch, history["loss_avg"][-1], history["acc_train"][-1], history["acc_test"][-1]))

    # Finished training
    print('Total time: %.1f min\n' % ((time.time()-t_start)/60))
    
    save_model(model, args.name, history, args)
    
    # Calculate and print confusion matrix
    cm_test = calc_cm(model, test_dl)
    cm_train = calc_cm(model, train_dl)

    print("Confusion matrix for training dataset:")
    print(cm_train)
    print('\n')

    print("Confusion matrix for testing dataset:")
    print(cm_test)
    print('\n')
