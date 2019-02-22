import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim

from wavetorch.data import load_all_vowels

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math, random

from torch.nn.utils.rnn import pad_sequence

class CustomRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_first=True, W_scale=1e-1):
        super(CustomRNN, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.W1 = torch.nn.Parameter((torch.rand(hidden_size, input_size)-torch.rand(hidden_size, input_size))*W_scale)
        self.W2 = torch.nn.Parameter((torch.rand(hidden_size, hidden_size)-torch.rand(hidden_size, hidden_size))*W_scale)
        self.W3 = torch.nn.Parameter((torch.rand(output_size, hidden_size)-torch.rand(output_size, hidden_size))*W_scale)

    def forward(self, x):
        h1 = torch.zeros(y_true.shape[0], self.hidden_size)
        ys = []

        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
            h1 = (torch.matmul(self.W2, h1.t()) + torch.matmul(self.W1, xi.t())).t()
            y  = torch.matmul(self.W3, h1.t())
            ys.append(y)

        ys = torch.stack(ys, dim=1)
        return ys

def integrate_y(y, scale=50):
    return scale * torch.sum(torch.pow(torch.abs(y), 2), dim=1).t()


if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--N_epochs', type=int, default=20)
    argparser.add_argument('--learning_rate', type=float, default=0.1)
    argparser.add_argument('--hidden_size', type=float, default=100)
    argparser.add_argument('--sr', type=float, default=1000)
    argparser.add_argument('--use-cuda', action='store_true')
    args = argparser.parse_args()

    if args.use_cuda and torch.cuda.is_available():
        print("Using GPU...")
        args.dev = torch.device('cuda')
    else:
        print("Using CPU...")
        args.dev = torch.device('cpu')

    directories_str = ("./data/vowels/a", "./data/vowels/e/", "./data/vowels/o/")
    x, y_true = load_all_vowels(directories_str, sr=args.sr, normalize=True)
    y_true = y_true.long()

    # # --- Define model
    model = CustomRNN(1, y_true.shape[1], args.hidden_size, W_scale=0.01)
    model.to(args.dev)

    y = model(x)
    t = np.arange(0, y.shape[1])
    plt.figure()
    plt.plot(t, np.square(np.abs(y[:, :, 1].t().detach().numpy())))
    plt.title("Before training")
    plt.show(block=False)

    # # --- Define optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate)

    # # --- Run training
    print("Training for %d epochs..." % args.N_epochs)
    t_start = time.time()
    for epoch in range(1, args.N_epochs + 1):
        t_epoch = time.time()
        def closure():
            optimizer.zero_grad()
            y = model(x)
            y_ = integrate_y(y)
            loss = criterion(y_, torch.max(y_true, 1)[1])
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        print('Epoch: %d/%d %d%%  |  %.1f sec  |  L = %.3e' % (epoch, args.N_epochs, epoch/args.N_epochs*100, time.time()-t_epoch, loss))

    print('Total time: %.1f min' % ((time.time()-t_start)/60))

    y = model(x)
    t = np.arange(0, y.shape[1])
    plt.figure()
    plt.plot(t, np.square(np.abs(y[:, :, 1].t().detach().numpy())))
    plt.title("After training")
    plt.show(block=False)
