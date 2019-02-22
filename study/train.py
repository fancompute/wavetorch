import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from wavetorch.wave import WaveCell
from wavetorch.data import load_all_vowels

from torch.utils.data import TensorDataset, random_split, DataLoader

import argparse
import time

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# Plot the final c distribution, which is the local propagation speed
def plot_c(model):       
    plt.figure()
    plt.imshow(np.sqrt(model.c2.detach().numpy()).transpose())
    plt.colorbar()
    plt.show(block=False)

if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--N_epochs', type=int, default=5)
    argparser.add_argument('--Nx', type=int, default=160)
    argparser.add_argument('--Ny', type=int, default=90)
    argparser.add_argument('--dt', type=float, default=0.707)
    argparser.add_argument('--probe_space', type=int, default=15)
    argparser.add_argument('--probe_x', type=int, default=110)
    argparser.add_argument('--probe_y', type=int, default=30)
    argparser.add_argument('--src_x', type=int, default=40)
    argparser.add_argument('--src_y', type=int, default=45)
    argparser.add_argument('--sr', type=int, default=5000)
    argparser.add_argument('--learning_rate', type=float, default=0.01)
    argparser.add_argument('--ratio_train', type=float, default=0.5)
    argparser.add_argument('--batch_size', type=int, default=10)
    argparser.add_argument('--num_of_each', type=int, default=2)
    argparser.add_argument('--use-cuda', action='store_true')
    args = argparser.parse_args()

    if args.use_cuda and torch.cuda.is_available():
        print("Using GPU...")
        args.dev = torch.device('cuda')
    else:
        print("Using CPU...")
        args.dev = torch.device('cpu')

    h  = args.dt * 2.01 / 1.0

    directories_str = ("./data/vowels/a/",
                       "./data/vowels/e/",
                       "./data/vowels/o/")

    x, y_labels = load_all_vowels(directories_str, sr=args.sr, normalize=True, num_of_each=args.num_of_each)
    N_samples, N_classes = y_labels.shape

    full_ds = TensorDataset(x, y_labels)
    train_ds, test_ds = random_split(full_ds, [int(args.ratio_train*N_samples), N_samples-int(args.ratio_train*N_samples)])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    # --- Setup probe coords and loss func
    probe_x = args.probe_x
    probe_y = torch.arange(args.probe_y, args.probe_y + N_classes*args.probe_space, args.probe_space)

    def integrate_probes(y):
        return torch.sum(torch.abs(y[:,:,probe_x, probe_y]).pow(2), dim=1)

    criterion = torch.nn.CrossEntropyLoss()

    # --- Define model
    model = WaveCell(args.dt, args.Nx, args.Ny, h, args.src_x, args.src_y, probe_x, probe_y, pml_max=3, pml_p=4.0, pml_N=20)
    model.to(args.dev)

    model.animate(x)

    # --- Define optimizer
    optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate)

    # --- Run training
    print("Using a sample rate of %d Hz (sequence length of %d)" % (args.sr, x.shape[1]))
    print("Using %d total samples (%d of each vowel)" % (len(train_ds)+len(test_ds), args.num_of_each) )
    print("   %d for training" % len(train_ds))
    print("   %d for validation" % len(test_ds))
    print(" --- ")
    print("Now begining training for %d epochs ..." % args.N_epochs)
    t_start = time.time()
    for epoch in range(1, args.N_epochs + 1):
        t_epoch = time.time()

        loss_batches = []
        test_acc = []
        train_acc = []

        for xb, yb in train_dl:
            def closure():
                optimizer.zero_grad()
                loss = criterion(integrate_probes(model(xb)), yb.argmax(dim=1))
                # loss += torch.sum(model.c2 < 0.1**2) + 1.0
                # loss += torch.sum(model.c2 > 1.1**2) + 1.0
                loss.backward()
                return loss

            # Track loss
            loss = optimizer.step(closure)
            loss_batches.append(loss.item())

            # Track train accuracy
            with torch.no_grad():
                train_acc.append( accuracy(integrate_probes(model(xb)), yb.argmax(dim=1)) )

        # Track test accuracy
        with torch.no_grad():
            for xb, yb in test_dl:
                test_acc.append( accuracy(integrate_probes(model(xb)), yb.argmax(dim=1)) )

        print('Epoch: %2d/%2d  %4.1f sec  |  L = %.3e ,  train_acc = %0.3f ,  val_acc = %.3f ' % 
            (epoch, args.N_epochs, time.time()-t_epoch, np.mean(loss_batches), np.mean(train_acc), np.mean(test_acc)))

    print(" --- ")
    print('Total time: %.1f min' % ((time.time()-t_start)/60))

    model.show()
