import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from wavetorch.wave import WaveCell
from wavetorch.data import load_all_vowels
from wavetorch.plot import plot_cm

from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.nn.functional import pad
import argparse
import time

from sklearn.metrics import confusion_matrix

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean().item()

# Plot the final c distribution, which is the local propagation speed
def plot_c(model):       
    plt.figure()
    plt.imshow(np.sqrt(model.c2.detach().numpy()).transpose())
    plt.colorbar()
    plt.show(block=False)

if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, required=True)
    argparser.add_argument('--sr', type=int, default=5000)
    argparser.add_argument('--ratio_train', type=float, default=0.5)
    argparser.add_argument('--batch_size', type=int, default=10)
    argparser.add_argument('--num_of_each', type=int, default=2)
    argparser.add_argument('--num_threads', type=int, default=4)
    argparser.add_argument('--pad_fact', type=float, default=1.0)
    args = argparser.parse_args()

    torch.set_num_threads(args.num_threads)

    # Load the model
    model = torch.load(args.model)

    # Load the data
    directories_str = ("./data/vowels/a/",
                       "./data/vowels/e/",
                       "./data/vowels/o/")

    x, y_labels = load_all_vowels(directories_str, sr=args.sr, normalize=True, num_of_each=args.num_of_each)
    x = pad(x, (1, int(x.shape[1] * args.pad_fact)))
    N_samples, N_classes = y_labels.shape

    full_ds = TensorDataset(x, y_labels)
    train_ds, test_ds = random_split(full_ds, [int(args.ratio_train*N_samples), N_samples-int(args.ratio_train*N_samples)])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    def integrate_probes(y):
        I = torch.sum(torch.abs(y[:, :, model.probe_x, model.probe_y]).pow(2), dim=1)
        return I / torch.sum(I, dim=1, keepdim=True)

    with torch.no_grad():
        list_yb_pred = []
        list_yb = []
        i = 1
        for xb, yb in train_dl:
            yb_pred = integrate_probes(model(xb))
            list_yb_pred.append(yb_pred)
            list_yb.append(yb)
            print("Processing training batch %d" % i)
            i += 1

        y_pred = torch.cat(list_yb_pred, dim=0)
        y_truth = torch.cat(list_yb, dim=0)

        cm = confusion_matrix(y_truth.argmax(dim=1).numpy(), y_pred.argmax(dim=1).numpy())
        cm_train = cm / cm.sum(axis=1).transpose()

        list_yb_pred = []
        list_yb = []
        i = 1
        for xb, yb in test_dl:
            yb_pred = integrate_probes(model(xb))
            list_yb_pred.append(yb_pred)
            list_yb.append(yb)
            print("Processing validation batch %d" % i)
            i += 1

        y_pred = torch.cat(list_yb_pred, dim=0)
        y_truth = torch.cat(list_yb, dim=0)

        cm = confusion_matrix(y_truth.argmax(dim=1).numpy(), y_pred.argmax(dim=1).numpy())
        cm_test = cm / cm.sum(axis=1).transpose()

    plot_cm(cm_train, title="Training")
    plot_cm(cm_test, title="Validation")
    plt.show()


