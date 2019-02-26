import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import confusion_matrix

import torch


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean().item()


def calc_cm(model, train_dl, test_dl, silent=False):
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
            if not silent: print("cm: processing training batch %d" % i)
            i += 1

        y_pred = torch.cat(list_yb_pred, dim=0)
        y_truth = torch.cat(list_yb, dim=0)

        cm_train = confusion_matrix(y_truth.argmax(dim=1).numpy(), y_pred.argmax(dim=1).numpy())

        list_yb_pred = []
        list_yb = []
        i = 1
        for xb, yb in test_dl:
            yb_pred = integrate_probes(model(xb))
            list_yb_pred.append(yb_pred)
            list_yb.append(yb)
            if not silent: print("cm: processing validation batch %d" % i)
            i += 1

        y_pred = torch.cat(list_yb_pred, dim=0)
        y_truth = torch.cat(list_yb, dim=0)

        cm_test = confusion_matrix(y_truth.argmax(dim=1).numpy(), y_pred.argmax(dim=1).numpy())

    return cm_train, cm_test


# Plot the final c distribution, which is the local propagation speed
def plot_c(model):       
    plt.figure()
    plt.imshow(np.sqrt(model.c2.detach().numpy()).transpose())
    plt.colorbar()
    plt.show(block=False)
