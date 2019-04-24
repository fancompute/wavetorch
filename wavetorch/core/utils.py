import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import confusion_matrix

import torch
import wavetorch

import time
import os
import socket

def save_model(model, name, savedir='./study/', 
               history=None, history_model_state=None, cfg=None, verbose=True):
    """Save the model state and history to a file
    """
    str_filename = name +  '.pt'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    str_savepath = savedir + str_filename
    data = {"model_state": model.state_dict(),
            "history": history,
            "history_model_state": history_model_state,
            "cfg": cfg}
    if verbose:
        print("Saving model to %s" % str_savepath)
    torch.save(data, str_savepath)


def load_model(str_filename):
    """Load a previously saved model and its history from a file
    """
    from .cell import WaveCell
    print("Loading model from %s" % str_filename)
    data = torch.load(str_filename)

    wavetorch.core.set_dtype(data["cfg"]['dtype'])
    model_state = data['model_state']
    model = WaveCell(model_state['dt'].numpy(),
                     model_state['Nx'].numpy(), 
                     model_state['Ny'].numpy(), 
                     model_state['src_x'].numpy(), 
                     model_state['src_y'].numpy(), 
                     model_state['px'].numpy(), 
                     model_state['py'].numpy())
    model.load_state_dict(model_state)
    model.eval()
    return model, data["history"], data["history_model_state"], data["cfg"]


def accuracy(out, yb):
    """Compute the accuracy
    """
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean().item()


def calc_cm(model, dataloader, verbose=True):
    """Calculate the confusion matrix
    """
    with torch.no_grad():
        list_yb_pred = []
        list_yb = []
        i = 1
        for xb, yb in dataloader:
            yb_pred = model(xb)
            list_yb_pred.append(yb_pred)
            list_yb.append(yb)
            if verbose: print("cm: processing batch %d" % i)
            i += 1

        y_pred = torch.cat(list_yb_pred, dim=0)
        y_truth = torch.cat(list_yb, dim=0)

    return confusion_matrix(y_truth.argmax(dim=1).numpy(), y_pred.argmax(dim=1).numpy())

def set_dtype(dtype=None):
    if dtype == 'float32' or dtype is None:
        torch.set_default_dtype(torch.float32)
    elif dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError('Unsupported data type: %s; should be either float32 or float64' % dtype)
