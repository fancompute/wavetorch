import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import confusion_matrix

import torch

import time
import os
import socket

def save_model(model, name, savedir='./study/', history=None, args=None, cm_train=None, cm_test=None, cm_train0=None, cm_test0=None):
    str_filename = name +  '.pt'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    str_savepath = savedir + str_filename
    data = {"model_state": model.state_dict(),
            "history": history,
            "args": args,
            "cm_train": cm_train,
            "cm_test": cm_test,
            "cm_train0": cm_train0,
            "cm_test0": cm_test0}
    print("Saving model to %s" % str_savepath)
    torch.save(data, str_savepath)


def load_model(str_filename):
    from .cell import WaveCell
    print("Loading model from %s" % str_filename)
    data = torch.load(str_filename)
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
    return model, data["history"], data["args"], data["cm_train"], data["cm_test"], data["cm_train0"], data["cm_test0"]


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean().item()


def calc_cm(model, dataloader, verbose=True):
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
