import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import confusion_matrix

import torch
import wavetorch

import time
import os
import socket

def to_tensor(x):
    if type(x) is np.ndarray: 
        x = torch.from_numpy(x)
    else:
        x = torch.tensor(x)

    return x


def setup_src_coords(src_x, src_y, Nx, Ny, Npml):
    if (src_x is not None) and (src_y is not None):
        # Coordinate are specified
        return [wavetorch.Source(src_x, src_y)]
    else:
        # Center at left
        return [wavetorch.Source(Npml+20, int(Ny/2))]


def setup_probe_coords(N_classes, px, py, pd, Nx, Ny, Npml):
    if (py is not None) and (px is not None):
        # All probe coordinate are specified
        assert len(px) == len(py), "Length of px and py must match"

        return [wavetorch.IntensityProbe(px[j], py[j]) for j in range(0,len(px))]

    if (py is None) and (pd is not None):
        # Center the probe array in y
        span = (N_classes-1)*pd
        y0 = int((Ny-span)/2)
        assert y0 > Npml, "Bottom element of array is inside the PML"
        y = [y0 + i*pd for i in range(N_classes)]

        if px is not None:
            assert len(px) == 1, "If py is not specified then px must be of length 1"
            x = [px[0] for i in range(N_classes)]
        else:
            x = [Nx-Npml-20 for i in range(N_classes)]

        return [wavetorch.IntensityProbe(x[j], y[j]) for j in range(0,len(x))]

    raise ValueError("px = {}, py = {}, pd = {} is an invalid probe configuration".format(pd))


def window_data(X, window_length):
    """Window the sample, X, to a length of window_length centered at the middle of the original sample
    """
    return X[int(len(X)/2-window_length/2):int(len(X)/2+window_length/2)]

def save_model(model,
               name, 
               savedir='./study/', 
               history=None, 
               history_model_state=None, 
               cfg=None, 
               verbose=True):
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

    print("Loading model from %s" % str_filename)

    data = torch.load(str_filename)

    # Set the type for floats from the save
    try:
        set_dtype(data['cfg']['dtype'])
    except:
        pass

    # Setup a blank model
    model = wavetorch.WaveCell(Nx=data['model_state']['Nx'].item(),
                               Ny=data['model_state']['Ny'].item(),
                               dt=data['model_state']['dt'].item(), 
                               h =data['model_state']['h'].item())

    # Load in everything other than the probes and sources (because pytorch is too stupid to know how to handle ModuleLists)
    loaded_dict = {k: data['model_state'][k] for k in data['model_state'] if 'probes' not in k and 'sources' not in k}
    model.load_state_dict(loaded_dict)

    # Parse out the probe and source coords
    px = [data['model_state'][k].item() for k in data['model_state'] if 'probes' in k and 'x' in k]
    py = [data['model_state'][k].item() for k in data['model_state'] if 'probes' in k and 'y' in k]
    sx = [data['model_state'][k].item() for k in data['model_state'] if 'sources' in k and 'x' in k]
    sy = [data['model_state'][k].item() for k in data['model_state'] if 'sources' in k and 'y' in k]

    # Manually add the probes and sources
    for (x, y) in zip(px, py):
        model.add_probe(wavetorch.IntensityProbe(x,y))

    for (x, y) in zip(sx, sy):
        model.add_source(wavetorch.Source(x,y))
    
    # Put into eval mode (doesn't really matter for us but whatever)
    model.eval()

    return model, data['history'], data['history_model_state'], data['cfg']


def accuracy_onehot(y_pred, y_label):
    """Compute the accuracy for a onehot
    """
    return (y_pred.argmax(dim=1) == y_label).float().mean().item()


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
