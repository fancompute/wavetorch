import argparse
import yaml
import time
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_ as clip_grad
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from study.vanilla.models import CustomRNN, CustomRes, CustomLSTM

from wavetorch.data import load_all_vowels
from wavetorch.core.utils import accuracy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math, random

import os

from torch.nn.utils.rnn import pad_sequence

def compute_acc(model, elements_set):
    acc_tmp = []
    num = 1
    for xb, yb in elements_set:
        y_ = norm_int_y(model(xb))
        acc_tmp.append(accuracy(y_, yb.argmax(dim=1)))
    return np.mean(np.array(acc_tmp))

def norm_int_y(y, scale=50):
    # Input dim: [N_batch, T, N_classes]
    # Output dim: [N_batch, N_classes]
    # Integrates abs(y)^2 along the T-dimension and normalizes it such that the output adds to 1 along the N_classes dimension
    
    y_int = torch.sum(torch.pow(torch.abs(y), 2), dim=1)
    return y_int / torch.sum(y_int, dim=1, keepdim=True)  

def save_model(model, name, savedir='./study/vanilla/data/', history=None, args=None):
    str_filename = name +  '.pt'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    str_savepath = savedir + str_filename
    dsave = {"model_state": model.state_dict(),
             "history": history,
             "args": args}
    print("Saving model to %s" % str_savepath)
    torch.save(dsave, str_savepath)

def load_model(str_filename):
    print("Loading model from %s" % str_filename)
    data = torch.load(str_filename)
    model_state = data['model_state']
    model = CustomRNN(model_state['W1'].size(1), model_state['W3'].size(0), model_state['W1'].size(0))
    model.load_state_dict(model_state)
    model.eval()
    return data['history'], model


def main(args):
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--use-cuda', action='store_true')
    argparser.add_argument('--config', type=str, required=True,
                            help='Config file to use')
    argparser.add_argument('--name', type=str, default=None,
                            help='Name to use when saving or loading the model file. If not specified when saving a time and date stamp is used')
    argparser.add_argument('--savedir', type=str, default='./study/vanilla/data/',
                            help='Directory in which the model file is saved. Defaults to ./study/')
    args = argparser.parse_args(args)

    if args.use_cuda and torch.cuda.is_available():
        print("Using GPU...")
        args.dev = torch.device('cuda')
    else:
        print("Using CPU...")
        args.dev = torch.device('cpu')

    print("Using configuration from %s: " % args.config)
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        print(yaml.dump(cfg, default_flow_style=False))

    if cfg['general']['rand_seed'] is not None:
        torch.manual_seed(cfg['general']['rand_seed'])
        np.random.seed(cfg['general']['rand_seed'])

    N_classes = len(cfg['data']['vowels'])
    N_epochs = cfg['training']['N_epochs']
    N_batch = cfg['training']['batch_size']
    disp_step = cfg['training']['display_step']

    # # Load the data; x_train is dim [N_train, T], while y_train is dim [N_train, N_classes]
    # x_train, x_test, y_train, y_test = load_selected_vowels(
    #     cfg['data']['vowels'],
    #     gender=cfg['data']['gender'], 
    #     sr=cfg['data']['sr'], 
    #     normalize=True, 
    #     train_size=cfg['training']['train_size'], 
    #     test_size=cfg['training']['test_size']
    #     )

    X, Y = load_all_vowels(
            cfg['data']['vowels'],
            gender=cfg['data']['gender'], 
            sr=cfg['data']['sr'], 
            normalize=True,
            max_samples=cfg['training']['max_samples']
            )

    # Some manual scaling
    mean = 0
    std = 0
    X_norm = []
    for samp in X:
        mean += samp.mean()
        std += samp.std()
    mean = mean/len(X)
    std = std/len(X)

    for samp in X:
        X_norm.append((samp - mean)/std)

    X = X_norm

    skf = StratifiedKFold(n_splits=cfg['training']['train_test_divide'], random_state=None, shuffle=True)
    samps = [y.argmax().item() for y in Y]
    num = 1

    history = {"loss_iter": [],
               "acc_train": [],
               "acc_test": [],
               "acc_epoch": []}

    acc_fin_train = []
    acc_fin_test = []

    for train_index, test_index in skf.split(np.zeros(len(samps)), samps):
        if cfg['training']['use_cross_validation']: print("Cross validation fold #%d" % num)

        if cfg['data']['window_size']:
            crop = cfg['data']['window_size']
            x_train = torch.nn.utils.rnn.pad_sequence([X[i][int(len(X[i])/2-crop/2):int(len(X[i])/2+crop/2)] for i in train_index], batch_first=True)
        else:
            x_train = torch.nn.utils.rnn.pad_sequence([X[i] for i in train_index], batch_first=True)

        x_test = torch.nn.utils.rnn.pad_sequence([X[i] for i in test_index], batch_first=True)
        y_train = torch.nn.utils.rnn.pad_sequence([Y[i] for i in train_index], batch_first=True)
        y_test = torch.nn.utils.rnn.pad_sequence([Y[i] for i in test_index], batch_first=True)

        x_train = x_train.to(args.dev)
        x_test  = x_test.to(args.dev)
        y_train = y_train.to(args.dev)
        y_test  = y_test.to(args.dev)

        # # --- Define model
        if cfg['rnn']['model']=='rnn':
            model = CustomRNN(1, N_classes, cfg['rnn']['N_hidden'], W_scale=cfg['rnn']['W_scale'], f_hidden=cfg['rnn']['f_hidden'])
        elif cfg['rnn']['model']=='rnn':
            model = CustomRes(1, N_classes, cfg['rnn']['N_hidden'], W_scale=cfg['rnn']['W_scale'], f_hidden=cfg['rnn']['f_hidden'])
        elif cfg['rnn']['model']=='lstm':
            model = CustomLSTM(1, N_classes, cfg['rnn']['N_hidden'], W_scale=cfg['rnn']['W_scale'])
        model.to(args.dev)

        # Print the total number of parameters in the model
        print("Total number of parameters in model: %d" % sum(p.numel() for p in model.parameters()))

        # # Output of the model is dim [N_train, T, N_classes]
        # y_pred = model(x_train)
        # # Plot the N_classes intensities recorded for the first training element at the three outputs for the un-optimized model
        # t = np.arange(0, y_pred.shape[1])
        # plt.figure()
        # plt.plot(t, np.square(np.abs(y_pred[1, :, :].detach().numpy())))
        # plt.title("Before training")
        # plt.show(block=False)

        # # --- Define optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'], weight_decay=2*cfg['rnn']['L2_reg'])
        if cfg['training']['lr_step'] and cfg['training']['lr_gamma']: 
            scheduler = StepLR(optimizer, step_size=cfg['training']['lr_step'],
            gamma=cfg['training']['lr_gamma'])
            scale_lr = True

        # Split data into batches
        train_ds = TensorDataset(x_train, y_train)
        test_ds  = TensorDataset(x_test, y_test)

        train_dl = DataLoader(train_ds, batch_size=N_batch, shuffle=True)
        test_dl  = DataLoader(test_ds, batch_size=N_batch)

        # # --- Run training
        print("Training for %d epochs..." % N_epochs)
        t_start = time.time()
        for epoch in range(1, N_epochs + 1):
            t_epoch = time.time()
            if scale_lr:
                scheduler.step()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                y = model(xb)
                y_ = norm_int_y(y)
                # print(y_, torch.max(yb, 1)[1])
                loss = criterion(y_, torch.max(yb, 1)[1])
                history['loss_iter'].append(loss.item())
                loss.backward()
                if cfg['rnn']['grad_clip'] is not None:
                    clip_grad(model.parameters(), cfg['rnn']['grad_clip'], norm_type=1)
                optimizer.step()

            if epoch % disp_step == 0 or epoch == 1:
                with torch.no_grad():
                    tep = time.time()-t_epoch
                    acc_train = compute_acc(model, train_dl)
                    acc_test = compute_acc(model, test_dl)
                    print('Epoch: %d/%d %d%%  | Time for last epoch %.1f sec  |  L = %.3e' 
                        % (epoch, N_epochs, epoch/N_epochs*100, tep, loss))
                    print('Training accuracy: %.2f | Testing accuracy: %.2f' 
                        % (acc_train, acc_test))
                    history['acc_train'].append(acc_train)
                    history['acc_test'].append(acc_test)
                    history['acc_epoch'].append(epoch)            

        print('Total time: %.1f min' % ((time.time()-t_start)/60))

        # Print the final training set accuracy
        with torch.no_grad():
            (acc_final_train, acc_final_test) = (compute_acc(model, train_dl), compute_acc(model, test_dl))
            history['acc_train'].append(acc_final_train)
            history['acc_test'].append(acc_final_test)
            history['acc_epoch'].append(N_epochs)  
            print("Final accuracy - train: {:.2f} %, test: {:.2f} %".format(
                acc_final_train, acc_final_test))
            acc_fin_train.append(acc_final_train)
            acc_fin_test.append(acc_final_test)

        ### Save model and results
        if args.name is None:
            args.name = time.strftime("%Y_%m_%d-%H_%M_%S")
        if (cfg['training']['prefix'] is not None) and (num == 1):
            args.name = cfg['training']['prefix'] + '_' + args.name

        if cfg['training']['use_cross_validation']:
            # If we are doing cross validation, then save this model's iteration            
            save_model(model, args.name + "_cv_" + str(num), args.savedir, history, cfg)
            num += 1
        else:
            # If not doing cross validation, save and finish
            save_model(model, args.name, args.savedir, history, cfg)
            break

    # y_pred = model(x_train)
    # # Plot the N_classes intensities recorded for the first training element at the three outputs for the optimized model
    # t = np.arange(0, y_pred.shape[1])
    # plt.figure()
    # plt.plot(t, np.square(np.abs(y_pred[1, :, :].detach().numpy())))
    # plt.title("After training")
    # plt.show(block=False)
    return (acc_fin_train, acc_fin_test)

if __name__ == '__main__':
    (acc_final_train, acc_final_test) = main(sys.argv[1:])


