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

from sklearn.model_selection import StratifiedKFold

from wavetorch.data import load_selected_vowels, load_all_vowels
from wavetorch.core.utils import accuracy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math, random

import os

from torch.nn.utils.rnn import pad_sequence

class CustomRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_first=True, W_scale=1e-1, f_hidden=None):
        super(CustomRNN, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.f_hidden = f_hidden

        self.W1 = torch.nn.Parameter((torch.rand(hidden_size, input_size)-0.5)*W_scale)
        self.W2 = torch.nn.Parameter((torch.rand(hidden_size, hidden_size)-0.5)*W_scale)
        self.W3 = torch.nn.Parameter((torch.rand(output_size, hidden_size)-0.5)*W_scale)
        self.b_h = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        h1 = torch.zeros(x.shape[0], self.hidden_size)
        ys = []

        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
            h1 = (torch.matmul(self.W2, h1.t()) + torch.matmul(self.W1, xi.t())).t() + self.b_h
            if self.f_hidden is not None:
                h1 = getattr(F, self.f_hidden)(h1)
            y = torch.matmul(self.W3, h1.t()).t()
            ys.append(y)

        ys = torch.stack(ys, dim=1)
        return ys

    def compute_acc(self, x, y):
        y_pred = self.forward(x)
        y_norm = norm_int_y(y_pred)
        return accuracy(y_norm, y.argmax(dim=1))


def norm_int_y(y, scale=50):
    # Input dim: [N_batch, T, N_classes]
    # Output dim: [N_batch, N_classes]
    # Integrates abs(y)^2 along the T-dimension and normalizes it such that the output adds to 1 along the N_classes dimension
    
    y_int = torch.sum(torch.pow(torch.abs(y), 2), dim=1)
    return y_int / torch.sum(y_int, dim=1, keepdim=True)  

def save_model(model, name, savedir='./study/vanilla/', history=None, args=None):
    str_filename = name +  '.pt'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    str_savepath = savedir + str_filename
    dsave = {"model_state": model.state_dict(),
             "history": history,
             "args": args}
    print("Saving model to %s" % str_savepath)
    torch.save(dsave, str_savepath)


def main(args):
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--use-cuda', action='store_true')
    argparser.add_argument('--config', type=str, required=True,
                            help='Config file to use')
    argparser.add_argument('--name', type=str, default=None,
                            help='Name to use when saving or loading the model file. If not specified when saving a time and date stamp is used')
    argparser.add_argument('--savedir', type=str, default='./study/vanilla/',
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

        x_train = torch.nn.utils.rnn.pad_sequence([X[i] for i in train_index], batch_first=True)
        x_test = torch.nn.utils.rnn.pad_sequence([X[i] for i in test_index], batch_first=True)
        y_train = torch.nn.utils.rnn.pad_sequence([Y[i] for i in train_index], batch_first=True)
        y_test = torch.nn.utils.rnn.pad_sequence([Y[i] for i in test_index], batch_first=True)

        x_train = x_train.to(args.dev)
        x_test  = x_test.to(args.dev)
        y_train = y_train.to(args.dev)
        y_test  = y_test.to(args.dev)

        # # --- Define model
        model = CustomRNN(1, N_classes, cfg['rnn']['N_hidden'], W_scale=cfg['rnn']['W_scale'], f_hidden=cfg['rnn']['f_hidden'])
        model.to(args.dev)

        # Print the starting training set accuracy
        print("Initial accuracy - train: {:.2f} %, test: {:.2f} %".format(
            model.compute_acc(x_train, y_train), model.compute_acc(x_test, y_test)))

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
            for xb, yb in train_dl:
                optimizer.zero_grad()
                y = model(xb)
                y_ = norm_int_y(y)
                loss = criterion(y_, torch.max(yb, 1)[1])
                history['loss_iter'].append(loss)
                loss.backward()
                if cfg['rnn']['grad_clip'] is not None:
                    clip_grad(model.parameters(), cfg['rnn']['grad_clip'], norm_type=1)
                optimizer.step()

            if epoch % disp_step == 0 or epoch == 1:
                with torch.no_grad():
                    tep = time.time()-t_epoch
                    with torch.no_grad():
                        acc_train = model.compute_acc(x_train, y_train)
                        acc_test = model.compute_acc(x_test, y_test)
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
            (acc_final_train, acc_final_test) = (model.compute_acc(x_train, y_train), model.compute_acc(x_test, y_test))
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
            args.name += "_cv_" + str(num)
            save_model(model, args.name, args.savedir, history, cfg)
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


