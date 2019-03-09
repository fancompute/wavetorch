import torch
from torch.nn.functional import conv2d
from torch import tanh
import time
import numpy as np
from .utils import accuracy

def train(model, optimizer, criterion, train_dl, test_dl, N_epochs, batch_size):
    
    history = {"loss_iter": [],
               "loss_train": [],
               "loss_test": [],
               "acc_train": [],
               "acc_test": []}

    t_start = time.time()
    for epoch in range(0, N_epochs + 1):
        t_epoch = time.time()
        print('Epoch: %2d/%2d' % (epoch, N_epochs))

        if epoch == 0:
            print(" ... NOTE: We only characterize the starting structure on epoch 0 (no optimizer step is taken)")

        num = 1
        for xb, yb in train_dl:
            def closure():
                optimizer.zero_grad()
                loss = criterion(model(xb), yb.argmax(dim=1))
                loss.backward()
                return loss

            if epoch == 0: # Don't take a step and just characterize the starting structure
                with torch.no_grad():
                    loss = criterion(model(xb), yb.argmax(dim=1))
            else: # Take an optimization step
                loss = optimizer.step(closure)
                model.clip_to_design_region()

            history["loss_iter"].append(loss.item())
            
            print(" ... Training batch   %2d/%2d   |   loss = %.3e" % (num, len(train_dl), history["loss_iter"][-1]))
            num += 1

        history["loss_train"].append( np.mean(history["loss_iter"][-batch_size:]) )

        print(" ... Computing accuracies ")
        with torch.no_grad():
            acc_tmp = []
            num = 1
            for xb, yb in train_dl:
                acc_tmp.append( accuracy(model(xb), yb.argmax(dim=1)) )
                print(" ... Training %2d/%2d " % (num, len(train_dl)))
                num += 1

            history["acc_train"].append( np.mean(acc_tmp) )

            acc_tmp = []
            loss_tmp = []
            num = 1
            for xb, yb in test_dl:
                pred = model(xb)
                loss_tmp.append( criterion(pred, yb.argmax(dim=1)) )
                acc_tmp.append( accuracy(pred, yb.argmax(dim=1)) )
                print(" ... Testing  %2d/%2d " % (num, len(test_dl)))
                num += 1

        history["loss_test"].append( np.mean(loss_tmp) )
        history["acc_test"].append( np.mean(acc_tmp) )

        print(" ... ")
        print(' ... elapsed time: %4.1f sec   |   loss = %.4e (train) / %.4e (test)   accuracy = %.4f (train) / %.4f (test) \n' % 
                (time.time()-t_epoch, history["loss_train"][-1], history["loss_test"][-1], history["acc_train"][-1], history["acc_test"][-1]))

    print('Total time: %.1f min\n' % ((time.time()-t_start)/60))

    return history
