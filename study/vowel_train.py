"""Perform vowel recognition training.
"""

import torch
import wavetorch
from torch.utils.data import TensorDataset, DataLoader

import argparse
import yaml
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser() 
parser.add_argument('config', type=str, 
                    help='Configuration file for geometry, training, and data preparation')
parser.add_argument('--num_threads', type=int, default=4,
                    help='Number of threads to use')
parser.add_argument('--use-cuda', action='store_true',
                    help='Use CUDA to perform computations')
parser.add_argument('--name', type=str, default=time.strftime('%Y%m%d%H%M%S'),
                    help='Name to use when saving or loading the model file. If not specified when saving a time and date stamp is used')
parser.add_argument('--savedir', type=str, default='./study/',
                    help='Directory in which the model file is saved. Defaults to ./study/')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.use_cuda and torch.cuda.is_available():
        args.dev = torch.device('cuda')
    else:
        args.dev = torch.device('cpu')

    torch.set_num_threads(args.num_threads)

    print("Using configuration from %s: " % args.config)
    with open(args.config, 'r') as ymlfile:
         cfg = yaml.load(ymlfile)
         print(yaml.dump(cfg, default_flow_style=False))

    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])

    if cfg['training']['prefix'] is not None:
        args.name = cfg['training']['prefix'] + '_' + args.name

    N_classes = len(cfg['data']['vowels'])

    X, Y, _ = wavetorch.data.load_all_vowels(cfg['data']['vowels'], gender=cfg['data']['gender'], sr=cfg['data']['sr'], normalize=True, max_samples=cfg['training']['max_samples'], random_state=cfg['seed'])

    skf = StratifiedKFold(n_splits=cfg['training']['N_folds'], random_state=cfg['seed'], shuffle=True)
    samps = [y.argmax().item() for y in Y]

    history = None
    history_model_state = []
    for num, (train_index, test_index) in enumerate(skf.split(np.zeros(len(samps)), samps)):
        if cfg['training']['cross_validation']: print("Cross Validation Fold %2d/%2d" % (num+1, cfg['training']['N_folds']))

        if cfg['data']['window_size']:
            x_train = torch.nn.utils.rnn.pad_sequence([wavetorch.utils.window_data(X[i], cfg['data']['window_size']) for i in train_index], batch_first=True)
        else:
            x_train = torch.nn.utils.rnn.pad_sequence([X[i] for i in train_index], batch_first=True)

        x_test = torch.nn.utils.rnn.pad_sequence([X[i] for i in test_index], batch_first=True)
        y_train = torch.nn.utils.rnn.pad_sequence([Y[i] for i in train_index], batch_first=True)
        y_test = torch.nn.utils.rnn.pad_sequence([Y[i] for i in test_index], batch_first=True)

        x_train = x_train.to(args.dev)
        x_test  = x_test.to(args.dev)
        y_train = y_train.to(args.dev)
        y_test  = y_test.to(args.dev)

        train_ds = TensorDataset(x_train, y_train)
        test_ds  = TensorDataset(x_test, y_test)

        train_dl = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True)
        test_dl  = DataLoader(test_ds, batch_size=cfg['training']['batch_size'])

        ### Define model
        px, py = wavetorch.utils.setup_probe_coords(
                            N_classes, cfg['geom']['px'], cfg['geom']['py'], cfg['geom']['pd'], 
                            cfg['geom']['Nx'], cfg['geom']['Ny'], cfg['geom']['pml']['N']
                            )
        src_x, src_y = wavetorch.utils.setup_src_coords(
                            cfg['geom']['src_x'], cfg['geom']['src_y'], cfg['geom']['Nx'],
                            cfg['geom']['Ny'], cfg['geom']['pml']['N']
                            )

        if cfg['geom']['use_design_region']: # Limit the design region
            design_region = torch.zeros(cfg['geom']['Nx'], cfg['geom']['Ny'], dtype=torch.uint8)
            design_region[src_x+5:np.min(px)-5] = 1 # For now, just hardcode this in
        else: # Let the design region be the enire non-PML area
            design_region = None

        geom = wavetorch.Geometry(cfg['geom']['Nx'], cfg['geom']['Ny'], h=cfg['geom']['h'], init=cfg['geom']['init'], c0=cfg['geom']['c0'], c1=cfg['geom']['c1'], design_region=design_region)
        geom.add_boundary_absorber(sigma=cfg['geom']['pml']['max'], N=cfg['geom']['pml']['N'], p=cfg['geom']['pml']['p'])
        geom.add_source(wavetorch.PointSource(geom, src_x, src_y))
        for j in range(0, len(px)):
            geom.add_probe(wavetorch.IntensityProbe(px[j], py[j], label='{}'.format(j)))

        # Define the model
        model = wavetorch.WaveCell(cfg['geom']['dt'], geom)

        model.to(args.dev)

        ### Train
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()

        history, history_model_state = wavetorch.train(
                                            model,
                                            optimizer,
                                            criterion, 
                                            train_dl, 
                                            test_dl, 
                                            cfg['training']['N_epochs'], 
                                            cfg['training']['batch_size'], 
                                            history=history,
                                            history_model_state=history_model_state,
                                            fold=num if cfg['training']['cross_validation'] else -1,
                                            name=args.name,
                                            savedir=args.savedir,
                                            accuracy=wavetorch.utils.accuracy_onehot,
                                            cfg=cfg)
        
        wavetorch.utils.save_model(model, args.name, args.savedir, history, history_model_state, cfg)

        if not cfg['training']['cross_validation']:
            break
