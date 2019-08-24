"""Perform vowel recognition training.

This script is a **work-in-progress** of trying to get training to work with skorch / scikit-learn

"""

import torch
import wavetorch

import argparse
import yaml
import time

import numpy as np
import sklearn
import skorch

class CroppedDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

# TODO: move this into lib
class ClipDesignRegion(skorch.callbacks.Callback):
    def on_batch_end(self, net, Xi=None, yi=None, training=None, **kwargs):
        if training:
            net.module_.clip_to_design_region()

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

### Define the geometry
probes = wavetorch.utils.setup_probe_coords(
                    N_classes, cfg['geom']['px'], cfg['geom']['py'], cfg['geom']['pd'], 
                    cfg['geom']['Nx'], cfg['geom']['Ny'], cfg['geom']['pml']['N']
                    )
source = wavetorch.utils.setup_src_coords(
                    cfg['geom']['src_x'], cfg['geom']['src_y'], cfg['geom']['Nx'],
                    cfg['geom']['Ny'], cfg['geom']['pml']['N']
                    )

design_region = torch.zeros(cfg['geom']['Nx'], cfg['geom']['Ny'], dtype=torch.uint8)
design_region[source[0].x.item()+5:probes[0].x.item()-5] = 1

def my_train_split(ds, y):
    return ds, skorch.dataset.Dataset(corpus.valid[:200], y=None)

### Perform training
net = skorch.NeuralNetClassifier(
    module=wavetorch.WaveCell,

    # Training configuration
    max_epochs=cfg['training']['N_epochs'],
    batch_size=cfg['training']['batch_size'],
    lr=cfg['training']['lr'],
    # train_split=skorch.dataset.CVSplit(cfg['training']['N_folds'], stratified=True, random_state=cfg['seed']),
    optimizer=torch.optim.Adam,
    criterion=torch.nn.CrossEntropyLoss,
    callbacks=[
        ClipDesignRegion,
        skorch.callbacks.EpochScoring('accuracy', lower_is_better=False, on_train=True, name='train_acc'),
        skorch.callbacks.Checkpoint(monitor=None, fn_prefix='1234_', dirname='test', f_params="params_{last_epoch[epoch]}.pt", f_optimizer='optimizer.pt', f_history='history.json')
        ],
    callbacks__print_log__keys_ignored=None,
    train_split=None,

    # These al get passed as options to WaveCell
    module__Nx=cfg['geom']['Nx'],
    module__Ny=cfg['geom']['Ny'],
    module__h=cfg['geom']['h'],
    module__dt=cfg['geom']['dt'],
    module__init=cfg['geom']['init'], 
    module__c0=cfg['geom']['c0'], 
    module__c1=cfg['geom']['c1'], 
    module__sigma=cfg['geom']['pml']['max'], 
    module__N=cfg['geom']['pml']['N'], 
    module__p=cfg['geom']['pml']['p'],
    module__design_region=design_region,
    module__output_probe=True,
    module__probes=probes,
    module__sources=source
    )

X, Y, _ = wavetorch.data.load_all_vowels(cfg['data']['vowels'], gender=cfg['data']['gender'], sr=cfg['data']['sr'], normalize=True, max_samples=cfg['training']['max_samples'], random_state=cfg['seed'])
X = torch.nn.utils.rnn.pad_sequence(X).numpy().transpose()
Y = torch.nn.utils.rnn.pad_sequence(Y).argmax(dim=0, keepdim=False).numpy().transpose()

# TODO(ian): Need to implement cropping of the training samples inside the data loader
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.25, random_state=42)
if cfg['data']['window_size']:
    t_mid = int(X.shape[1]/2)
    t_half_window = int(cfg['data']['window_size']/2)
    X = X[:, (t_mid-t_half_window):(t_mid+t_half_window)]

from sklearn.model_selection import cross_validate, StratifiedKFold
y_pred = cross_validate(net, X, Y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=None))

# model = net.fit(X, Y)
