"""Perform vowel recognition training.
"""

import torch
import wavetorch

import argparse
import yaml
import time

import numpy as np
import sklearn
import skorch

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

probes  = [wavetorch.IntensityProbe(px[j], py[j]) for j in range(0,len(px))]
sources = [wavetorch.Source(src_x, src_y)]

### Perform training
net = skorch.NeuralNetClassifier(
    module=wavetorch.WaveCell,

    # Training configuration
    max_epochs=cfg['training']['N_epochs'],
    batch_size=cfg['training']['batch_size'],
    lr=cfg['training']['lr'],
    train_split=skorch.dataset.CVSplit(cfg['training']['N_folds'], stratified=True, random_state=cfg['seed']),
    optimizer=torch.optim.Adam,
    criterion=torch.nn.CrossEntropyLoss,
    callbacks=[ClipDesignRegion],
    callbacks__print_log__keys_ignored=None,

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
    module__sources=sources
    )

X, Y, _ = wavetorch.data.load_all_vowels(cfg['data']['vowels'], gender=cfg['data']['gender'], sr=cfg['data']['sr'], normalize=True, max_samples=cfg['training']['max_samples'], random_state=cfg['seed'])
X = torch.nn.utils.rnn.pad_sequence(X).numpy().transpose()
Y = torch.nn.utils.rnn.pad_sequence(Y).argmax(dim=0, keepdim=False).numpy().transpose()

# TODO: Need to implement cropping of the training samples inside the data loader
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.25, random_state=42)
# if cfg['data']['window_size']:
#     t_mid = int(x_train.shape[1]/2)
#     t_half_window = int(cfg['data']['window_size']/2)
#     x_train = x_train[:, (t_mid-t_half_window):(t_mid+t_half_window)]

class CroppedDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

model = net.fit(X, Y)
