import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def to_tensor(x, dtype=None):
	dtype = dtype if dtype is not None else torch.get_default_dtype()
	if type(x) is np.ndarray:
		return torch.from_numpy(x).type(dtype=dtype)
	else:
		return torch.tensor(x, dtype=dtype)


def set_dtype(dtype=None):
	if dtype == 'float32' or dtype is None:
		torch.set_default_dtype(torch.float32)
	elif dtype == 'float64':
		torch.set_default_dtype(torch.float64)
	else:
		raise ValueError('Unsupported data type: %s; should be either float32 or float64' % dtype)


def window_data(X, window_length):
	"""Window the sample, X, to a length of window_length centered at the middle of the original sample
	"""
	return X[int(len(X) / 2 - window_length / 2):int(len(X) / 2 + window_length / 2)]


def accuracy_onehot(y_pred, y_label):
	"""Compute the accuracy for a onehot
	"""
	return (y_pred.argmax(dim=1) == y_label).float().mean().item()


def normalize_power(X):
	return X / torch.sum(X, dim=1, keepdim=True)
