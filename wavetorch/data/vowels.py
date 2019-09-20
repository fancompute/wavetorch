import glob
import math
import os
import random

import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def normalize_vowel(wav_data):
	"""Normalize the amplitude of a vowel waveform
	"""
	total_power = np.square(wav_data).sum()

	return wav_data / np.sqrt(total_power)


def load_vowel(file, sr=None, normalize=True):
	"""Use librosa to to load a single vowel with a specified sample rate
	"""

	data, rate = librosa.load(file, sr=sr)

	if normalize:
		data = normalize_vowel(data)

	return data


def load_all_vowels(str_classes, gender='both', sr=None, normalize=True, dir='data/vowels/', ext='.wav',
					max_samples=None, random_state=None):
	"""Loads all available vowel samples associated with `str_classes` and `gender`

	If `max_samples` is specified, then the *total* number of samples returned is limited to this number. In the case of
	both genders being sampled, the vowels are equally distributed among men and women.
	"""

	assert gender in ['women', 'men', 'both'], "gender must be either 'women', 'men', or 'both'"

	x_w = []
	y_w = []
	F_w = []
	x_m = []
	y_m = []
	F_m = []
	for i, str_class in enumerate(str_classes):
		y = np.eye(len(str_classes))[i]

		# Women
		files = os.path.join(dir, 'w*' + str_class + ext)
		for file in sorted(glob.glob(files)):
			x = load_vowel(file, sr=sr, normalize=normalize)
			F_w.append(file)
			x_w.append(x)
			y_w.append(y)

		# Men
		files = os.path.join(dir, 'm*' + str_class + ext)
		for file in sorted(glob.glob(files)):
			x = load_vowel(file, sr=sr, normalize=normalize)
			F_m.append(file)
			x_m.append(x)
			y_m.append(y)

	if max_samples is not None:
		# Limit the number of returned samples
		if gender == 'both':
			num_samples_men = math.floor(max_samples / 2)
			num_samples_women = math.ceil(max_samples / 2)
		else:
			num_samples_men = num_samples_women = max_samples

		# Here, we "abuse" train_test_split() to get a stratified subset by only
		# utilizing the returned training set (testing set is dropped)
		x_m, _, y_m, _, F_m, _ = train_test_split(x_m, y_m, F_m, train_size=num_samples_men, test_size=len(str_classes),
												  stratify=y_m, shuffle=True, random_state=random_state)
		x_w, _, y_w, _, F_w, _ = train_test_split(x_w, y_w, F_w, train_size=num_samples_women,
												  test_size=len(str_classes), stratify=y_w, shuffle=True,
												  random_state=random_state)

	# Pack the samples into a list of tensors
	if gender == 'both':
		X = [torch.tensor(x, dtype=torch.get_default_dtype()) for x in x_m + x_w]
		Y = [torch.tensor(y, dtype=torch.get_default_dtype()) for y in y_m + y_w]
		F = F_m + F_w
	elif gender == 'women':
		X = [torch.tensor(x, dtype=torch.get_default_dtype()) for x in x_w]
		Y = [torch.tensor(y, dtype=torch.get_default_dtype()) for y in y_w]
		F = F_w
	else:
		X = [torch.tensor(x, dtype=torch.get_default_dtype()) for x in x_m]
		Y = [torch.tensor(y, dtype=torch.get_default_dtype()) for y in y_m]
		F = F_m

	print("Dataset: %d vowel samples" % len(X))

	return X, Y, F


def select_vowel_sample(X, Y, F, y_class, ind=None):
	"""Select a specific vowel sample from the set
	"""
	labels_ints = [y.argmax().item() for y in Y]
	inds_this_clss = [i for i in range(len(labels_ints)) if labels_ints[i] == y_class]

	if ind is None:
		ind = int(random.random() * len(inds_this_clss))

	print('Selected sample %d, corresponding to file %s' % (ind, F[inds_this_clss[ind]]))

	return X[inds_this_clss[ind]].unsqueeze(0), Y[inds_this_clss[ind]].unsqueeze(0)
