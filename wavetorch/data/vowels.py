import torch
import librosa
import numpy as np
import os
import sklearn
import glob

from sklearn.model_selection import train_test_split

def normalize_vowel(wav_data):
    total_power = np.square(wav_data).sum()

    return wav_data / np.sqrt(total_power)


def load_vowel(file, sr=None, normalize=True):
    """
    Use librosa to to load a single vowel with a specified sample rate
    """

    data, rate = librosa.load(file, sr=sr)

    if normalize:
        data = normalize_vowel(data)

    return data


def load_all_vowels(str_classes, gender='both', sr=None, normalize=True, dir='data/vowels/', ext='.wav', max_samples=None):
    """
    Loads all available vowel samples associated with `str_classes` and `gender`

    If `max_samples` is specified, then the *total* number of samples returned is limited to this number. In the case of 
    both genders being sampled, the vowels are equally distributed among men and women.
    """
   
    assert gender in ['women', 'men', 'both'], "gender must be either 'women', 'men', or 'both'"

    x_w = []
    y_w = []
    x_m = []
    y_m = []
    for i, str_class in enumerate(str_classes):
        y = np.eye(len(str_classes))[i]

        # Women
        files = os.path.join(dir, 'w*' + str_class + ext)
        for file in glob.glob(files):
            x = load_vowel(file, sr=sr, normalize=normalize)
            x_w.append(x)
            y_w.append(y)

        # Men
        files = os.path.join(dir, 'm*' + str_class + ext)
        for file in glob.glob(files):
            x = load_vowel(file, sr=sr, normalize=normalize)
            x_m.append(x)
            y_m.append(y)

    if max_samples is not None:
        # Limit the number of returned samples select
        if gender == 'both':
            num_samples = int(max_samples/2)
        else:
            num_samples = max_samples

        # Here, we "abuse" train_test_split() to get a stratified subset by only
        # utilizing the returned training set (testing set is dropped)
        x_m, _, y_m, _ = train_test_split(x_m, y_m, train_size=num_samples, test_size=len(str_classes), stratify=y_m, shuffle=True)
        x_w, _, y_w, _ = train_test_split(x_w, y_w, train_size=num_samples, test_size=len(str_classes), stratify=y_w, shuffle=True)

    # Pack the samples into a list of tensors
    if gender == 'both':
        X = [torch.tensor(x) for x in x_m + x_w]
        Y = [torch.tensor(y) for y in y_m + y_w]
    elif gender == 'women':
        X = [torch.tensor(x) for x in x_w]
        Y = [torch.tensor(y) for y in y_w]
    else:
        X = [torch.tensor(x) for x in x_m]
        Y = [torch.tensor(y) for y in y_m]

    print("dataset: selected %d vowel samples" % len(X))

    return X, Y
