import torch
import librosa
import numpy as np
import os
import sklearn
import glob

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

def load_selected_vowels(str_classes, gender='both', sr=None, normalize=True, train_size=3, test_size=3, dir='data/vowels/', ext='.wav'):
   
    assert gender in ['women', 'men', 'both'], "gender must be either 'women', 'men', or 'both'"

    if gender is 'both':
        # split evenly between the genders
        train_size_w = int(train_size/2)
        train_size_m = train_size - train_size_w
        test_size_w  = int(test_size/2)
        test_size_m  = train_size - test_size_w
    else:
        train_size_w = train_size_m = train_size
        test_size_w  = test_size_m  = test_size

    inputs_w = []
    labels_w = []
    inputs_m = []
    labels_m = []
    for i, str_class in enumerate(str_classes):
        label = np.eye(len(str_classes))[i]

        # Women
        files = os.path.join(dir, 'w*' + str_class + ext)
        for file in glob.glob(files):
            input = load_vowel(file, sr=sr, normalize=normalize)
            inputs_w.append(input)
            labels_w.append(label)

        # Men
        files = os.path.join(dir, 'm*' + str_class + ext)
        for file in glob.glob(files):
            input = load_vowel(file, sr=sr, normalize=normalize)
            inputs_m.append(input)
            labels_m.append(label)

    x_men_train, x_men_test, y_men_train, y_men_test = sklearn.model_selection.train_test_split(inputs_m, labels_m, train_size=train_size_m, test_size=test_size_m, stratify=labels_m)
    x_women_train, x_women_test, y_women_train, y_women_test = sklearn.model_selection.train_test_split(inputs_w, labels_w, train_size=train_size_w, test_size=test_size_w, stratify=labels_w)
    

    if gender is 'both':
        x_train = [torch.tensor(x) for x in x_women_train + x_men_train]
        x_test  = [torch.tensor(x) for x in x_women_test + x_men_test]
        y_train = [torch.tensor(y) for y in y_women_train + y_men_train]
        y_test  = [torch.tensor(y) for y in y_women_test + y_men_test]
    elif gender is 'women':
        x_train = [torch.tensor(x) for x in x_women_train]
        x_test  = [torch.tensor(x) for x in x_women_test]
        y_train = [torch.tensor(y) for y in y_women_train]
        y_test  = [torch.tensor(y) for y in y_women_test]
    else:
        x_train = [torch.tensor(x) for x in x_men_train]
        x_test  = [torch.tensor(x) for x in x_men_test]
        y_train = [torch.tensor(y) for y in y_men_train]
        y_test  = [torch.tensor(y) for y in y_men_test]

    x_train = torch.nn.utils.rnn.pad_sequence(x_train, batch_first=True)
    x_test  = torch.nn.utils.rnn.pad_sequence(x_test, batch_first=True)
    y_train = torch.nn.utils.rnn.pad_sequence(y_train, batch_first=True)
    y_test  = torch.nn.utils.rnn.pad_sequence(y_test, batch_first=True)

    return x_train, x_test, y_train, y_test
