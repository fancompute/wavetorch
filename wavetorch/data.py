import torch
import librosa
import numpy as np
import os
import sklearn

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

    return torch.tensor(data)


def load_all_vowels(directories_str, sr=None, normalize=True, train_size=3, test_size=3, pad_factor=1.0):
    """
    Use librosa to load all vowels from the collection of directories.
    """
    
    inputs = []
    labels = []
    for i, directory_str in enumerate(directories_str):
        directory = os.fsencode(directory_str)
        label = torch.eye(len(directories_str))[i]

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".wav"):
                input = load_vowel(os.path.join(directory_str, filename), sr=sr, normalize=normalize)
                inputs.append(input)
                labels.append(label)
                continue
            else:
                continue

    x_all = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    x_all = torch.nn.functional.pad(x_all, (1, int(x_all.shape[1] * pad_factor))).numpy()
    y_all = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True).numpy()

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_all, y_all, train_size=train_size, test_size=test_size, stratify=y_all)

    return torch.tensor(x_train), torch.tensor(x_test), torch.tensor(y_train), torch.tensor(y_test)
