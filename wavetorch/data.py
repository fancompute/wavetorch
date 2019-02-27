import torch
import librosa
import numpy as np
import os

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


def load_all_vowels(directories_str, sr=None, normalize=True, num_of_each=1000):
    """
    Use librosa to load all vowels from the collection of directories.

    This function generates corresponding one hot target vectors, treating each
    directory as a distinct class. The total number of samples from each directory
    can be limited by the parameter num_of_each.
    """
    
    inputs = []
    labels = []
    for i, directory_str in enumerate(directories_str):
        directory = os.fsencode(directory_str)
        label = torch.eye(len(directories_str))[i]

        num_this = 0
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".wav"):
                if num_this >= num_of_each:
                    break

                input = load_vowel(os.path.join(directory_str, filename), sr=sr, normalize=normalize)
                inputs.append(input)
                labels.append(label)
                num_this += 1
                continue
            else:
                continue

    return torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True), torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
