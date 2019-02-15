import torch
import librosa
import numpy as np

PATH = 'data/vowels'

def load_vowel(prefix, sr=None, normalize=True):
    data, rate = librosa.load('{}/{}.wav'.format(PATH, prefix), sr=sr)

    if normalize:
        data = normalize_vowel(data)

    return torch.tensor(data), rate

def normalize_vowel(wav_data):
    total_power = np.square(wav_data).sum()

    return wav_data / np.sqrt(total_power)
