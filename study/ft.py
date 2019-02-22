import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy
import librosa
import librosa.display

sr_new = 5000
n_fft = 512

A_test, sr = librosa.load('data/vowels/test/a_test.wav')
A_train, sr = librosa.load('data/vowels/a.wav')
A_test_rs  = librosa.resample(A_test, sr, sr_new)
A_train_rs  = librosa.resample(A_train, sr, sr_new)

def plot_spectra(vowel_id, sr_new = 5000, n_fft = 256):

    vowel_test, sr = librosa.load('data/vowels/test/{}_test.wav'.format(vowel_id))
    vowel_train, sr = librosa.load('data/vowels/{}.wav'.format(vowel_id))
    vowel_test_rs  = librosa.resample(vowel_test, sr, sr_new)
    vowel_train_rs  = librosa.resample(vowel_train, sr, sr_new)

    fig=plt.figure()

    plt.subplot(2,2,1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(vowel_test, n_fft=n_fft)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear',
                                x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('%s (original testing)  -  sr=%.2f' % (vowel_id, sr))

    plt.subplot(2,2,3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(vowel_test_rs, n_fft=n_fft)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr_new)
    plt.colorbar(format='%+2.0f dB')
    plt.title('%s (decimated testing)  -  sr=%.2f' % (vowel_id, sr_new))

    plt.subplot(2,2,2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(vowel_train, n_fft=n_fft)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear',
                                x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('%s (original training)  -  sr=%.2f' % (vowel_id, sr))

    plt.subplot(2,2,4)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(vowel_train_rs, n_fft=n_fft)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr_new)
    plt.colorbar(format='%+2.0f dB')
    plt.title('%s (decimated training)  -  sr=%.2f' % (vowel_id, sr_new))

    fig.tight_layout()

    plt.show(block=False)

    return fig

fig = plot_spectra('a', sr_new=sr_new, n_fft=n_fft)
# fig.savefig('./spectra_a.png', dpi=190)

fig = plot_spectra('e', sr_new=sr_new, n_fft=n_fft)
# fig.savefig('./spectra_e.png', dpi=190)

fig = plot_spectra('o', sr_new=sr_new, n_fft=n_fft)
# fig.savefig('./spectra_o.png', dpi=190)


