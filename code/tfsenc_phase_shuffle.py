import math

import numexpr as ne
# import jax.numpy as jnp
import numpy as np
from scipy.fftpack import fft, ifft


def phase_shuffle(input_signal):
    # Returns a vector of the same size and amplitude spectrum but with shuffled
    # phase information.

    N = input_signal.shape[1]

    if N % 2:
        h = input_signal[-1]
        input_signal = input_signal[:-1]

    F = np.fft.fft(input_signal, axis=-1)
    t = np.zeros(F.shape)
    t[0, 1:math.
      floor(N /
            2)] = np.random.rand(math.floor(N / 2) - 1) * 2 * math.pi - math.pi
    t[:, (math.floor(N / 2) + 1):] = -t[:, math.floor(N / 2) - 1:0:-1]

    output_signal = np.real(np.fft.ifft(F * np.exp(1j * t), axis=-1))

    if N % 2:
        output_signal.append(h)

    return output_signal


def phase_randomize(data):
    # Returns a vector of the same size and amplitude spectrum but with shuffled
    # phase information.
    # Adapted from
    # https://github.com/brainiak/brainiak/blob/master/brainiak/utils/utils.py

    n_examples, n_lags, n_samples = data.shape

    # Get randomized phase shifts
    if n_samples % 2 == 0:
        pos_freq = np.arange(1, n_samples // 2)
        neg_freq = np.arange(n_samples - 1, n_samples // 2, -1)
    else:
        pos_freq = np.arange(1, (n_samples - 1) // 2 + 1)
        neg_freq = np.arange(n_samples - 1, (n_samples - 1) // 2, -1)

    phase_shifts = np.random.rand(n_examples, n_lags,
                                  len(pos_freq)) * 2 * np.math.pi

    # Fast Fourier transform along time dimension of data
    fft_data = np.fft.fft(data, axis=-1)

    # Shift pos and neg frequencies symmetrically, to keep signal real
    fft_data[:, :, pos_freq] *= np.exp(1j * phase_shifts)
    fft_data[:, :, neg_freq] *= np.exp(-1j * phase_shifts)

    # a = fft_data[:, :, pos_freq]
    # b = fft_data[:, :, neg_freq]

    # fft_data[:, :, pos_freq] = ne.evaluate('a * exp(1j * phase_shifts)')
    # fft_data[:, :, neg_freq] = ne.evaluate('b * exp(-1j * phase_shifts)')

    # Inverse FFT to put data back in time domain
    shifted_data = np.real(ifft(fft_data, axis=-1))

    return shifted_data


def phase_randomize_1d(data):
    '''Adapted from
    data is (n_examples, n_samples, n_electrodes)
    https://github.com/brainiak/brainiak/blob/master/brainiak/utils/utils.py'''

    n_samples = data.shape[0]

    # Get randomized phase shifts
    if n_samples % 2 == 0:
        pos_freq = np.arange(1, n_samples // 2)
        neg_freq = np.arange(data.shape[0] - 1, n_samples // 2, -1)
    else:
        pos_freq = np.arange(1, (data.shape[0] - 1) // 2 + 1)
        neg_freq = np.arange(data.shape[0] - 1, (n_samples - 1) // 2, -1)

    phase_shifts = np.random.rand(len(pos_freq), 1) * 2 * np.math.pi

    # Fast Fourier transform along time dimension of data
    fft_data = fft(data, axis=0)

    # Shift pos and neg frequencies symmetrically, to keep signal real
    fft_data[pos_freq, :] *= np.exp(1j * phase_shifts)
    fft_data[neg_freq, :] *= np.exp(-1j * phase_shifts)

    # Inverse FFT to put data back in time domain
    shifted_data = np.real(ifft(fft_data, axis=0))

    # Make it a contiguous array (for numba compatibility])
    shifted_data = np.ascontiguousarray(shifted_data)

    return shifted_data


if __name__ == '__main__':
    iter = 10

    input_vec = np.arange(1, 101).reshape(1, -1)
    print(f'input vec sum: {sum(sum(input_vec))}')
    for i in range(iter):
        np.random.seed(i)
        output_vec0 = phase_shuffle(input_vec)
        print(sum(sum(output_vec0)))

    print("============================")
    input_vec = np.arange(1, 101).reshape(-1, 1)
    print(f'input vec sum: {sum(sum(input_vec))}')
    for i in range(iter):
        np.random.seed(i)
        output_vec1 = phase_randomize_1d(input_vec)
        print(sum(sum(output_vec1)))

    print("============================")
    input_vec = np.arange(1, 101).reshape(1, 1, -1)
    print(f'input vec sum: {sum(sum(sum((input_vec))))}')
    for i in range(iter):
        np.random.seed(i)
        output_vec2 = phase_randomize(input_vec)
        print(sum(sum(sum(output_vec2))))
