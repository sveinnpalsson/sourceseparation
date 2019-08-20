import os
from os.path import join
import numpy as np
import soundfile as sf
from scipy.signal import resample
import librosa.core as lc
from librosa.util import fix_length
import librosa
import tensorflow as tf
import tensorflow.contrib.slim as slim



def count_params(varlist):
    total_parameters = 0
    for variable in varlist:
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters


def dropout(x, keep_prob=0.5):
    if keep_prob == 1.0:
        return x
    else:
        return tf.nn.dropout(x, keep_prob=keep_prob)


def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                     biases_initializer=None)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def power(x):
    return np.sum(np.square(x)) / len(x)


def segment_image(im, width=8, overlap=0):
    m = int(width * overlap)
    segments = []
    for i in range(0, im.shape[1], width - m):
        if i > im.shape[1] - width:
            segment = np.zeros_like(segments[0])
            res_seg = im[:, i:]
            segment[:, :res_seg.shape[1]] = res_seg
            segments.append(segment)
        else:
            segments.append(im[:, i:i + width])
    return segments


def merge(segments, overlap=0):
    if overlap == 0:
        return np.concatenate(segments, axis=1)
    width = segments[0].shape[1]
    m = int(width * overlap)
    L = ((len(segments) - 1) * (width - m)) + width
    merged = np.zeros([segments[0].shape[0], L, 2])
    factors = np.zeros(L)
    for i in range(len(segments)):
        segment = segments[i]
        start = i * (width - m)
        stop = (i * (width - m)) + width
        factors[start:stop] += 1
        merged[:, start:stop] = segment + merged[:, start:stop]
    for i in range(L):
        merged[:, i] /= factors[i]
    return merged


def to_batches(segments, batch_size):
    n_batches = int(np.ceil(len(segments) / batch_size))
    batches = [np.zeros(shape=(batch_size,) + tuple(segments[0].shape)) for i in range(n_batches)]
    for i in range(len(segments)):
        batch_idx = i // batch_size
        idx = i % batch_size
        batches[batch_idx][idx] = segments[i]
    return np.array(batches), len(segments)


def downsample(x, down_factor):
    n = x.shape[0]
    y = np.floor(np.log2(n))
    nextpow2 = int(np.power(2, y + 1))
    x = np.concatenate((np.zeros((nextpow2 - n), dtype=x.dtype), x))
    x = resample(x, len(x) // down_factor)
    return x[(nextpow2 - n) // down_factor:]


def combine_stereo(data):
    if len(data.shape) > 1:
        return 0.5 * data[:, 0] + 0.5 * data[:, 1]
    else:
        return data


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def to_stft(seq, nfft):
    """
	:param seq:  Raw audio
	:param nfft: parameter of STFT
	:return: STFT of the input seq, broken down into magnitude in one channel and phase in the other.
	"""
    nfft_padlen = int(len(seq) + nfft / 2)
    stft = lc.stft(fix_length(seq, nfft_padlen), n_fft=nfft)
    return np.array([np.abs(stft), np.angle(stft)]).transpose(1, 2, 0)


def save_audio_sample(samples, path, samplerate):
    sample = np.concatenate(samples)
    sf.write(path, sample, samplerate=samplerate)


def remove_silent_sources(pure_in, pure_out):
    silent_in = [True if np.concatenate(i, axis=1).sum() == 0.0 else False for i in pure_in]
    pure_in = [pure_in[i] for i in range(len(pure_in)) if not (silent_in[i])]
    pure_out = [pure_out[i] for i in range(len(pure_out)) if not (silent_in[i])]
    return pure_in, pure_out, silent_in


def random_crop_sources(sources, width):
    ceil = sources[0].shape[1] - width
    ind = np.random.randint(ceil)
    return [i[:, ind:ind + width] for i in sources]


def from_polar(image):
    """
    :param image: STFT with magnitude in one channel and phase in the other.
    :return: The STFT in its original form.
    """
    return np.array([image[:, :, 0] * np.cos(image[:, :, 1]), image[:, :, 0] * np.sin(image[:, :, 1])]).transpose(1, 2,
                                                                                                                  0)

def to_time(image):
    """
    :param image: STFT with magnitude in one channel and phase in the other.
    :return: Raw audio
    """
    return lc.istft(image[:, :, 0] + 1j * image[:, :, 1])


def true_wp(prob):
    if np.random.random() < prob:
        return 1.0
    else:
        return 0.0


def main():
    return

if __name__ == '__main__':
    main()
