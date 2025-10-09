import os
from os.path import join
import numpy as np
import soundfile as sf
from scipy.signal import resample
import librosa.core as lc
from librosa.util import fix_length
import librosa

# --- TF1 on TF2 ---
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def count_params(varlist):
    total_parameters = 0
    for variable in varlist:
        shape = variable.get_shape().as_list()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= (dim or 1)
        total_parameters += variable_parameters
    return total_parameters


def dropout(x, keep_prob=0.5):
    if keep_prob == 1.0:
        return x
    else:
        # TF1-style dropout (keep_prob); valid under compat.v1
        return tf.nn.dropout(x, keep_prob=keep_prob)


def batch_norm(x, name="batch_norm", decay=0.9, epsilon=1e-5, scale=True):
    """
    Inference-only BN that matches legacy variable names:
    {scope}/beta, {scope}/gamma, {scope}/moving_mean, {scope}/moving_variance
    Uses the moving stats (no updates), which is what you want at test time.
    """
    with tf.variable_scope(name):
        ch = x.get_shape().as_list()[-1]
        beta = tf.get_variable("beta", [ch], initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", [ch], initializer=tf.ones_initializer()) if scale else None
        moving_mean = tf.get_variable("moving_mean", [ch], initializer=tf.zeros_initializer(), trainable=False)
        moving_var  = tf.get_variable("moving_variance", [ch], initializer=tf.ones_initializer(),  trainable=False)
        # Inference: normalize with moving stats. (No update_ops in inference.)
        return tf.nn.batch_normalization(x, moving_mean, moving_var, beta, gamma, epsilon)


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    """TF1 conv that mimics slim naming: {scope}/Conv/weights (no bias, no activation)."""
    with tf.variable_scope(name):
        # allow int or (kh, kw) / (sh, sw)
        kh, kw = (ks, ks) if isinstance(ks, int) else ks
        sh, sw = (s, s)   if isinstance(s, int)  else s
        in_ch  = input_.get_shape().as_list()[-1]
        W = tf.get_variable(
            "Conv/weights",
            shape=[kh, kw, in_ch, output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev)
        )
        return tf.nn.conv2d(input_, W, strides=[1, sh, sw, 1], padding=padding)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    """
    Mimic slim naming: {scope}/Conv2d_transpose/weights
    Transpose kernel shape = [kh, kw, out_ch, in_ch]
    Output shape is computed from input and stride (SAME padding keeps spatial*stride).
    """
    with tf.variable_scope(name):
        kh, kw = (ks if isinstance(ks, (list, tuple)) else (ks, ks))
        in_ch  = input_.get_shape().as_list()[-1]
        W = tf.get_variable(
            "Conv2d_transpose/weights",
            shape=[kh, kw, output_dim, in_ch],
            initializer=tf.truncated_normal_initializer(stddev=stddev)
        )
        in_shape = tf.shape(input_)
        out_h = in_shape[1] * s
        out_w = in_shape[2] * s
        out_shape = tf.stack([in_shape[0], out_h, out_w, output_dim])
        return tf.nn.conv2d_transpose(input_, W, output_shape=out_shape, strides=[1, s, s, 1], padding='SAME')



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
    batches = [np.zeros(shape=(batch_size,) + tuple(segments[0].shape)) for _ in range(n_batches)]
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
    :return: STFT with magnitude in one channel and phase in the other.
    """
    nfft_padlen = int(len(seq) + nfft / 2)
    stft = lc.stft(fix_length(seq, size=nfft_padlen), n_fft=nfft)
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
    """STFT with magnitude in one channel and phase in the other -> complex STFT"""
    return np.array([image[:, :, 0] * np.cos(image[:, :, 1]),
                     image[:, :, 0] * np.sin(image[:, :, 1])]).transpose(1, 2, 0)


def to_time(image):
    """Complex STFT -> raw audio"""
    return lc.istft(image[:, :, 0] + 1j * image[:, :, 1])


def true_wp(prob):
    return 1.0 if np.random.random() < prob else 0.0
