import importlib
import time
import librosa.core as lc
from librosa.util import fix_length
from sklearn.utils import shuffle
from os.path import join
import tensorflow as tf
import numpy as np
from librosa.core import phase_vocoder
from librosa.effects import pitch_shift
from datetime import datetime as dt
from tensorboardX import SummaryWriter
from utils import *


class SourceSeparator(object):
    def __init__(self, sess, config):
        """

        :param sess: Tensorflow session
        :param config: args defined in main
        """
        self.sess = sess
        self.num_sources = config.nsources
        self.batch_size = config.batch_size
        self.nf = config.nf
        self.clip_sec = config.clipsec
        self.samplerate = config.samplerate
        self.checkpoint_dir = config.checkpoint_dir
        self.nfft = config.nfft
        self.dropout_keep_prob = config.dropout_keep_prob
        self.dropout = config.dropout
        self.ydim = int(4 * self.clip_sec * self.samplerate / self.nfft)
        self.xdim = int(self.nfft / 2 + 1)
        self.in_shape = [self.batch_size, self.xdim, self.ydim, 2]
        self.res_blocks = config.resblocks
        self.use_mag_loss = config.use_mag_loss
        self.share_decoder = config.share_decoder
        self.use_mse = config.use_mse
        self.masking = config.masking
        self.freq_filt = config.freqfilt
        self.max_bins = config.maxbins
        self.mag_loss_weight = config.mag_loss_weight
        self.comb_loss_weight = config.comb_loss_weight
        self.mse_weight = config.mse_weight

        self.build_model()

    def build_model(self):
        self.inputs_mix = tf.placeholder(tf.float32, shape=(self.batch_size, self.xdim, self.ydim, 2))
        self.inputs_pure = tf.placeholder(tf.float32, shape=(self.num_sources, self.batch_size, self.xdim, self.ydim, 2))

        def sqr(input):
            return tf.pow(input, 2)

        def magnitude(x):
            return sqr(x[:, :, :, 0]) + sqr(x[:, :, :, 1])

        self.pure_stack = tf.concat([self.inputs_pure[i] for i in range(self.num_sources)], axis=3)
        self.E = self.encoder(self.inputs_mix[:, :self.max_bins])
        self.res_out = self.resnet(self.E, self.res_blocks, self.nf * 4, name="generator_res")
        res_channels_per_source = int(self.nf * 4 / self.num_sources)
        assert self.res_out.shape[3] == res_channels_per_source * self.num_sources
        self.output = []
        if self.share_decoder:
            for i in range(self.num_sources):
                if i == 0:
                    self.output.append(self.decoder(self.res_out[:, :, :, :res_channels_per_source]))
                else:
                    self.output.append(self.decoder(
                        self.res_out[:, :, :, i * res_channels_per_source:(i + 1) * res_channels_per_source],
                        reuse=True))
        else:
            for i in range(self.num_sources):
                self.output.append(
                    self.decoder(self.res_out[:, :, :, i * res_channels_per_source:(i + 1) * res_channels_per_source],
                                 name="generator_decoder_%d" % (i)))

        print("input shape: ", self.inputs_mix.shape)
        print("output shape: ", self.output[0].shape)
        print("resnet channels per source: ", res_channels_per_source)

        self.output_magnitudes = []
        self.input_pure_magnitudes = []
        for i in range(self.num_sources):
            self.input_pure_magnitudes.append(magnitude(self.inputs_pure[i]))
            self.output_magnitudes.append(magnitude(self.output[i]))
        self.input_magnitude = magnitude(self.inputs_mix)

        if self.masking:
            self.output_masked = []
            self.masks = []
            self.sum_magnitudes = tf.zeros_like(self.output_magnitudes[0])
            for i in range(self.num_sources):
                self.sum_magnitudes = self.sum_magnitudes + self.output_magnitudes[i]
            for i in range(self.num_sources):
                self.masks.append(tf.divide(tf.clip_by_value(self.output_magnitudes[i], 1e-10, 1e10),
                                            tf.clip_by_value(self.sum_magnitudes, 1e-10, 1e10)))
                mask = tf.stack([self.masks[-1], self.masks[-1]], axis=3)
                tmp = tf.multiply(mask, self.inputs_mix[:, :self.max_bins])
                self.output_masked.append(
                    tf.concat([tmp, tf.zeros(shape=(self.batch_size, self.xdim - self.max_bins, self.ydim, 2))],
                              axis=1))

        else:
            self.output_masked = []
            for i in range(self.num_sources):
                self.output_masked.append(tf.concat(
                    [self.output[i], tf.zeros(shape=(self.batch_size, self.xdim - self.max_bins, self.ydim, 2))],
                    axis=1))

        self.output = self.output_masked
        self.out_stack = tf.concat([self.output[i] for i in range(self.num_sources)], axis=3)

        def rec_loss(x, y):
            if self.use_mse:
                return tf.reduce_mean(tf.pow(x - y, 2))
            else:
                return tf.reduce_mean(tf.abs(x - y))


        self.loss_mse = 0
        self.loss_mag = 0
        self.loss_comb = rec_loss(self.inputs_mix, tf.reduce_sum(self.output, axis=0)) * self.comb_loss_weight


        for i in range(self.num_sources):
            self.loss_mse = self.loss_mse + rec_loss(self.output[i][:, :self.max_bins],
                                                         self.inputs_pure[i, :, :self.max_bins])
            if self.use_mag_loss:
                self.loss_mag = self.loss_mag + tf.reduce_mean(
                    sqr(tf.log(tf.clip_by_value(self.output_magnitudes[i], 1e-10, 1e10)) - tf.log(
                        tf.clip_by_value(self.input_pure_magnitudes[i], 1e-10, 1e10))))
            else:
                self.loss_mag = tf.reduce_mean(tf.ones_like(self.output[0]))
        self.loss = self.loss_mse * self.mse_weight
        if self.use_mag_loss:
            self.loss = self.loss + (self.loss_mag * self.mag_loss_weight)


        self.loss += self.loss_comb * self.comb_loss_weight
        t_vars = tf.trainable_variables()
        enc_vars = [var for var in t_vars if 'encoder' in var.name]
        dec_vars = [var for var in t_vars if 'decoder' in var.name]
        res_vars = [var for var in t_vars if 'generator_res' in var.name]
        print('generator parameters --- E: %d, R: %d, D: %d' % (
            count_params(enc_vars), count_params(res_vars), count_params(dec_vars)))
        assert count_params(t_vars) == count_params(enc_vars) + count_params(res_vars) + count_params(
            dec_vars)
        self.saver = tf.train.Saver()

    def train(self, sess, config):
        t_vars = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer
        self.writer = SummaryWriter(config.log_dir)
        optim = optimizer(config.lr).minimize(self.loss, var_list=t_vars)
        tf.global_variables_initializer().run()
        start_time = time.time()

        def random_crop(sources, width):
            ceil = sources[0].shape[1] - width
            if ceil == 0:
                return sources
            ind = np.random.randint(ceil)
            return [i[:, ind:ind + width] for i in sources]


        if config.nrecordings == -1:
            n_recordings = 100
        else:
            n_recordings = config.nrecordings

        source_names = ["vocals", "drums", "bass", "other"]
        if "Dev" in os.listdir(join(config.data_path,"Sources")):
            wav_path = join(config.data_path,"Sources","Dev")
            wav_path_test = join(config.data_path,"Sources","Test")
        else:
            wav_path = join(config.data_path,"Sources","train")
            wav_path_test = join(config.data_path,"Sources","test")

        train_recordings = [join(wav_path, i) for i in os.listdir(wav_path)[:n_recordings]]
        if not config.nrecordings==-1:
            train_recordings = train_recordings[:config.nrecordings]
        test_recordings = [join(wav_path_test, i) for i in os.listdir(wav_path_test)[:10]]

        test_data_wav = [
            [combine_stereo(sf.read(join(k, source_names[i] + ".wav"))[0]) for i in range(4)] for
            k in
            test_recordings]
        train_data_wav = [
            [combine_stereo(sf.read(join(k, source_names[i] + ".wav"))[0]) for i in range(4)] for k in
            train_recordings]

        stretch_rate_list = [0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2]
        pitch_shift_list = [-2, -1 ,0, 1, 2]

        def get_data(train=True):
            batch_out = []
            interval = int(self.clip_sec * self.samplerate) * 2
            for batch_idx in range(self.batch_size):
                if train:
                    rec_idx = np.random.randint(len(train_data_wav))
                    crop_idx = np.random.randint(len(train_data_wav[rec_idx][0]) - interval)
                    sources = [i[crop_idx:crop_idx + interval] for i in train_data_wav[rec_idx]]
                else:
                    rec_idx = np.random.randint(len(test_data_wav))
                    crop_idx = np.random.randint(len(test_data_wav[rec_idx][0]) - interval)
                    sources = [i[crop_idx:crop_idx + interval] for i in test_data_wav[rec_idx]]

                if config.pitch_aug and train:
                    n_steps = pitch_shift_list[np.random.randint(len(pitch_shift_list))]
                    if not n_steps==0:
                        sources = [pitch_shift(i, self.samplerate, n_steps=n_steps) for i in sources]

                sources = [from_polar(to_stft(i, self.nfft)) for i in sources]
                if config.bpm_aug and train:
                    rate = stretch_rate_list[np.random.randint(len(stretch_rate_list))]
                    if not rate==1.0:
                        for i in range(len(sources)):
                            augmented = phase_vocoder(sources[i][:, :, 0] + 1j * sources[i][:, :, 1], rate=rate)
                            sources[i] = np.array([np.real(augmented), np.imag(augmented)]).transpose(1, 2, 0)
                if config.amp_aug and train:
                    sources = [i * (0.75 + (np.random.random() * 0.5)) for i in sources]
                sources = random_crop(sources, self.ydim)
                batch_out.append(sources)

            batch_out = np.array(batch_out).transpose(1, 0, 2, 3, 4)
            if train and true_wp(config.shuffle_sources_aug_prob) == 1.0:
                for source_i in range(self.num_sources):
                    np.random.shuffle(batch_out[source_i])
            return batch_out


        def data_loader(train=True):
            pure_b = get_data(train=train)
            mix_b = np.sum(pure_b, axis=0)
            return mix_b, pure_b


        if self.load(self.checkpoint_dir, checkpoint_num=config.checkpoint):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for iteration in range(self.step,config.max_iterations):

            mix_b, pure_b = data_loader()
            _, loss_mse, loss_mag, loss_comb, loss = sess.run(
                [optim, self.loss_mse, self.loss_mag, self.loss_comb, self.loss],
                feed_dict={self.inputs_mix: mix_b, self.inputs_pure: pure_b})


            if np.mod(self.step, config.loss_eval_every) == 0:
                """
                Could use tensorboard instead. This saves many small files with loss values. 
                See experiments notebook, the parse function replaces these files with a dataframe
                """
                mix_b,pure_b = data_loader(train=True)
                mix_b_test,pure_b_test = data_loader(train=False)

                out, mse_loss, mag_loss, comb_loss, loss = self.sess.run(
                    [self.output, self.loss_mse, self.loss_mag, self.loss_comb, self.loss],
                    feed_dict={self.inputs_mix: mix_b, self.inputs_pure: pure_b})
                out_test, mse_loss_test, mag_loss_test, comb_loss_test, loss_test = self.sess.run(
                    [self.output, self.loss_mse, self.loss_mag ,self.loss_comb, self.loss],
                    feed_dict={self.inputs_mix: mix_b_test, self.inputs_pure: pure_b_test})

                self.writer.add_scalar('losses/loss', loss, self.step)
                self.writer.add_scalar('losses/mse', mse_loss * self.mse_weight, self.step)
                self.writer.add_scalar('losses/mag', mag_loss * self.mag_loss_weight, self.step)
                self.writer.add_scalar('losses/comb', comb_loss * self.comb_loss_weight, self.step)
                self.writer.add_scalar('val losses/loss', loss_test, self.step)
                self.writer.add_scalar('val losses/mse', mse_loss_test * self.mse_weight, self.step)
                self.writer.add_scalar('val losses/mag', mag_loss_test * self.mag_loss_weight, self.step)
                self.writer.add_scalar('val losses/comb', comb_loss_test * self.comb_loss_weight, self.step)
                print("Step: [%d] time: %4.4f,  reconstruction: %.4f, reconstruction val: %.4f "
                        % (self.step, time.time() - start_time, mse_loss, mse_loss_test))

            if np.mod(self.step, config.checkpoint_every) == 0:
                self.save(self.checkpoint_dir, self.step)
            self.step += 1

    def encoder(self, image, reuse=False, name="generator_encoder"):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            gf_dim = self.nf
            c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            c1 = dropout(tf.nn.relu(
                batch_norm(conv2d(c0, gf_dim, [self.freq_filt, 7], 1, padding='VALID', name='g_e1_c'), 'g_e1_bn')),
                keep_prob=self.dropout_keep_prob)
            c2 = dropout(
                tf.nn.relu(batch_norm(conv2d(c1, gf_dim * 2, [self.freq_filt, 3], 2, name='g_e2_c'), 'g_e2_bn')),
                keep_prob=self.dropout_keep_prob)
            c3 = dropout(
                tf.nn.relu(batch_norm(conv2d(c2, gf_dim * 4, [self.freq_filt, 3], 2, name='g_e3_c'), 'g_e3_bn')),
                keep_prob=self.dropout_keep_prob)
            return c3

    def resnet(self, input, num_blocks, num_filters, reuse=False, name="generator_resnet"):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            def residule_block(x, dim, ks=3, s=1, name='res', use_dropout=False):
                p = int((ks - 1) / 2)
                y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = batch_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c1'), name + '_bn1')
                y = tf.nn.relu(y)
                if use_dropout:
                    y = dropout(y, keep_prob=self.dropout_keep_prob)
                y = tf.pad(y, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = batch_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c2'), name + '_bn2')
                return y + x

            current = input
            for i in range(num_blocks):
                current = residule_block(current, num_filters, name=name + '_g_r%d' % (i), use_dropout=self.dropout)
            return current

    def decoder(self, input, reuse=False, name="generator_decoder"):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            gf_dim = self.nf
            d1 = deconv2d(input, gf_dim * 2, [self.freq_filt, 3], 2, name='g_d1_dc')
            d1 = tf.nn.relu(batch_norm(d1, 'g_d1_bn'))
            d2 = deconv2d(d1, gf_dim, [self.freq_filt, 3], 2, name='g_d2_dc')
            d2 = tf.nn.relu(batch_norm(d2, 'g_d2_bn'))
            d2 = tf.pad(d2, [[0, 0], [self.freq_filt - 4, self.freq_filt - 4], [5, 4], [0, 0]], "REFLECT")
            pred = conv2d(d2, 2, [self.freq_filt, 10], 1, padding='VALID', name='g_pred_c')
            pred = tf.nn.relu(batch_norm(pred, 'g_pred_bn0'))
            pred = conv2d(pred, 2, 3, 1, padding='SAME', name='g_pred_c2')
            return pred


    def save(self, checkpoint_dir, step):
        model_name = "RA.model"
        checkpoint_dir = checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir, checkpoint_num=-1):
        """Load the model, the checkpoint closest to checkpoint_num is selected, -1 for newest"""
        print(" [*] Reading checkpoints...")
        if len(os.listdir(checkpoint_dir))==1: #because of old version had extra folder
            checkpoint_dir = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint_num != -1:
            all_checkpoints = [i for i in os.listdir(checkpoint_dir) if ".meta" in i]
            nums = [int(i.split(".")[1].replace("model-", "")) for i in all_checkpoints]
            closest_num = nums[np.argmin([(num - checkpoint_num) ** 2 for num in nums])]

        if ckpt and ckpt.model_checkpoint_path:
            with tf.device('/cpu:0'):
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                if checkpoint_num != -1:
                    newest = ckpt_name.split("-")[-1]
                    ckpt_name = ckpt_name.replace(newest, str(closest_num))
                step = int(ckpt_name.split("-")[-1].split(".")[0])
                self.step = step
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            self.step = 1
            return False
