import os
from os.path import join
import tensorflow as tf
from model import SourceSeparator
from datetime import datetime as dt
import pickle
import shutil
import argparse



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# PATHS
parser.add_argument("--checkpoint-dir", type=str, default="checkpoint/", help="Directory name to save the checkpoints")
parser.add_argument("--load-dir", type=str, default="", help="path to checkpoint directory to load model")
parser.add_argument("--log-dir", type=str, default="logs/", help="path to save tensorboard logs")
parser.add_argument("--data-path", type=str, default="data/", help="path dataset")
parser.add_argument("--group", type=str, default="experiments", help="assign the run to a group (good for keeping track of many)")
parser.add_argument("--checkpoint",type=int, default=-1, help="Which checkpoint to load if loading, -1 for newest")


# LOSS
parser.add_argument("--use-mag-loss", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, help="use magnitude loss term")
parser.add_argument("--use-mse", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True, help="if True then use mse, else mae")
parser.add_argument("--mag-loss-weight", type=float, default=0.5, help="weight of magnitude in loss")
parser.add_argument("--comb-loss-weight", type=float, default=0.5, help="The weight of combination loss")
parser.add_argument("--mse-weight", type=float, default=1.0, help="weight of mse in loss")

# MODEL
parser.add_argument("--nsources", type=int, default=4, help="number of sources to separate")
parser.add_argument("--dropout", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, help="Set true to use dropout in model, and tune keep_prob")
parser.add_argument("--dropout-keep-prob", type=float, default=1.0, help="dropout keep prob")
parser.add_argument("--nf", type=int, default=64, help="scales the size of the model")
parser.add_argument("--samplerate",type=int, default=44100, help="sample rate of the input audio")
parser.add_argument("--clipsec", type=float, default=0.815, help="length of audio clips for training in seconds")
parser.add_argument("--nfft", type=int, default=1024, help="STFT parameter")
parser.add_argument("--maxbins", type=int, default=513, help="The maximum number of frequency bands to use (lower than (nfft/2+1) excludes the highest frequncies)")
parser.add_argument("--resblocks", type=int, default=6, help="number of residual blocks")
parser.add_argument("--share-decoder",type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True, help="if True, then the sources all share the decoder")
parser.add_argument("--masking", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, help="if True, then the model will be trained with a mask")
parser.add_argument("--freqfilt", type=int, default=12, help="The frequency axis length of conv filters")


# DATA
parser.add_argument("--shuffle-sources-aug-prob", type=float, default=0.0, help="for data augmentation, shuffle instruments within batch with this probability")
parser.add_argument("--bpm-aug", type=lambda x: (str(x).lower() in ['true','1', 'yes']),default=False, help="use time stretching augmentation")
parser.add_argument("--pitch-aug", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, help="use pitch shifting augmentation")
parser.add_argument("--amp-aug", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, help="use amplitude scaling augmentation")
parser.add_argument("--nrecordings", type=int, default=-1, help="number of training recordings to use, -1 for all")

# TRAINING
parser.add_argument("--max-iterations", type=int, default=int(1e6), help="max iterations to train")
parser.add_argument("--checkpoint-every", type=int, default=1000, help="How often to save checkpoint")
parser.add_argument("--loss-eval-every", type=int, default=10, help="How often to save loss to tensorboard")
parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate for optimizer")
parser.add_argument("--batch-size", type=int, default=8, help="The size of batch images")

args = parser.parse_args()


def main():
    date_str = str(dt.now())[:-7].replace(":", "-").replace(" ", "-")
    group = args.group
    if group != "":
        args.checkpoint_dir = join(args.checkpoint_dir, group)
        args.log_dir = join(args.log_dir, group)
    dataset_name = os.path.split(args.data_path)[-1]
    if args.load_dir != "":
        load_path = join(args.log_dir, dataset_name, args.load_dir)
        if os.path.isdir(load_path):
            date_str = args.load_dir
        else:
            raise Exception("[!] load directory not found: %s" % (load_path))

    checkpoint_path = join(args.checkpoint_dir, dataset_name, date_str)
    log_path = join(args.log_dir, dataset_name, date_str)
    args.checkpoint_dir = checkpoint_path
    args.log_dir = log_path
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    run_config.allow_soft_placement = True

    argspath = os.path.join(args.checkpoint_dir, 'args.pkl')
    with open(argspath, 'wb') as f:
        pickle.dump(args, f)

    with tf.Session(config=run_config) as sess:
        model = SourceSeparator(
            sess, args)
        model.train(sess, args)


if __name__ == '__main__':
    main()
