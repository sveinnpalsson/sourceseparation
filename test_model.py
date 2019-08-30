import sys,os
from os.path import join
import pickle
import tensorflow as tf
from utils import *
from model import SourceSeparator
from datetime import datetime as dt
import argparse
from resampy import resample
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,help='path to model (group/dataset/date_str)')
parser.add_argument('--data_path', type=str,help='path to directory, containing .wav mixtures')
parser.add_argument('--overlap', type=float, default=0.0, help='The amount of overlap, >= 0 and <1, more overlap--> better results, takes longer')
parser.add_argument('--mono', type=bool, default=False, help='Set true to get mono output or if input is mono')
parser.add_argument('--max_length', type=float, default=1.0, help='The fraction of song length to transform, (set less than 1 if you have memory issues)')
args = parser.parse_args()
model_path = args.model_path

if not os.path.isdir(join("checkpoint",model_path)):
    raise Exception("[!] load directory not found: %s" % (join("checkpoint",model_path)))

with open(join("checkpoint",model_path,"args.pkl"),"rb") as f:
    model_args = pickle.load(f)

print(model_args)
model_args = AttrDict(model_args)

def mp3_to_wav(file):
    from pydub import AudioSegment
    sound = AudioSegment.from_mp3(file)
    sound.export(file.replace(".mp3", ".wav"), format="wav")


data_path = args.data_path
nfft = model_args.nfft
sample_files = os.listdir(data_path)
for file in sample_files:
    if file.endswith(".mp3") and not file.replace(".mp3",".wav") in sample_files:
        print("converting %s to .wav format" % file)
        mp3_to_wav(join(data_path, file))
sample_files = [i for i in os.listdir(data_path) if i.lower().endswith(".wav")]
n_sample = len(sample_files)
checkpoint_path = model_args.checkpoint_dir
if not os.path.isdir(checkpoint_path):
    if os.path.isdir(join("checkpoint",model_path)):
        checkpoint_path = join("checkpoint",model_path)
    else:
        raise Exception("[!] load directory not found: %s" % (checkpoint_path))
overlap = args.overlap

out_path = join("my_music_transformed",model_path)
os.makedirs(out_path,exist_ok=True)
tf.reset_default_graph()
max_len = args.max_length
with tf.Session() as sess:
    model = SourceSeparator(sess, model_args)
    if not model.load(checkpoint_path):
        raise Exception("[!] Did not load model")
    source_names = ["vocals", "drums", "bass", "other"]
    for i in range(n_sample):
        start = dt.now()
        file = sample_files[i]
        save_path = join(out_path,file.replace(".wav",""))
        os.makedirs(save_path,exist_ok=True)
        data = sf.read(join(data_path, sample_files[i]))[0]
        fs = sf.read(join(data_path, sample_files[i]))[1]
        print("Processing song: %s of length: %d seconds"%(sample_files[i],len(data)//fs))
        data = data[:int(len(data)*max_len)]
        model_fs = model_args.samplerate
        if not model_fs==fs:
            print("resampling song from %d Hz to %d Hz"%(fs,model_fs))
            data = resample(data,fs,model_fs,axis=0)
        n = len(data)
        source_names = ["vocals", "drums", "bass", "other"]
        if args.mono==True or len(data.shape)==1:
            data = np.expand_dims(combine_stereo(data),axis=1)
            n_channels = 1
        else:
            n_channels = 2
        channels_wav = []
        for channel in range(n_channels):
            print("channel: %d/%d"%(channel+1,n_channels))
            audio = librosa.util.fix_length(data[:, channel], n + nfft // 2)
            sample_mix = from_polar(to_stft(audio, nfft))
            N = sample_mix.shape[1]
            sample_mix_segments = segment_image(sample_mix, width=model.ydim, overlap=overlap)
            sources_out = [[] for i in range(model.num_sources)]
            sample_batches, num_segments = to_batches(sample_mix_segments, model.batch_size)
            for sample_idx in tqdm(range(len(sample_batches))):
                sample_batch = sample_batches[sample_idx]
                samples_gen = model.sess.run(model.output, feed_dict={model.inputs_mix: sample_batch})
                for source_i in range(model.num_sources):
                    _ = [sources_out[source_i].append(k) for k in samples_gen[source_i]]
            sources_out = [merge(k[:num_segments], overlap=overlap) for k in sources_out]
            sources_out = [k[:, :N] for k in sources_out]
            channels_wav.append([lc.istft(k[:, :, 0] + 1j * k[:, :, 1], length=n) for k in sources_out])
        channels_wav=np.array(channels_wav).transpose(1,2,0)
        for j in range(model.num_sources):
            sf.write(join(save_path,source_names[j]+".wav"),channels_wav[j],model_fs)
        sf.write(join(save_path,sample_files[i]),data,model_fs)
        print("Finished in %.2f seconds"%((dt.now()-start).total_seconds()))
