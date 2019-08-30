This repository contains code related to the paper 
**Monaural Music Source Separation using a ResNet Latent Separator Network**

To train a source separation model, use main.py. It defines many parameters to
control the model architecture and training. The default parameters
will be the ones of the baseline model discussed in the paper.

## Output examples
Here are some output examples using the pre-trained model provided in this repository. 
The separation output examples are hosted on https://clyp.it/
1. [Original](https://www.youtube.com/watch?v=dQw4w9WgXcQ) -- [Vocals](https://clyp.it/o52vmanz) -- [Drums](https://clyp.it/kjpjunev) -- [Bass](https://clyp.it/i4cai5md) -- [Rest](https://clyp.it/zb4biluo)
2. [Original](https://www.youtube.com/watch?v=-r679Hhs9Zs) -- [Vocals](https://clyp.it/ojbvd24s) -- [Drums](https://clyp.it/v5oscl5h) -- [Bass](https://clyp.it/ita0d002) -- [Rest](https://clyp.it/kltpn3ib)
3. [Original]() -- [Vocals](https://clyp.it/o52vmanz) -- [Drums]() -- [Bass]() -- [Rest]()
4. [Original]() -- [Vocals](https://clyp.it/o52vmanz) -- [Drums]() -- [Bass]() -- [Rest]()


## Running pre-trained model

To run the pre-trained model on your own favorite music simply run:
```
python test_model.py --model_path "RA/MUSDB18/2018-08-09-16-53-46" --data_path <directory with some music>
```

This was tested with the package versions listed in requirements.txt and python 3.6.
In case you experience package version problems, starting a clean python environment
and running
```
pip install -r requirements.txt
```
should set you up with all the right package versions.
I have tested that this works with tensorflow versions 1.1-1.8 at least.

test_model.py can be used with either .wav or .mp3. In case of .mp3, it will first
be converted to wav format. The output will be avilable in the my_music_transformed
directory.

When "test_model.py" tries to convert .mp3 files, it imports pydub which requires
the following software to be installed:
To install, run
```
apt-get install ffmpeg libavcodec-extra
```
If not using linux, take a look at https://github.com/jiaaro/pydub

A working example tested on AWS with AMI: 64-bit
Deep Learning AMI (Ubuntu) Version 14.0, and g2.2xlarge machine:
```
git clone <this repo>
cd <this repo>
pip install -r requirements.txt
pip install -U tensorflow-gpu #because the deeplearning ami doesn't have libcublas 8
sudo apt-get install ffmpeg libavcodec-extra
python test_model.py --model_path "RA/MUSDB18/2018-08-09-16-53-46" --data_path "music_to_test"
```

Processing a 3 minute track can take from about 30 seconds on a Titan X GPU,
about 4 minutes on my laptop GPU (GTX 950M), and about 25 minutes on a low-end 
GPU (like on AWS g2.2xlarge).
In case of no GPU, uninstall tensorflow-gpu and install tensorflow. Tested with
32 CPUs, takes about 10 minutes.


## Training a model

To train a model you need data. Check out the [MUSDB dataset](https://sigsep.github.io/datasets/musdb.html).
It consists of 150 songs and is used as a benchmark for audio source separation in most recent literature.

When training a model, the code assumes that the data is stored as .wav files with folder structure as
/path-to-dataset/Sources/train/song_folders/source_i.wav, where in the Sources folder, all the songs in the dataset 
have a folder and each song folder has .wav files, one for each source, e.g. vocals.wav, drums.wav, bass.wav, other.wav.

When you have the data, you can try training a model with the default parameters like this:
```
python main.py --data_path /path/to/dataset/
```
The default parameters are set to match the baseline model in the paper.  You may
want to play around with the parameters, they are all defined in the beginning of 
main.py. The default parameters correspond to a pretty large model that may not
fit into your GPU, so you might want try training a smaller model with 
```
python main.py --data_path /path/to/dataset/ --nf 16
```

