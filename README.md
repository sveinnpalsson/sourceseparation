# Audio source separation

This repository contains code for my master's project on audio source separation.
The main focus is on separation of musical instruments and vocals. The code
should be easy to use. The master branch contains the main code for the project
while the other branches are for the many experiments and ideas we had during 
the project. These pther branches contain code that is not guaranteed to work but
if interested you can check it out.

To train a source separation model, use main.py. It defines many parameters to
control the model architecture and other design choices. The default parameters
will be the ones of the baseline model discussed in the thesis.

The model and training process are defined in model.py
ops.py defines some tensorflow operations and utils.py contains most other
functions.

Each model you train will be saved at checkpoint/model_name/dataset/date_str/
and the samples + log files saved during training are stored at samples/model_name/dataset/date_str.

For example, the pre-trained model included in this repository is saved at
RA/MUSDB18/2018-08-09-16-53-46

## Running pre-trained model

To run the pre-trained model on music contained in the  music_to_test directory, run:
```
python test_model.py --model_path "RA/MUSDB18/2018-08-09-16-53-46" --data_path "music_to_test"
```

This was tested with the package versions listed in requirements.txt and python 3.6.
In case you experience package version problems, starting a clean python environment
and running
```
pip install -r requirements.txt
```
should set you up with all the right package versions.

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
/path/to/dataset/Sources/song_folders/source_i.wav, where in the Sources folder, all the songs in the dataset 
have a folder and each song folder has .wav files, one for each source, e.g. vocals.wav, drums.wav, bass.wav, other.wav.

When you have the data, you can try training a model with the default parameters like this:
```
python main.py --data_path /path/to/dataset/
```
The default parameters are set to match the baseline model in our paper.  You may
want to play around with the parameters, they are all defined in the beginning of 
main.py. The default parameters correspond to a pretty large model that may not
fit into your GPU, so you might want try training a smaller model with 
```
python main.py --data_path /path/to/dataset/ --ngf 16
```

Also note that the dataset is pretty big, and by default the program loads all of 
the .wav files into memory. If you don't have that much memory, you can deal with
this somehow in the code or set --n_recordings to limit the number of songs
used for training. One easy thing to do is to generate a (pretty huge) dataset
from the .wav files that is stored on disk, many small segments of audio and 
at training time just load some in each iteration. This slows down training because
of all the reading. Best would probably be to use multithreading to load samples
while training.
