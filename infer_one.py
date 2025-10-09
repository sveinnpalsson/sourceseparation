# infer_one.py
import os, argparse, pickle, json, time, tempfile, re
import numpy as np, soundfile as sf
import librosa.core as lc
import boto3
from urllib.parse import urlparse
from os.path import expandvars

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import (combine_stereo, to_stft, from_polar, segment_image, to_batches, merge, AttrDict)
from model import SourceSeparator

import subprocess

# ------- S3 helpers -------
S3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
JOB_BUCKET = os.getenv("JOB_BUCKET", "")
JOB_PREFIX = os.getenv("JOB_PREFIX", "")

OUT_FORMAT   = os.getenv("OUT_FORMAT", "mp3").lower()  # "mp3" or "wav"
MP3_BITRATE  = os.getenv("MP3_BITRATE", "192k")        # e.g. "192k" or use VBR: "-q:a 2"

def to_mp3(in_wav: str, out_mp3: str):
    # CBR example:
    cmd = ["ffmpeg", "-y", "-i", in_wav, "-codec:a", "libmp3lame", "-b:a", MP3_BITRATE, out_mp3]
    # Or VBR quality (smaller files, high quality): cmd = ["ffmpeg","-y","-i",in_wav,"-codec:a","libmp3lame","-q:a","2",out_mp3]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def expand_envvars(s: str) -> str:
    return expandvars(s or "")

def parse_s3_url(url: str):
    m = re.match(r"^s3://([^/]+)/(.+)$", (url or "").strip())
    if not m:
        raise ValueError(f"Bad S3 URL: {url}")
    return m.group(1), m.group(2)

def _job_base():
    """Always return a non-empty prefix (derive from INPUT_KEY if needed)."""
    pref = (os.getenv("JOB_PREFIX", "") or "").strip("/")
    if pref:
        return pref
    ik = (os.getenv("INPUT_KEY", "") or "").strip("/")
    # derive folder from input key like jobs/.../<uuid>/input.mp3
    return os.path.dirname(ik) if "/" in ik else "jobs/unknown"


def resolve_input_path(in_arg: str) -> str:
    """
    Accept local path OR s3://bucket/key. Expands ${ENV} and $ENV.
    If S3, downloads to /tmp and returns the local path.
    """
    val = os.path.expandvars(in_arg or "")
    if "${" in val or (val.count("$") and val == in_arg):
        # Clear error if expansion didn’t happen (missing envs in container)
        raise RuntimeError(f"Unexpanded env in --in: {in_arg} (JOB_BUCKET/INPUT_KEY not set?)")
    if val.lower().startswith("s3://"):
        b, k = parse_s3_url(val)
        suffix = os.path.splitext(k)[1] or ".wav"
        fd, tmp_path = tempfile.mkstemp(prefix="in_", suffix=suffix, dir="/tmp")
        os.close(fd)
        S3.download_file(b, k, tmp_path)
        return tmp_path
    return val

def is_s3_url(p: str) -> bool:
    return isinstance(p, str) and p.startswith("s3://")


def s3_download(s3url: str, local_path: str):
    print(s3url)
    print(local_path)
    print("------")
    b, k = parse_s3_url(s3url)
    S3.download_file(b, k, local_path)

def s3_upload(local_path: str, s3key: str, content_type: str = "audio/wav"):
    if not JOB_BUCKET: return
    S3.put_object(
        Bucket=JOB_BUCKET,
        Key=s3key,
        Body=open(local_path, "rb"),
        ContentType=content_type,
        CacheControl="no-store"
    )

def put_progress(stage, pct, msg=""):
    body = json.dumps({"stage": stage, "pct": int(max(0, min(100, pct))), "msg": msg, "ts": int(time.time())})
    if JOB_BUCKET and JOB_PREFIX:
        S3.put_object(
            Bucket=JOB_BUCKET, Key=f"{JOB_PREFIX}/progress.json",
            Body=body.encode("utf-8"), ContentType="application/json", CacheControl="no-store"
        )
    else:
        # Fallback to console in local runs
        print(body)

# ------- Main -------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Path inside image to checkpoint dir")
    ap.add_argument("--in", dest="in_path", required=True, help="Local path or s3://bucket/key")
    ap.add_argument("--out", dest="out_dir", required=False, default="/tmp/out", help="Local output dir (debugging)")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    put_progress("booting", 3, "Starting container…")

    # ----- Load model args -----
    with open(os.path.join(args.ckpt_dir, "args.pkl"), "rb") as f:
        model_args = AttrDict(pickle.load(f))

    # Allow batch size override at inference
    model_args.batch_size = int(args.bs)

    tf.reset_default_graph()
    put_progress("booting", 7, "Loading TensorFlow graph…")

    with tf.Session() as sess:
        model = SourceSeparator(sess, model_args)
        ckpt_dir = model_args.checkpoint_dir if os.path.isdir(model_args.checkpoint_dir) else args.ckpt_dir
        if not model.load(ckpt_dir):
            raise RuntimeError("Failed to load checkpoint")
        put_progress("booting", 12, "Model loaded")
        print("ENV JOB_BUCKET =", os.environ.get("JOB_BUCKET"))
        print("ENV INPUT_KEY  =", os.environ.get("INPUT_KEY"))
        print("args.in_path   =", args.in_path)

        print(0, args.in_path)
        in_local = resolve_input_path(args.in_path)   # <- always a local file now
        print(1, in_local)
        out_dir  = os.path.expandvars(args.out_dir)
        print(2, out_dir)
        data, fs = sf.read(in_local)


        model_fs = int(model_args.samplerate)
        if fs != model_fs:
            from resampy import resample
            data = resample(data, fs, model_fs, axis=0)
            fs = model_fs

        # Stereo / mono handling
        if args.mono or data.ndim == 1:
            data = np.expand_dims(combine_stereo(data), axis=1)  # (N,1)
            n_channels = 1
        else:
            n_channels = 2

        source_names = ["vocals", "drums", "bass", "other"][:int(model.num_sources)]
        nfft = int(model_args.nfft)
        n = len(data)

        # ----- Run inference -----
        total_batches_all_channels = 0
        batches_per_channel = []
        # Precompute how many batches to report progress
        for ch in range(n_channels):
            audio = np.pad(data[:, ch] if n_channels == 2 else data[:, 0], (0, nfft // 2), mode="constant")
            sample_mix = from_polar(to_stft(audio, nfft))
            segs = segment_image(sample_mix, width=int(model.ydim), overlap=float(args.overlap))
            batches, num_segments = to_batches(segs, int(model.batch_size))
            bcount = len(batches)
            batches_per_channel.append((bcount, num_segments, sample_mix.shape[1]))
            total_batches_all_channels += bcount

        done_batches = 0
        channels_wav = []
        put_progress("running", 15, "Beginning separation…")

        for ch in range(n_channels):
            bcount, num_segments, T = batches_per_channel[ch]
            # Rebuild features for this channel (to save memory)
            audio = np.pad(data[:, ch] if n_channels == 2 else data[:, 0], (0, nfft // 2), mode="constant")
            mix = from_polar(to_stft(audio, nfft))
            segs = segment_image(mix, width=int(model.ydim), overlap=float(args.overlap))
            batches, _ = to_batches(segs, int(model.batch_size))

            sources_out = [[] for _ in range(int(model.num_sources))]
            for bi, batch in enumerate(batches):
                outs = sess.run(model.output, feed_dict={model.inputs_mix: batch})
                for s in range(int(model.num_sources)):
                    sources_out[s].extend(outs[s])

                done_batches += 1
                if (done_batches % 4) == 0:
                    pct = 15 + round(80 * done_batches / max(1, total_batches_all_channels))
                    put_progress("running", pct, f"Processing {done_batches}/{total_batches_all_channels} chunks")

            # Merge & trim
            sources_out = [merge(s[:num_segments], overlap=float(args.overlap)) for s in sources_out]
            sources_out = [s[:, :T] for s in sources_out]
            chans = [lc.istft(s[:, :, 0] + 1j * s[:, :, 1], length=n) for s in sources_out]
            channels_wav.append(chans)

        # ----- Write outputs locally (optional) -----
        channels_wav = np.array(channels_wav).transpose(1, 2, 0)  # (sources, samples, channels)
        for j, name in enumerate(source_names):
            local_out = os.path.join(out_dir, f"{name}.wav")
            sf.write(local_out, channels_wav[j], fs)

        vocals_removed = channels_wav[1:].sum(axis=0)
        vocals_removed_local = os.path.join(out_dir, "vocals_removed.wav")
        sf.write(vocals_removed_local, vocals_removed, fs)


        put_progress("packaging", 97, "Uploading results…")

        # Convert to mp3 if requested
        use_mp3 = (OUT_FORMAT == "mp3")
        def maybe_mp3(path_wav):
            if not use_mp3: return path_wav, "audio/wav", os.path.basename(path_wav)
            out_mp3 = path_wav.replace(".wav", ".mp3")
            to_mp3(path_wav, out_mp3)
            return out_mp3, "audio/mpeg", os.path.basename(out_mp3)


        base = _job_base()
        outputs = []

        # stems
        for name in source_names:
            wav_path = os.path.join(out_dir, f"{name}.wav")
            up_path, ctype, fname = maybe_mp3(wav_path)
            key = f"{base}/stems/{fname}"
            s3_upload(up_path, key, content_type=ctype)
            outputs.append({"name": fname, "url": f"s3://{JOB_BUCKET}/{key}"})

        # vocals_removed
        up_vr, c_vr, f_vr = maybe_mp3(vocals_removed_local)
        key_vr = f"{base}/stems/{f_vr}"
        s3_upload(up_vr, key_vr, content_type=c_vr)
        outputs.append({"name": f_vr, "url": f"s3://{JOB_BUCKET}/{key_vr}"})

        manifest = {
            "duration_sec": float(n / fs),
            "num_sources": int(model.num_sources),
            "sample_rate": int(fs),
            "format": "mp3" if use_mp3 else "wav",
            "outputs": outputs,
        }
        if JOB_BUCKET:
            S3.put_object(
                Bucket=JOB_BUCKET,
                Key=f"{base}/manifest.json",
                Body=json.dumps(manifest).encode("utf-8"),
                ContentType="application/json",
                CacheControl="no-store",
            )

        put_progress("complete", 100, "Done")
        

if __name__ == "__main__":
    main()
