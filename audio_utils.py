"""
File containing utility functions for audio processing tasks.
This file should be the same as in YAPA main repository, so the model is not dependent on some audio processing differences.
"""

import subprocess


def convert_to_wav(path, output_path):
    cmd = [
        "ffmpeg",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        output_path,
        "-y",
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


import noisereduce as nr
import librosa
import soundfile as sf


def denoise_wav(path):
    audio, sr = librosa.load(path, sr=None)
    reduced_audio = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.4)
    sf.write(path, reduced_audio, sr)


from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def trim_silence(path, silence_thresh=-30, min_silence_len=300):
    audio = AudioSegment.from_wav(path)
    ranges = detect_nonsilent(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )
    if ranges:
        start, end = ranges[0][0], ranges[-1][1]
        trimmed = audio[start:end]
        trimmed.export(path, format="wav")


from pydub import AudioSegment


def normalize_audio(path, target_dBFS=-20.0):
    audio = AudioSegment.from_wav(path)
    change = target_dBFS - audio.dBFS
    normalized = audio.apply_gain(change)
    normalized.export(path, format="wav")


def add_padding(path):
    audio = AudioSegment.from_wav(path)
    if len(audio) < 1500:  # only pad very short clips
        silence = AudioSegment.silent(duration=100)
        padded = silence + audio + silence
        padded.export(path, format="wav")
