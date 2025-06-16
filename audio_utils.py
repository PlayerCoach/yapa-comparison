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
        "24000",
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
    # Load audio
    audio, sr = librosa.load(path, sr=None)

    # Apply noise reduction
    reduced_audio = nr.reduce_noise(y=audio, sr=sr)

    # Save denoised audio
    sf.write(path, reduced_audio, sr)


from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def trim_silence(path, silence_thresh=-40, min_silence_len=200):
    audio = AudioSegment.from_wav(path)
    nonsilent_ranges = detect_nonsilent(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )

    if nonsilent_ranges:
        start, end = nonsilent_ranges[0][0], nonsilent_ranges[-1][1]
        trimmed_audio = audio[start:end]
        trimmed_audio.export(path, format="wav")


from pydub import AudioSegment


def normalize_audio(path):
    # Load the audio file
    audio = AudioSegment.from_wav(path)

    # Normalize the audio (this will scale the volume to make the peak 0 dBFS)
    normalized_audio = audio.apply_gain(-audio.max_dBFS)

    # Create the output path by combining the desired output name with '.wav' extension

    # Export the normalized audio to the output path
    normalized_audio.export(path, format="wav")


def add_padding(path):
    """
    Adds 50ms of silence padding to the start and end of the audio file
     if the audio is shorter than the target duration.
    """
    audio = AudioSegment.from_wav(path)
    current_duration = len(audio) / 1000.0  # duration in seconds

    silence = AudioSegment.silent(duration=200)  # 500ms of silence
    padded_audio = silence + audio + silence
    padded_audio.export(path, format="wav")
