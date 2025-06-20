"""
Temporary script to augment audio files in given directory.
Later it will be integrated into the main pipeline.

"""

import os
import numpy as np
import librosa
import soundfile as sf

sampling_rate = 16000  # This is the sampling rate from your data


# Function to add noise
def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_level * noise
    return augmented_audio


# Function to change pitch
def change_pitch(audio, sampling_rate, n_steps=5):
    return librosa.effects.pitch_shift(audio, sr=sampling_rate, n_steps=n_steps)


# Function to change speed
def change_speed(audio, speed_factor=1.5):
    return librosa.effects.time_stretch(audio, rate=speed_factor)


def augment_audio_files(directory):
    """
    Augment audio files in the given directory by adding noise, changing pitch, and changing speed.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            audio, sr = librosa.load(file_path, sr=sampling_rate)
            # Add noise
            noisy_audio = add_noise(audio)
            noisy_file_path = os.path.join(directory, f"{filename}_nosy.wav")
            sf.write(noisy_file_path, noisy_audio, sr)

            # Change pitch
            pitched_audio = change_pitch(audio, sr)
            pitched_file_path = os.path.join(directory, f"{filename}_pitch.wav")
            sf.write(pitched_file_path, pitched_audio, sr)

            # Change speed
            sped_audio = change_speed(audio)
            sped_file_path = os.path.join(directory, f"{filename}_speed.wav")
            sf.write(sped_file_path, sped_audio, sr)

    print("Audio augmentation completed.")


if __name__ == "__main__":
    # Specify the directory containing audio files
    audio_directory = "data/new_dataset/processed_audio/british"
    augment_audio_files(audio_directory)
