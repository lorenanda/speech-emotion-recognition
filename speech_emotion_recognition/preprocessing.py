"""
This function extracts the audio features
and information from the voice recordings.
"""

import os
import pandas as pd
import numpy as np
import librosa


def extract_audio_features(PATH):

    filenames = []
    y_list = []
    sr_list = []
    mfcc_list = []
    # fft_list = []
    # magnitude_list = []
    # frequency_list = []
    emotions_list = []
    gender_list = []
    intensity_list = []

    files = librosa.util.find_files(PATH, ext=["wav"])
    files = np.asarray(files)

    for file in files:
        filename = file[-24:]
        filenames.append(filename)

        if filename[6:8] == "01":
            emotions_list.append("neutral")
        elif filename[6:8] == "02":
            emotions_list.append("calm")
        elif filename[6:8] == "03":
            emotions_list.append("happy")
        elif filename[6:8] == "04":
            emotions_list.append("sad")
        elif filename[6:8] == "05":
            emotions_list.append("angry")
        elif filename[6:8] == "06":
            emotions_list.append("fearful")
        elif filename[6:8] == "07":
            emotions_list.append("disgusted")
        elif filename[6:8] == "08":
            emotions_list.append("surprised")

        if int(filename[-6:-4]) % 2 == 0:
            gender_list.append("female")
        elif int(filename[-6:-4]) % 2 != 0:
            gender_list.append("male")

        if filename[9:11] == "01":
            intensity_list.append("normal")
        elif filename[9:11] == "02":
            intensity_list.append("strong")

        y, sample_rate = librosa.load(file, res_type="kaiser_fast")
        y_list.append(y)
        sr_list.append(sample_rate)

        # fft = np.fft.fft(y)
        # fft_list.append(fft)

        # magnitude = np.abs(fft)
        # magnitude_list.append(magnitude)

        # frequency = np.linspace(0, sr, len(magnitude))
        # frequency_list.append(frequency)

        mfcc = np.mean(librosa.feature.mfcc(y=y, n_mfcc=40).T, axis=0)
        mfcc_list.append(np.hstack(mfcc))

        audio_df = pd.DataFrame()
        audio_df["filename"] = filenames
        audio_df["emotion"] = emotions_list
        audio_df["gender"] = gender_list
        audio_df["intensity"] = intensity_list
        audio_df["y"] = y_list
        audio_df["sr"] = sr_list
        audio_df["mfcc"] = mfcc_list
        # audio_df['fft'] = fft_list
        # audio_df['magnitude'] = magnitude_list
        # audio_df['frequency'] = frequency_list

    audio_df.to_csv("audio_features.csv")
    return audio_df


if __name__ == "__main__":
    extract_audio_features("data/")
