import os
import time
import joblib
import librosa
import numpy as np
import pandas as pd


def extract_file_info():
    DATA_PATH = "speech_emotion_recognition/data"

    df = pd.DataFrame(columns=["file", "gender", "emotion", "intensity"])

    for dirname, _, filenames in os.walk(DATA_PATH):
        for filename in filenames:

            emotion = filename[7]
            if emotion == "1":
                emotion = "neutral"
            elif emotion == "2":
                emotion = "calm"
            elif emotion == "3":
                emotion = "happy"
            elif emotion == "4":
                emotion = "sad"
            elif emotion == "5":
                emotion = "angry"
            elif emotion == "6":
                emotion = "fearful"
            elif emotion == "7":
                emotion = "disgusted"
            elif emotion == "8":
                emotion == "surprised"

            intensity = filename[10]
            if intensity == "1":
                emotion_intensity = "normal"
            elif intensity == "2":
                emotion_intensity = "strong"

            gender = filename[-6:-4]
            if int(gender) % 2 == 0:
                gender = "female"
            else:
                gender = "male"

            df = df.append(
                {
                    "file": filename,
                    "gender": gender,
                    "emotion": emotion,
                    "intensity": emotion_intensity,
                },
                ignore_index=True,
            )

    df.to_csv("speech_emotion_recognition/features/df_features_new.csv", index=False)


def extract_features(path, save_dir):
    """
    Description
    """
    feature_list = []

    start_time = time.time()
    for dir, _, files in os.walk(path):
        for file in files:
            try:
                y_lib, sample_rate = librosa.load(
                    os.path.join(dir, file), res_type="kaiser_fast"
                )
                mfccs = np.mean(
                    librosa.feature.mfcc(y=y_lib, sr=sample_rate, n_mfcc=40).T, axis=0
                )

                file = int(file[7:8])  # - 1
                arr = mfccs, file
                feature_list.append(arr)

            except ValueError as error:
                print(error)
                continue

    print("Data loaded in %s seconds." % (time.time() - start_time))

    X, y = zip(*feature_list)
    X, y = np.asarray(X), np.asarray(y)
    print(X.shape, y.shape)

    X_save, y_save = "X.joblib", "y.joblib"
    joblib.dump(X, os.path.join(save_dir, X_save))
    joblib.dump(y, os.path.join(save_dir, y_save))

    return "Preprocessing completed."


if __name__ == "__main__":
    print("Extracting file info...")
    extract_file_info()
    print("Extracting audio features...")
    FEATURES = extract_features(path="data/", save_dir="features/")
    print("Finished extracting audio features.")
