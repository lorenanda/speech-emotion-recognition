import os
import time
import joblib
import librosa
import numpy as np


def extract_features(path, save_dir):
"""
Description
"""
    feature_list = []

    start_time = time.time()
    for subdir, dirs, files in os.walk(path):
        for file in files:
            try:
                y_lib, sample_rate = librosa.load(
                    os.path.join(subdir, file), res_type="kaiser_fast"
                )
                mfccs = np.mean(
                    librosa.feature.mfcc(y=y_lib, sr=sample_rate, n_mfcc=40).T, axis=0
                )

                file = int(file[7:8]) - 1
                arr = mfccs, file
                feature_list.append(arr)

            except ValueError as err:
                print(err)
                continue

    print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

    X, y = zip(*feature_list)
    X, y = np.asarray(X), np.asarray(y)
    #print(X.shape, y.shape)

    X_save, y_save = "X.joblib", "y.joblib"
    joblib.dump(X, os.path.join(save_dir, X_save))
    joblib.dump(y, os.path.join(save_dir, y_save))

    return "Completed"


if __name__ == "__main__":
    print("Routine started")
    FEATURES = extract_features(path="data/", save_dir="features/")
    print("Routine completed.")
