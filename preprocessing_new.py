import os
import time
import joblib
import librosa
import numpy as np


def features_creator(path, save_dir) -> str:
    """
    This function creates the dataset and saves both data and labels in
    two files, X.joblib and y.joblib in the joblib_features folder.
    With this method, you can persist your features and train quickly
    new machine learning models instead of reloading the features
    every time with this pipeline.
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

    # Creating X and y: zip makes a list of all the first elements, and a list of all the second elements.
    X, y = zip(*feature_list)

    # Array conversion
    X, y = np.asarray(X), np.asarray(y)

    # Array shape check
    print(X.shape, y.shape)

    # Preparing features dump
    X_name, y_name = "X_mfcc.joblib", "y.joblib"

    joblib.dump(X_mfcc, os.path.join(save_dir, X_name))
    joblib.dump(y, os.path.join(save_dir, y_name))

    return "Completed"


if __name__ == "__main__":
    print("Routine started")
    FEATURES = features_creator(path="data/", save_dir="features/")
    print("Routine completed.")
