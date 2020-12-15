import os
import time
import joblib
import librosa
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


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

    df.to_csv("speech_emotion_recognition/features/df_file_features.csv", index=False)


def extract_features(path, save_dir):
"""This function loops over the audio files,
extracts the MFCC, and saves X and y in joblib format.
"""
    feature_list = []

    start_time = time.time()
    for dir, _, files in os.walk(path):
        for file in files:
            y_lib, sample_rate = librosa.load(
                os.path.join(dir, file), res_type="kaiser_fast"
            )
            mfccs = np.mean(
                librosa.feature.mfcc(y=y_lib, sr=sample_rate, n_mfcc=40).T, axis=0
            )

            file = int(file[7:8]) - 1
            arr = mfccs, file
            feature_list.append(arr)

    print("Data loaded in %s seconds." % (time.time() - start_time))

    X, y = zip(*feature_list)
    X, y = np.asarray(X), np.asarray(y)
    print(X.shape, y.shape)

    X_save, y_save = "X.joblib", "y.joblib"
    joblib.dump(X, os.path.join(save_dir, X_save))
    joblib.dump(y, os.path.join(save_dir, y_save))

    return "Preprocessing completed."




"""This function loops over the audio files,
extracts four audio feature and saves them in a dataframe.
"""

data_path = "speech_emotion_recognition/data/"
actor_folders = os.listdir(data_path) 

emotion_nr = []
emotion = []
gender = []
actor = []
file_path = []

for i in actor_folders:
    filename = os.listdir(data_path + i) 
    for f in filename: 
        part = f.split('.')[0].split('-')
        emotion_nr.append(int(part[2]))
        emotion.append(int(part[2]))
        actor.append(int(part[6]))
        bg = int(part[6])
        if bg%2 == 0:
            bg = "female"
        else:
            bg = "male"
        gender.append(bg)
        file_path.append(data_path + i + '/' + f)


def extract_audio_features():
    """This function loops over the audio files,
    extracts four audio feature and saves them in a dataframe.
    """
    audio_df = pd.DataFrame(emotion)
    audio_df = audio_df.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fearful', 7:'disgusted', 8:'surprised'})
    audio_df = pd.concat([pd.DataFrame(gender),audio_df,pd.DataFrame(actor)],axis=1)
    audio_df.columns = ['gender','emotion','emotion_nr','actor']
    audio_df = pd.concat([audio_df,pd.DataFrame(file_path, columns = ['path'])],axis=1)
    audio_df.tail()
    #audio_df.to_csv('speech_emotion_recognition/features/df_features_new.csv')

    df = pd.DataFrame(columns=['chroma'])

    counter=0

    for index,path in enumerate(audio_df.path):
        X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
            
        chroma = librosa.feature.chroma_stft(y=X, sr=sample_rate)
        chroma = np.mean(chroma, axis = 0)
        df.loc[counter] = [chroma]
        counter=counter+1   

        # spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000) 
        # db_spec = librosa.power_to_db(spectrogram)
        # log_spectrogram = np.mean(db_spec, axis = 0)
        # df.loc[counter] = [log_spectrogram]
        # counter=counter+1  

    # df_chroma = pd.concat([audio_df,pd.DataFrame(df['chroma'].values.tolist())],axis=1)
    # df_chroma = df_combined.fillna(0)
    # df_chroma.head()

    # df_chroma.to_csv("speech_emotion_recognition/features/df_chroma.csv", index=0)

def oversample(X, y):
    X = joblib.load("speech_emotion_recognition/features/X.joblib")  # mfcc
    y = joblib.load("speech_emotion_recognition/features/y.joblib")
    print(Counter(y))  # {7: 192, 4: 192, 3: 192, 1: 192, 6: 192, 2: 192, 5: 192, 0: 96}

    oversample = RandomOverSampler(sampling_strategy="minority")
    X_over, y_over = oversample.fit_resample(X, y)

    X_over_save, y_over_save = "X_over.joblib", "y_over.joblib"
    joblib.dump(X_over, os.path.join("speech_emotion_recognition/features/", X_over_save))
    joblib.dump(y_over, os.path.join("speech_emotion_recognition/features/", y_over_save))

if __name__ == "__main__":
    print("Extracting file info...")
    extract_file_info()
    print("Extracting audio features...")
    FEATURES = extract_features(
        path="speech_emotion_recognition/data/",
        save_dir="speech_emotion_recognition/features/",
    )
    print("Finished extracting audio features.")
