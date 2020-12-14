import os
import pandas as pd


def extract_file_info(path):

    # path = "speech_emotion_recognition/data/"
    actor_folders = os.listdir(path)

    file_path = []
    actor = []
    emotion = []
    intensity = []
    gender = []

    for i in actor_folders:
        filename = os.listdir(path + i)
        for f in filename:
            file_path.append(path + i + "/" + f)
            emotion.append(int(part[2]))
            actor.append(int(part[6]))
            intensity.append(int(part[3]))
            bg = int(part[6])
            if bg % 2 == 0:
                bg = "female"
            else:
                bg = "male"
            gender.append(bg)

    audio_df = pd.DataFrame(emotion)
    audio_df = audio_df.replace(
        {
            1: "neutral",
            2: "calm",
            3: "happy",
            4: "sad",
            5: "angry",
            6: "fearful",
            7: "disgusted",
            8: "surprised",
        }
    )
    audio_df = pd.concat(
        [pd.DataFrame(gender), audio_df, pd.DataFrame(intensity), pd.DataFrame(actor)],
        axis=1,
    )
    audio_df.columns = ["gender", "emotion", "intensity", "actor"]
    audio_df = pd.concat([audio_df, pd.DataFrame(file_path, columns=["path"])], axis=1)
    audio_df.to_csv("speech_emotion_recognition/features/df_features_new.csv", index=0)


extract_file_info(path="speech_emotion_recognition/data/")