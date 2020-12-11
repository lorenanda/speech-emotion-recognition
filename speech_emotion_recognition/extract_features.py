import os
import pandas as pd


def extract_features():
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


if __name__ == "__main__":
    print("Extracting features...")
    extract_features()
    print("Exported features from", len(df), "files.")


# def extract_features(PATH):

#     files_list = []
#     gender_list = []
#     emotions_list = []
#     intensity_list = []

#     for dir, _, files in os.walk(PATH):
#         for item in files:
#             if int(item[7]) == "1":
#                 emotions_list.append("neutral")
#             elif item[7] == "2":
#                 emotions_list.append("calm")
#             elif item[7] == "3":
#                 emotions_list.append("happy")
#             elif item[7] == "4":
#                 emotions_list.append("sad")
#             elif item[7] == "5":
#                 emotions_list.append("angry")
#             elif item[7] == "6":
#                 emotions_list.append("fearful")
#             elif item[7] == "7":
#                 emotions_list.append("disgusted")
#             elif item[7] == "8":
#                 emotions_list.append("surprised")

#             if int(item[-6:-4]) % 2 == 0:
#                 gender_list.append("female")
#             elif int(item[-6:-4]) % 2 != 0:
#                 gender_list.append("male")

#             if item[10] == "1":
#                 intensity_list.append("normal")
#             elif item[10] == "2":
#                 intensity_list.append("strong")

#             files_list.append(item)

#         df = pd.DataFrame()
#         df["file"] = files_list
#         df["gender"] = gender_list
#         df["emotion"] = emotions_list
#         df["intensity"] = intensity_list

#         df.to_csv("features/df_features_new.csv", index=False)


# if __name__ == "__main__":
#     # PATH = os.listdir("data/")
#     # PATH = os.walk("data/")
#     extract_features("data/")