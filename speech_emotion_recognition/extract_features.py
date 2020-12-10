import os
import pandas as pd


def extract_features(PATH):

    files_list = []
    gender_list = []
    emotions_list = []
    intensity_list = []

    for item in PATH:
        if item[7] == "1":
            emotions_list.append("neutral")
        elif item[7] == "2":
            emotions_list.append("calm")
        elif item[7] == "3":
            emotions_list.append("happy")
        elif item[7] == "4":
            emotions_list.append("sad")
        elif item[7] == "5":
            emotions_list.append("angry")
        elif item[7] == "6":
            emotions_list.append("fearful")
        elif item[7] == "7":
            emotions_list.append("disgusted")
        elif item[7] == "8":
            emotions_list.append("surprised")

        if int(item[-6:-4]) % 2 == 0:
            gender_list.append("female")
        elif int(item[-6:-4]) % 2 != 0:
            gender_list.append("male")

        if item[10] == "1":
            intensity_list.append("normal")
        elif item[10] == "2":
            intensity_list.append("strong")

        files_list.append(item)

    df = pd.DataFrame()
    df["file"] = files_list
    df["gender"] = gender_list
    df["emotion"] = emotions_list
    df["intensity"] = intensity_list

    df.to_csv("features/df_features.csv", index=False)


PATH = os.listdir("data/")
extract_features(PATH)