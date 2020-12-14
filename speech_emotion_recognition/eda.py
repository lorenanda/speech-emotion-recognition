import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def make_eda_plots():
    fig_dims = (10, 7)
    fig, ax = plt.subplots(figsize=fig_dims)

    df = pd.read_csv("speech_emotion_recognition/features/df_features.csv")

    plot_emotions = sns.countplot(
        x="emotion", data=df, color="lightseagreen", ax=ax
    ).set_title("RAVDESS Audio Dataset")
    plot_emotions.figure.savefig("images/plot_emotions.png")
    plot_intensity = sns.countplot(
        x="intensity", data=df, color="lightseagreen", ax=ax
    ).set_title("RAVDESS Audio Dataset")
    plot_intensity.figure.savefig("images/plot_intensity.png")
    plot_gender = sns.countplot(
        x="gender", data=df, color="lightseagreen", ax=ax
    ).set_title("RAVDESS Audio Dataset")
    plot_gender.figure.savefig("images/plot_gender.png")

    print("Successfully created plots.")


if __name__ == "__main__":
    make_eda_plots()
