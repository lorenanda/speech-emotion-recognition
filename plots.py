import matplotlib.pyplot as plt
import seaborn as sns


def make_plots():
    fig_dims = (10, 7)
    fig, ax = plt.subplots(figsize=fig_dims)

    df = pd.read_csv("features/speech_features.csv")

    plot_emotions = sns.countplot(x="emotion", data=df, color="lightseagreen", ax=ax)
    plot_intensity = sns.countplot(x="intensity", data=df, color="lightseagreen", ax=ax)
