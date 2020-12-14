import os
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    Flatten,
    Dropout,
    Activation,
    MaxPooling1D,
    BatchNormalization,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def mlp_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MLPClassifier()
    model.fit(X_train, y_train)
    mlp_pred = model.predict(X_test)

    accuracy = accuracy_score(y_true=y_test, y_pred=mlp_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))


def cnn_model(X, y):
    """
    This function transforms the X and y features,
    trains a convolutional neural network, and plots the results.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)

    model = Sequential()
    model.add(Conv1D(64, 5, padding="same", input_shape=(40, 1)))
    model.add(Activation("relu"))
    model.add(Conv1D(128, 5, padding="same"))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(
        Conv1D(
            128,
            5,
            padding="same",
        )
    )
    model.add(Activation("relu"))
    model.add(
        Conv1D(
            128,
            5,
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",  # rmsprop
        metrics=["accuracy"],
    )

    cnn_history = model.fit(
        x_traincnn,
        y_train,
        batch_size=100,
        epochs=100,
        validation_data=(x_testcnn, y_test),
    )

    # plot_model(
    #     model,
    #     to_file="speech_emotion_recognition/images/cnn_model_summary.png",
    #     show_shapes=True,
    #     show_layer_names=True,
    # )

    # Plot model loss
    plt.plot(cnn_history.history["loss"])
    plt.plot(cnn_history.history["val_loss"])
    plt.title("CNN model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"])
    plt.savefig("speech_emotion_recognition/images/cnn_loss.png")
    plt.close()

    # Plot model accuracy
    plt.plot(cnn_history.history["accuracy"])
    plt.plot(cnn_history.history["val_accuracy"])
    plt.title("CNN model accuracy")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.legend(["train", "test"])
    plt.savefig("speech_emotion_recognition/images/cnn_accuracy.png")

    # Evaluate the model
    cnn_pred = model.predict_classes(x_testcnn)
    y_test_int = y_test.astype(int)

    matrix = confusion_matrix(y_test_int, cnn_pred)
    print(matrix)

    plt.figure(figsize=(12, 10))
    emotions = [
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgusted",
        "surprised",
    ]
    cm = pd.DataFrame(matrix)
    ax = sns.heatmap(
        matrix,
        linecolor="white",
        cmap="crest",
        linewidth=1,
        annot=True,
        fmt="",
        xticklabels=emotions,
        yticklabels=emotions,
    )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title("CNN Model Confusion Matrix", size=20)
    plt.xlabel("predicted emotion", size=14)
    plt.ylabel("actual emotion", size=14)
    plt.savefig("speech_emotion_recognition/images/CNN_confusionmatrix.png")
    plt.show()

    predictions_array = np.array([cnn_pred, y_test])
    predictions_df = pd.DataFrame(data=predictions_array)  # .flatten())
    predictions_df = predictions_df.T
    predictions_df = predictions_df.rename(columns={0: "cnn_pred", 1: "y_test"})

    clas_report = pd.DataFrame(
        classification_report(y_test_int, cnn_pred, output_dict=True)
    ).transpose()
    las_report.to_csv("speech_emotion_recognition/features/cnn_clas_report.csv")
    print(classification_report(y_test_int, cnn_pred))

    # Export the trained model
    model_name = "cnn_model.h5"

    if not os.path.isdir("speech_emotion_recognition/models"):
        os.makedirs("speech_emotion_recognition/models")

    model_path = os.path.join("speech_emotion_recognition/models", model_name)
    model.save(model_path)
    print("Saved trained model at %s " % model_path)


if __name__ == "__main__":
    print("Training started")
    X = joblib.load("speech_emotion_recognition/features/X.joblib")
    y = joblib.load("speech_emotion_recognition/features/y.joblib")
    cnn_model(X=X, y=y)
    print("Model finished.")
