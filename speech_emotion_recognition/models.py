"""This module trains three neural network models on 
the dataset recordings and saves the X and y features."""

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
    LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


def mlp_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(100,),
        solver="adam",
        alpha=0.001,
        shuffle=True,
        verbose=True,
        momentum=0.8,
    )
    mlp_model.fit(x_train, y_train)

    mlp_pred = mlp_model.predict(x_test)
    mlp_accuracy = mlp_model.score(x_test, y_test)
    print("Accuracy: {:.2f}%".format(mlp_accuracy * 100))  # 47.57%

    mlp_clas_report = pd.DataFrame(
        classification_report(y_test, mlp_pred, output_dict=True)
    ).transpose()
    clas_report.to_csv("speech_emotion_recognition/features/mlp_clas_report.csv")
    print(classification_report(y_test, mlp_pred))


def lstm_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_lstm = np.expand_dims(X_train, axis=2)
    X_test_lstm = np.expand_dims(X_test, axis=2)

    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(40, 1), return_sequences=True))
    lstm_model.add(LSTM(32))
    lstm_model.add(Dense(32, activation="relu"))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(Dense(8, activation="softmax"))

    lstm_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    lstm_model.summary()

    # train model
    lstm_history = lstm_model.fit(X_train_lstm, y_train, batch_size=32, epochs=100)

    # evaluate model on test set
    test_loss, test_acc = lstm_model.evaluate(X_test_lstm, y_test, verbose=2)
    print("\nTest accuracy:", test_acc)

    # plot accuracy/error for training and validation
    plt.plot(lstm_history.history["loss"])
    plt.title("LSTM model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig("speech_emotion_recognition/images/lstm_loss.png")
    plt.close()

    # Plot model accuracy
    plt.plot(lstm_history.history["accuracy"])
    plt.title("LSTM model accuracy")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig("speech_emotion_recognition/images/lstm_accuracy.png")
    plt.close


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
    model.add(Conv1D(16, 5, padding="same", input_shape=(40, 1)))
    model.add(Activation("relu"))
    model.add(Conv1D(8, 5, padding="same"))
    model.add(Activation("relu"))
    # model.add(Dropout(0.1))  # 0.3
    # model.add(MaxPooling1D(pool_size=(8)))
    model.add(
        Conv1D(
            8,
            5,
            padding="same",
        )
    )
    model.add(Activation("relu"))
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
        batch_size=50,  # 100
        epochs=100,  # 50
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
    plt.savefig("speech_emotion_recognition/images/cnn_loss2.png")
    plt.close()

    # Plot model accuracy
    plt.plot(cnn_history.history["accuracy"])
    plt.plot(cnn_history.history["val_accuracy"])
    plt.title("CNN model accuracy")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.legend(["train", "test"])
    plt.savefig("speech_emotion_recognition/images/cnn_accuracy2.png")

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

    # predictions_array = np.array([cnn_pred, y_test])
    # predictions_df = pd.DataFrame(data=predictions_array)  # .flatten())
    # predictions_df = predictions_df.T
    # predictions_df = predictions_df.rename(columns={0: "cnn_pred", 1: "y_test"})

    clas_report = pd.DataFrame(
        classification_report(y_test_int, cnn_pred, output_dict=True)
    ).transpose()
    clas_report.to_csv("speech_emotion_recognition/features/cnn_clas_report.csv")
    print(classification_report(y_test_int, cnn_pred))

    if not os.path.isdir("speech_emotion_recognition/models"):
        os.makedirs("speech_emotion_recognition/models")

    model_path = os.path.join("speech_emotion_recognition/models", "cnn_model.h5")
    model.save(model_path)
    print("Saved trained model at %s " % model_path)


if __name__ == "__main__":
    print("Training started")
    X = joblib.load("speech_emotion_recognition/features/X.joblib")
    y = joblib.load("speech_emotion_recognition/features/y.joblib")
    cnn_model(X=X, y=y)
    print("Model finished.")
