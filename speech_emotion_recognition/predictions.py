from tensorflow import keras


def make_predictions(file, model):
    cnn_model = keras.models.load_model(
        "speech_emotion_recognition/models/cnn_model.h5"
    )
    prediction_data, prediction_sr = librosa.load(
        "speech_emotion_recognition/recordings/work.wav",
        res_type="kaiser_fast",
        duration=3,
        sr=22050,
        offset=0.5,
    )

    mfccs = np.mean(
        librosa.feature.mfcc(y=prediction_data, sr=prediction_sr, n_mfcc=40).T, axis=0
    )
    x = np.expand_dims(mfccs, axis=1)
    x = np.expand_dims(x, axis=0)
    predictions = cnn_model.predict_classes(x)
    print("Prediction is", predictions)  # self.convert_class_to_emotion(predictions))
