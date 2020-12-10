
"""
Neural network train file.
"""
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, Activation
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def cnn_model(X, y) -> None:
    """
    This function trains the neural network.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)

    print(x_traincnn.shape, x_testcnn.shape)

    model = Sequential()
    model.add(Conv1D(64, 5, padding='same',
                        input_shape=(40, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('softmax'))

    print(model.summary)

    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

    cnn_history = model.fit(x_traincnn, y_train,
                            batch_size=16, epochs=50,
                            validation_data=(x_testcnn, y_test))

    # Loss plotting
    plt.plot(cnn_history.history['loss'])
    plt.plot(cnn_history.history['val_loss'])
    plt.title('CNN model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('images/cnn_loss.png')
    plt.close()

    # Accuracy plotting
    plt.plot(cnn_history.history['accuracy'])
    plt.plot(cnn_history.history['val_accuracy'])
    plt.title('CNN model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('images/cnn_accuracy.png')

    predictions = model.predict_classes(x_testcnn)
    new_y_test = y_test.astype(int)
    matrix = confusion_matrix(new_y_test, predictions)

    print(classification_report(new_y_test, predictions))
    print(matrix)

    model_name = 'cnn_model.h5'

    # Save model and weights
    if not os.path.isdir('./models'):
        os.makedirs('./models')
    model_path = os.path.join('./models', model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


if __name__ == '__main__':
    print('Training started')
    X = joblib.load('features/X.joblib')
    y = joblib.load('features/y.joblib')
    CNN_MODEL = cnn_model(X=X, y=y)