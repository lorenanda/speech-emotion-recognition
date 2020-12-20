"""Main module that classifies the emotion in a live-recorded voice."""
from voice_recorder import record_voice
from predictions import make_predictions


def classify_myvoice():
    record_voice()
    make_predictions("speech_emotion_recognition/recordings/myvoice.wav")


if __name__ == "__main__":
    classify_myvoice()
