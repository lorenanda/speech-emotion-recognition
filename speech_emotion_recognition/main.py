"""Main module."""
from voice_recorder import record_voice
from predictions import make_predictions


def predict_myvoice():
    record_voice()
    make_predictions("speech_emotion_recognition/recordings/myvoice.wav")
