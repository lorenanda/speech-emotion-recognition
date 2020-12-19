"""Tests for `speech_emotion_recognition` package."""
import pytest
import sys

sys.path.insert(1, "../speech_emotion_recognition")
import preprocessing_new


def test_invalid_path():
    from preprocessing_new import extract_features

    path = 25
    with pytest.raises(TypeError):
        extract_features(path)
