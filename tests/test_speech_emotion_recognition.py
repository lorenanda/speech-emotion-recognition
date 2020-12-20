"""Tests for `speech_emotion_recognition` package."""
import sys
import pytest

sys.path.insert(1, "../speech_emotion_recognition")


def test_invalid_path():
    from preprocessing import extract_features

    path = 25
    with pytest.raises(TypeError):
        extract_features(path)
