#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `speech_emotion_recognition` package."""
import pytest
import sys

sys.path.insert(1, "../speech_emotion_recognition")
import preprocessing_new

# from speech_emotion_recognition import preprocessing_new


# @pytest.fixture
# def response():
#     import requests

#     return requests.get("https://github.com/lorenanda/speech-emotion-recognition")


# def test_content(response):
#     """Sample pytest test function with the pytest fixture as an argument."""
#     from bs4 import BeautifulSoup

#     assert "lorenanda" in BeautifulSoup(response.content, features="lxml").title.string


def test_invalid_path():
    from preprocessing_new import extract_features

    path = 25
    with pytest.raises(TypeError):
        extract_features(path)
