==========================
Speech Emotion Recognition
==========================


.. image:: https://img.shields.io/pypi/v/speech_emotion_recognition.svg
        :target: https://pypi.python.org/pypi/speech_emotion_recognition

.. image:: https://img.shields.io/travis/lorenanda/speech_emotion_recognition.svg
        :target: https://travis-ci.org/lorenanda/speech_emotion_recognition

.. image:: https://readthedocs.org/projects/speech-emotion-recognition/badge/?version=latest
        :target: https://speech-emotion-recognition.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status



.. image:: https://www.mathworks.com/help/examples/audio_wavelet/win64/SpeakerIdentificationUsingPitchAndMFCCExample_01.png
        :target: https://www.mathworks.com/help/examples/audio_wavelet/win64/SpeakerIdentificationUsingPitchAndMFCCExample_01.png

Description
--------
This project detects emotions for voice recordings and makes live predictions on self-recorded voices. 
I used audio files (16bit, 48kHz .wav) from the `Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) <https://zenodo.org/record/1188976#.X152FYaxWis)>`_ These are 1440 recordings of 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. Filename identifiers:

- Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
- Vocal channel (01 = speech, 02 = song).
- Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
- Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
- Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
- Repetition (01 = 1st repetition, 02 = 2nd repetition).
- Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

From these audio files I extracted the Mel-Frequency Cepstral Coefficients (MFCC) and used these values to train three Neural Networks models to predict the emotion. 

========  ========
Model     Accuracy
========  ========
MLP       47.18%
CNN       60.06%
LSTM      51.29%
========  ========

Model summaries and performace plots are saved in images. I used the two best performing models (CNN and LSTM) to make predictions on new audio files (movie dialogue clips and self-recorded voice). Overall, both models predicted the correct emotion; they misclassified similar emotions in the cases when the speech expression was ambiguous.

How to use
--------
1. Clone this repo: `git clone https://github.com/lorenanda/speech-emotion-recognition.git`
2. Install the necessary libraries: `pip install -r requirements.txt`
3. Run `main.py`.

Resources
--------
Papers that inspired and helped me in this project:

* Utkarsh Garg et al. (2020). Prediction of Emotions from the Audio Speech Signals using MFCC, MEL and Chroma.
* Md. Sham-E-Ansari et al. (2020). A Neural Network Based Approach for Recognition of Basic Emotions from Speech.
* Sabur Ajibola Alim & Nahrul Khair Alang Rashid (2018). Some Commonly Used Speech Feature Extraction Algorithms.
* Wootaek Lim, Daeyoung Jang & Taejin Lee (2016). Speech Emotion Recognition using Convolutional and Recurrent Neural Networks.

Credits
-------

This package was created with Cookiecutter_ and the
`Spiced Academy Cookiecutter PyPackage <https://github.com/spicedacademy/spiced-cookiecutter-pypackage>`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

* Free software: MIT license
* Documentation: https://speech-emotion-recognition.readthedocs.io.