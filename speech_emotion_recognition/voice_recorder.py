import soundfile as sf
import sounddevice as sd
from scipy.io.wavfile import write


def record_voice():
    """This function records your voice and saves the output as .wav file."""
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording
    # sd.default.device = "Built-in Audio"  # Speakers full name here

    print("Say something:")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write("speech_emotion_recognition/recordings/myvoice.wav", fs, myrecording)
    print("Voice recording saved.")
