'''# Using IO module to read Audio Files
from scipy.io import wavfile
freq_sample, sig_audio = wavfile.read("/content/Welcome.wav")
# Output the parameters: Signal Data Type, Sampling Frequency and Duration
print('\nShape of Signal:', sig_audio.shape)
print('Signal Datatype:', sig_audio.dtype)
print('Signal duration:', round(sig_audio.shape[0] / float(freq_sample), 2), 'seconds')
'''





# Install the SpeechRecognition and pipwin classes to work with the Recognizer() class
# Below are a few links that can give details about the PyAudio class we will be using to read direct microphone input into the Jupyter Notebook
# https://anaconda.org/anaconda/pyaudio
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
# To install PyAudio, Run in the Anaconda Terminal CMD: conda install -c anaconda pyaudio
# Pre-requisite for running PyAudio installation - Microsoft Visual C++ 14.0 or greater will be required. Get it with "Microsoft C++ Build Tools" : https://visualstudio.microsoft.com/visual-cpp-build-tools/
# To run PyAudio on Colab, please install PyAudio.whl in your local system and give that path to colab for installation
import speech_recognition as speech_recog
# Creating a recording object to store input
rec = speech_recog.Recognizer()
# Importing the microphone class to check availabiity of microphones
mic_test = speech_recog.Microphone()
# List the available microphones
speech_recog.Microphone.list_microphone_names()
# We will now directly use the microphone module to capture voice input. Specifying the second microphone to be used for a duration of 3 seconds. The algorithm will also adjust given input and clear it of any ambient noise
with speech_recog.Microphone(device_index=1) as source: 
    rec.adjust_for_ambient_noise(source, duration=3)
    print("Reach the Microphone and say something!")
    audio = rec.listen(source)

try:
    print("I think you said: \n" + rec.recognize_google(audio))
except Exception as e:
    print(e)