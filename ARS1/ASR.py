import os
import sys
import signal
import argparse
import joblib
from signal import signal
import warnings
import numpy as np
from pynput import mouse
from scipy.io import wavfile
from hmmlearn import hmm
#from speech_features import mfcc
import speech_recognition 
from python_speech_features import mfcc
from vad import VoiceActivityDetector
from scipy import interpolate
import matplotlib.pyplot as plt
import librosa
import librosa.display


def on_click(x, y, button, pressed):
    if pressed:
        print('Start')
        return False
    else:
        print ('End')
        return False

def record():
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    os.system("arecord -D plughw:2,0 -f S16_LE -r 8000 -c 2 -d 3 01.wav")
    os.system("sox 01.wav -r 8000 00.wav")
    
# Define a class to train the HMM 
class ModelHMM(object):
    def __init__(self, num_components=4, num_iter=1000):
        self.n_components = num_components
        self.n_iter = num_iter

        self.cov_type = 'diag' 
        self.model_name = 'GaussianHMM' 

        self.models = []

        self.model = hmm.GaussianHMM(n_components=self.n_components, 
                covariance_type=self.cov_type, n_iter=self.n_iter)
    # 'training_data' is a 2D numpy array where each row is 13-dimensional
    def train(self, training_data):
        np.seterr(all='ignore')
        cur_model = self.model.fit(training_data)
        self.models.append(cur_model)
    def compute_score(self, input_data):
        return self.model.score(input_data)
    def save(self, path="models.pkl"):
        joblib.dump(self.models, path)
    def load(self, path="models.pkl"):
        self.models = joblib.load(path)

# Define a function to build a model for each word
def build_models(input_folder,input_way):
    # Initialize the variable to store all the models
    speech_models = []
    if input_way==1:
        for dirname in os.listdir(input_folder):
            # Get the name of the subfolder 
            if dirname=='.DS_Store':
                continue
            subfolder = os.path.join(input_folder, dirname)

            if not os.path.isdir(subfolder): 
                continue
            label = subfolder[subfolder.rfind('/') + 1:]
            X = np.array([])
            # Create a list of files to be used for training
            training_files = [x for x in os.listdir(subfolder) if x.endswith('.wav')]
            # Iterate through the training files and build the models
            for filename in training_files: 
                filepath = os.path.join(subfolder, filename)
                # Get freq and audio single
                sampling_freq, signal = wavfile.read(filepath)
                # Extract the MFCC features
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    features_mfcc = mfcc(signal, sampling_freq)
                # Append to the variable X
                if len(X) == 0:
                    X = features_mfcc
                else:
                    X = np.append(X, features_mfcc, axis=0)
            # Create the HMM model
            model = ModelHMM()
            # Train the HMM
            model.train(X)
            model.save('Train/'+label+'.pkl')
            # Save the model for the current word
            speech_models.append((model, label))
            model = None
        return speech_models
    else:
        for dirname in os.listdir(input_folder):
            if dirname=='.DS_Store':
                continue
            label = dirname
            print(label)
            model = ModelHMM()
            model.load('Train/'+label+'.pkl')
            print(model)
            # Save the model for the current word
            speech_models.append((model, label))
            model = None
        return speech_models

def run_tests(test_files):
    for test_file in test_files:
        #sampling_freq, signal = wavfile.read(test_file)
        v = VoiceActivityDetector(test_file)
        sample_rate = v.rate
        data_length = len(v.data)
        raw_detection = v.detect_speech()
        speech_labels = v.convert_windows_to_readible_labels(raw_detection)
        # v.plot_detected_speech_regions()
        #print(speech_labels)
        list = []
        for index in speech_labels: #VAD
            x = index.get("speech_begin")
            y = index.get("speech_end")
            for i in range(int(sample_rate*x), int(sample_rate*y)):
                list.append(v.data[i])
        result = np.array(list)
        result = result.astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            features_mfcc = mfcc(result, sample_rate)
        '''print(features_mfcc)
        y, sr = librosa.load(test_file)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()'''
        max_score = -float('inf') 
        output_label = None 
        for item in speech_models:
            model, label = item
            score = model.compute_score(features_mfcc) #where score is log probability
            if score > max_score:
                max_score = score
                predicted_label = label
                print(predicted_label,score)
        print('Predicted:', predicted_label)




if __name__=='__main__':
    input_wav = 'hmm-speech-recognition-0.1/audio'
    # Build an HMM model for each number
    speech_models = build_models(input_wav,1)

    print("Ready!")
    record()

    test_files = ['00.wav']
    run_tests(test_files)
    i='N'
    '''while i=='N':
    	run_tests(test_files)
    	NS=input("Is answer correct?")
    	print(i=='N')'''
    
    
    
    
    
    
    
    
    
