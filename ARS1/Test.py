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
from scipy import interpolate


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

    # Run the HMM model for inference on input data
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

            # Extract the label
            label = subfolder[subfolder.rfind('/') + 1:]
            # Initialize the variables
            X = np.array([])

            # Create a list of files to be used for training
            # We will leave one file per folder for testing
            training_files = [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]

            # Iterate through the training files and build the models
            for filename in training_files: 
                # Extract the current filepath
                filepath = os.path.join(subfolder, filename)

                # Read the audio signal from the input file
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
            print(model)
            # Reset the variable
            model = None
        return speech_models
    else:
        for dirname in os.listdir(input_folder):
            if dirname=='.DS_Store':
                continue
            label = dirname
            print(label)
            model = ModelHMM()
            #model.train(X)
            model.load('Train/'+label+'.pkl')
            print(model)
            # Save the model for the current word
            speech_models.append((model, label))

            # Reset the variable
            model = None
        return speech_models

# Define a function to run tests on input files
def run_tests(test_files):
    # Classify input data
    for test_file in test_files:
        # Read input file
        sampling_freq, signal = wavfile.read(test_file)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            features_mfcc = mfcc(signal, sampling_freq)

        # Define variables
        max_score = -float('inf') 
        output_label = None 

        # Run the current feature vector through all the HMM
        # models and pick the one with the highest score
        for item in speech_models:
            model, label = item
            score = model.compute_score(features_mfcc)
            if score > max_score:
                max_score = score
                predicted_label = label
                print(predicted_label)

        print('Predicted:', predicted_label)




if __name__=='__main__':
    #args = build_arg_parser().parse_args()
    input_wav = 'hmm-speech-recognition-0.1/audio'
    # Build an HMM model for each word
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
    
    
    
    
    
    
    
    
    
