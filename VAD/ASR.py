import os
import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc

input_folder = args.input_folder
hmm_models = []

for dirname in os.listdir(input_folder):
    subfolder = os.path.join(input_folder, dirname)
    if not os.path.isdir(subfolder): 
        continue
    label = subfolder[subfolder.rfind('/') + 1:]

    # initializtion
    X = np.array([])
    y_words = []

    for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
        # read audiofile
        filepath = os.path.join(subfolder, filename)
        sampling_freq, audio = wavfile.read(filepath)

        # MFCC
        mfcc_features = mfcc(audio, sampling_freq)

        # Add mfcc_features
        if len(X) == 0:
            X = mfcc_features
        else:
            X = np.append(X, mfcc_features, axis=0)

        # add label
        y_words.append(label)

# training
    print 'X.shape =', X.shape
    hmm_trainer = HMMTrainer()
    hmm_trainer.train(X)
    hmm_models.append((hmm_trainer, label))
    hmm_trainer = None

input_files = [
        'data/pineapple/pineapple15.wav',
        'data/orange/orange15.wav',
        'data/apple/apple15.wav',
        'data/kiwi/kiwi15.wav'
        ]

# Read Test file
for input_file in input_files:
    sampling_freq, audio = wavfile.read(input_file)
    mfcc_features = mfcc(audio, sampling_freq)
    max_score = None
    output_label = None
    for item in hmm_models:
        hmm_model, label = item
        score = hmm_model.get_score(mfcc_features)
        if score > max_score:
            max_score = score
            output_label = label

    print "\nTrue:", input_file[input_file.find('/')+1:input_file.rfind('/')]
    print "Predicted:", output_label 
