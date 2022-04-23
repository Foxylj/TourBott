from __future__ import print_function
from ctypes import sizeof
from tempfile import tempdir
from tkinter import Label
from turtle import shape
import warnings
import os
#from scikits.talkbox.features import mfcc
import librosa
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np
import soundfile as sf
from vad import VoiceActivityDetector

def extract_mfcc(full_audio_path):
    '''v = VoiceActivityDetector(full_audio_path)
    sample_rate = v.rate
    data_length = len(v.data)
    t_per_sample = 1/sample_rate
    raw_detection = v.detect_speech()
    speech_labels = v.convert_windows_to_readible_labels(raw_detection)
    list = []
    for index in speech_labels:
        x = index.get("speech_begin")
        y = index.get("speech_end")
        for i in range(int(sample_rate*x), int(sample_rate*y)):
            list.append(v.data[i])
    result = np.array(list)
    result = result.astype(float)
    mfcc_features = librosa.feature.mfcc(y=result, sr=sample_rate, n_mfcc=20)'''






    (y, r) = librosa.load(full_audio_path, sr=None)
    #print (filename)
    #sample_rate, wave =  wavfile.read(full_audio_path)
    #wave = wave.astype(float)
    mfcc_features = librosa.feature.mfcc(y=y,sr=r,S=None, n_mfcc=20)# MFCC Result
    #mfcc_features=librosa.feature.mfcc(y=y, sr=r, S=None, n_mfcc=20)
    # mfcc(wave, nwin=int(sample_rate * 0.03), fs=sample_rate, nceps=12)[0]
    #print (mfcc_features)
    return mfcc_features

def buildDataSet(dir):
    # Filter out the wav audio files under the dir
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    #print("11111")
    for fileName in fileList:
        tmp = fileName.split('.')[0]
        label = tmp.split('_')[1] # Words to listen
        feature = extract_mfcc(dir+fileName)
        #type (feature)
        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)
        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature
        print (dataset)
    return dataset

def train_GMMHMM(dataset):
    GMMHMM_Models = {}
    states_num = 5
    GMM_mix_num = 3
    tmp_p = 1.0/(states_num-2)
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], \
                               [0, tmp_p, tmp_p, tmp_p , 0], \
                               [0, 0, tmp_p, tmp_p,tmp_p], \
                               [0, 0, 0, 0.5, 0.5], \
                               [0, 0, 0, 0, 1]],dtype=np.float64)

    startprobPrior = np.array([0.5, 0.5, 0, 0, 0],dtype=np.float64)
    #print (dataset)
    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
                           covariance_type='diag', n_iter=10)
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int64)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
        #print (trainData)
        """take transpose"""
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # get optimal parameters
        print("111111111")
        GMMHMM_Models[label] = model
    print (GMMHMM_Models)
    return GMMHMM_Models



def main():
    trainDir = './Train_test/'
    trainDataSet = buildDataSet(trainDir)
    print("Finish prepare the training data")

    hmmModels = train_GMMHMM(trainDataSet)
    print("Finish training of the GMM_HMM models for digits 0-9")

    testDir = './Test_test/'
    testDataSet = buildDataSet(testDir)

    score_cnt = 0
    for label in testDataSet.keys():
        feature = testDataSet[label]
        scoreList = {}
        for model_label in hmmModels.keys():
            model = hmmModels[model_label]
            score = model.score(feature[0])
            scoreList[model_label] = score
        predict = max(scoreList, key=scoreList.get)
        print("Test on true label ", label, ": predict result label is ", predict)
        if predict == label:
            score_cnt+=1
    print("Final recognition rate is %.2f"%(100.0*score_cnt/len(testDataSet.keys())), "%")




if __name__ == '__main__':
    main()