#https://hnjia00.github.io/2020/11/06/%E8%A8%80%E8%AF%AD%E4%BF%A1%E6%81%AF-%E5%9F%BA%E4%BA%8EHMM-GMM%E7%9A%84%E5%8D%95%E4%B8%AA%E8%AF%8D%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB/
from cProfile import label
from python_speech_features import *
from scipy.io import wavfile
from hmmlearn import hmm
import joblib
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

np.random.seed(10)

def generate_wav_label(wavpath):
    wavdict = {}
    labeldict = {}
    for (dirpath, dirnames, filenames) in os.walk(wavpath):
        # print(dirpath,dirnames,filenames)
        for filename in filenames:
            if filename.endswith('.wav'):
                filepath = os.sep.join([dirpath, filename])
                fileid = filepath
                wavdict[fileid] = filepath

                label = ''
                language = wavpath.split('_')[1]
                if language == 'en':
                    label = filepath.split('_')[-1].split('.')[0]
                if language == 'ch':
                    label = filepath.split('/')[1]

                labeldict[fileid] = label
    return wavdict, labeldict

def compute_mfcc(file):
    fs, audio = wavfile.read(file)
    mfcc_feat = mfcc(audio, samplerate=fs, numcep=13, winlen=0.025, winstep=0.01, nfilt=26, nfft=2048, lowfreq=0,
                     highfreq=None, preemph=0.97)
    d_mfcc_feat = delta(mfcc_feat, 1)
    d_mfcc_feat2 = delta(mfcc_feat, 2)

    feature_mfcc = np.hstack((mfcc_feat, d_mfcc_feat, d_mfcc_feat2))
    return feature_mfcc


class Model():
    def __init__(self, CATEGORY=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], n_comp=1, n_mix=1, cov_type='full'):
        super(Model, self).__init__()
        # print(CATEGORY)
        self.CATEGORY = CATEGORY
        self.category = len(CATEGORY)
        self.n_comp = n_comp
        self.n_mix = n_mix
        self.cov_type = cov_type

        self.models = []
        #self.n_mix = [4, 3, 2, 3, 2, 3, 4, 4, 2, 3]
        for k in range(self.category):
            # print(k)
            model = hmm.GMMHMM(n_components=self.n_comp, n_mix=self.n_mix, covariance_type=self.cov_type)
            self.models.append(model)

    # train model
    def train(self, wavdict=None, labeldict=None):
        for k in range(self.category):
            model = self.models[k]
            for x in wavdict:
                if labeldict[x] == self.CATEGORY[k]:
                    print('k=', k, wavdict[x])
                    mfcc_feat = compute_mfcc(wavdict[x])
                    print(mfcc_feat)
                    model.fit(mfcc_feat)

    # Test model
    def test(self, filepath):
        result = []
        for k in range(self.category):
            model = self.models[k]
            mfcc_feat = compute_mfcc(filepath)
            re = model.score(mfcc_feat)
            result.append(re)

        # Select the tag with the highest score
        result = np.argmax(np.array(result))
        result = self.CATEGORY[result]
        return result

    def save(self, path="models.pkl"):
        joblib.dump(self.models, path)

    def load(self, path="models.pkl"):
        self.models = joblib.load(path)

# 生成test_size个测试用例
def split_dataset(wavdict, labeldict, test_size=10):
    nums = len(labeldict)
    shuf_arr = np.arange(nums)
    np.random.shuffle(shuf_arr)
    labelarr = []
    for l in labeldict:
        labelarr.append(l)
    wavdict_test = {}
    labeldict_test = {}
    wavdict_train = {}
    labeldict_train = {}

    for i in range(test_size):
        wavdict_test[labelarr[shuf_arr[i]]] = wavdict[labelarr[shuf_arr[i]]]
        labeldict_test[labelarr[shuf_arr[i]]] = labeldict[labelarr[shuf_arr[i]]]
    for k in labeldict:
        if k not in labeldict_test:
            wavdict_train[k] = wavdict[k]
            labeldict_train[k] = labeldict[k]

    return wavdict_train, labeldict_train, wavdict_test, labeldict_test

if __name__ == '__main__':
    dataset = "data_en_train"
    wavdict, labeldict = generate_wav_label(dataset)
    #print(wavdict,labeldict)
    wavdict_train, labeldict_train, wavdict_test, labeldict_test = split_dataset(wavdict, labeldict)

    models = Model()
    print("GMMHMM train:", dataset)
    models.train(wavdict=wavdict_train, labeldict=labeldict_train)
    models.save()

    print("GMMHMM test:", dataset)
    models.load()

    TP = 0
    FP = 0
    for k in wavdict_test:
        wav_path = wavdict_test[k]
        res = models.test(wav_path)[0]
        print(wavdict_test[k], res, labeldict_test[k])
        if res == labeldict_test[k]:
            TP += 1
        else:
            FP += 1
    print('TP:', TP)
    print('FP:', FP)
    print('Acc:', TP / (TP + FP))
