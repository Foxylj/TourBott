
import scipy.io.wavfile as wf
import numpy as np
from vad import VoiceActivityDetector
import matplotlib.pyplot as plt
import librosa
import librosa.display
from hmmlearn import hmm
# rate, data = wf.read('audio_sample.wav')
# print(sum(data[0]))
# print(data.shape)

v = VoiceActivityDetector('audio_sample.wav')
#print("sample rate", v.rate)
#print("data lenghth", len(v.data))

sample_rate = v.rate
data_length = len(v.data)
t_per_sample = 1/sample_rate
#print(t_per_sample)
raw_detection = v.detect_speech()
speech_labels = v.convert_windows_to_readible_labels(raw_detection)
# v.plot_detected_speech_regions()
#print(speech_labels)

list = []
for index in speech_labels:
    x = index.get("speech_begin")
    y = index.get("speech_end")
    for i in range(int(sample_rate*x), int(sample_rate*y)):
        list.append(v.data[i])
result = np.array(list)
result = result.astype(float)
#print(v.data.shape)
#print(result.shape)
#print(result)

mfccs = librosa.feature.mfcc(y=result, sr=sample_rate, n_mfcc=3)
print(mfccs)
S = librosa.feature.melspectrogram(y=result, sr=sample_rate, n_mels=128,fmax=8000)
#print(S.shape)

'''fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),x_axis='time', y_axis='mel', fmax=8000,ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel spectrogram')
ax[0].label_outer()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')
plt.show()'''
# plt.figure()
# plt.plot(mfccs)
# plt.show()
