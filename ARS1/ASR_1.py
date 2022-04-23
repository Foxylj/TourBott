import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline

from utils import progress_bar_downloader
import os

#Hosting files on my dropbox since downloading from google code is painful
#Original project hosting is here: https://code.google.com/p/hmm-speech-recognition/downloads/list
#Audio is included in the zip file
link = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
dlname = 'audio.tar.gz'

if not os.path.exists('./%s' % dlname):
    progress_bar_downloader(link, dlname)
    os.system('tar xzf %s' % dlname)
else:
    print('%s already downloaded!' % dlname)
    fpaths = []
labels = []
spoken = []
for f in os.listdir('audio'):
    for w in os.listdir('audio/' + f):
        fpaths.append('audio/' + f + '/' + w)
        labels.append(f)
        if f not in spoken:
            spoken.append(f)
print('Words spoken:', spoken)

#Files can be heard in Linux using the following commands from the command line
#cat kiwi07.wav | aplay -f S16_LE -t wav -r 8000
#Files are signed 16 bit raw, sample rate 8000
from scipy.io import wavfile

data = np.zeros((len(fpaths), 32000))
maxsize = -1
for n,file in enumerate(fpaths):
    _, d = wavfile.read(file)
    data[n, :d.shape[0]] = d
    if d.shape[0] > maxsize:
        maxsize = d.shape[0]
data = data[:, :maxsize]

#Each sample file is one row in data, and has one entry in labels
print('Number of files total:', data.shape[0])
all_labels = np.zeros(data.shape[0])
for n, l in enumerate(set(labels)):
    all_labels[np.array([i for i, _ in enumerate(labels) if _ == l])] = n
    
print('Labels and label indices', all_labels)