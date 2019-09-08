import numpy as np
from os import listdir
from os.path import isfile
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from sklearn.metrics import accuracy_score
import pywt
from python_speech_features import mfcc, logfbank, delta
from hmmlearn import hmm
import pickle
import silentdetection
import sounddevice as sd
import keyboard
import time
import sys
dataset = np.genfromtxt('dataset.csv', delimiter=',')
X = dataset[0:,0:3]
y = dataset[0:,3]

def feature_extractor(audio,sample_rate) :
    wave = np.copy(audio)

    if wave.ndim == 2 :
        wave = (wave[0: , 1] + wave[0: , 0]) / 2
    wave /= np.max(np.abs(wave),axis=0)
    wave = silentdetection.detection(wave,X,y)
    mfcc_features = mfcc(wave  , samplerate=sample_rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=256, lowfreq=0, highfreq=None, preemph=0.95, ceplifter=22, appendEnergy=True)
    return mfcc_features

def extract_label(str):
    label = str.split('_')[0]
    return int(label)

def create_train_set(dir_path):
    file_list = [file for file in listdir(dir_path)]
    dataset = {}
    for fileName in file_list:
        label = extract_label(fileName)
        sample_rate, wave =  wavfile.read(dir_path+fileName)
        feature = feature_extractor(wave,sample_rate)

        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)
        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature

    return dataset

def create_test_set(dir_path):
    file_list = [file for file in listdir(dir_path)]
    dataset = []
    for fileName in file_list :
        sample_rate, wave =  wavfile.read(dir_path+fileName)
        feature = feature_extractor(wave,sample_rate)
        dataset.append(feature)
    return dataset

def train_samad_hmm (train_data_path,states_num):

    GMMHMM_Models = {}
    train_data = create_train_set(train_data_path)
    for label in train_data.keys():
        model = hmm.GMMHMM(n_components=states_num)
        t_Data = train_data[label]
        t_Data = np.vstack(t_Data)
        model = model.fit(t_Data)  # get optimal parameters
        GMMHMM_Models[label] = model

    return GMMHMM_Models

def test_samad_hmm (GMMHMMs,test_data_path):

    test_data = create_test_set(test_data_path)
    predict_labels = []

    for test in test_data:
        scoreList = []
        for model_label in GMMHMMs.keys():
            model = GMMHMMs[model_label]
            score = model.score(test)
            scoreList.append(score)

        prediction = scoreList.index(max(scoreList))
        predict_labels.append(prediction)

    return predict_labels

def test_sample(GMMHMMs,wave,sample_rate):
    scoreList = []
    test = feature_extractor(wave,sample_rate)
    for model_label in GMMHMMs.keys():
        model = GMMHMMs[model_label]
        score = model.score(test)
        scoreList.append(score)
    prediction = scoreList.index(max(scoreList))
    return prediction

train_path = "dataset/train/"
pickles_path = "pickles/"
test_path = "dataset/test/"
all_path = "dataset/all/"


# G = train_samad_hmm (all_path,15)
# s = test_samad_hmm (G,test_path)
# for i in G.keys():
#     with open(pickles_path + str(i) + "_HMM.pkl", "wb") as file: pickle.dump(G[i],file)
#
# file_list = [file for file in listdir(test_path)]
# labels = []
# for fileName in file_list:
#     label = extract_label(fileName)
#     label = np.int(label)
#     labels.append(label)
# score = accuracy_score(s, labels)
# print("accuracy : ",score * 100 ,"%")


file_list = [file for file in listdir(pickles_path)]
GMMHMM_Models = {}
for fileName in file_list:
    label = extract_label(fileName)
    label = np.int(label)
    with open(pickles_path + fileName, "rb") as file: model = pickle.load(file)
    GMMHMM_Models[label] = model
print('please press \'SHIFT\' on hold it :')
sys.stdout.flush()
duration = 2  # seconds
fs = 8000
myrecording = 0
i = 1
while True:
    if keyboard.is_pressed('shift'):  # if key 'shift' is pressed
        myrecording = sd.rec(duration * fs, samplerate=fs, channels=2)
        time.sleep(0.2)
        print('recording...')
        sys.stdout.flush()
        sd.wait()
        print('recorded')
        sys.stdout.flush()
        label = test_sample(GMMHMM_Models,myrecording,fs)
        print("\npredicted label : ",label)
        break  # finishing the loop
    else:
        pass
