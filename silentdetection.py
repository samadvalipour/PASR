import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from os import listdir
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import shutil
from mpl_toolkits.mplot3d import Axes3D
import csv

def detection(wave,X,y):
    l = len(wave)-1
    seglen = 500
    start = crop(wave ,seglen,X,y)
    waveinvers = wave[::-1]
    end = l - crop(waveinvers ,seglen,X,y)
    return wave[start:end]

def crop (wave ,seglen,X,y):
    start = 0
    for i in range(0 ,len(wave)-seglen,seglen):
        segment = wave[i :i+seglen]
        if not is_silent(segment,X,y) :
            start = i
            break
    return start

def is_silent(segment,X,y):
    feature = feature_extractor(segment)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)
    label = neigh.predict(np.array([feature]))
    if label == 0 :
        return True
    else:
        return False

def feature_extractor(segment):
    l = len(segment)
    E = np.log(np.sum(np.abs(segment)))
    fft = np.sum((np.abs(np.fft.fft(segment,256)))[0:256])
    zc = np.sum(np.abs(np.sign(segment)-np.sign(np.roll(segment,1)))) / 2
    return np.array([fft,E,zc])


def extract_label(str):
    label = str.split('_')[0]
    return np.int(label)
