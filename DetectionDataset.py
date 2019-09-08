import numpy as np
from os import listdir
from scipy.io import wavfile
import csv
import silentdetection

def create_ditection_dataset():
    a =[]
    path = "dataset/silent&vocal/"
    file_list = [file for file in listdir(path)]
    X = np.empty((1,3))
    y = np.array([])
    for fileName in file_list:
        label = silentdetection.extract_label(fileName)
        sample_rate, wave =  wavfile.read(path+fileName)
        f = silentdetection.feature_extractor(wave)
        X = np.concatenate((X,np.array([f])),axis=0)
        y = np.append(y,label)
    return X[1:,0:],y


# a,b = create_ditection_dataset()
# v = np.concatenate((a,np.reshape(b,(-1,1))),axis=1)
# np.savetxt("dataset.csv", v, delimiter=",")
