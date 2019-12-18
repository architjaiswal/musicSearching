"""
Name: Archit Jaiswal

HW 09: Learn to extract features from the spectrogram of an audio file.

Creating something like SHASHAM, a music searching application
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import soundfile as sf
from scipy.signal import spectrogram
import glob


def classifyMusic() :
    
    # reading the test song seperately
    testsongInput = 'testSong.wav'
    testInput, testFs = sf.read(testsongInput) 
    testF, testT, testSxx = spectrogram(testInput, testFs, nperseg = testFs//2)
    resTest = np.amax(testSxx, axis = 0) #this finds 
    tempTest = testSxx.T
    testSignature = []
    
    for i in range(len(resTest)):
        index = np.where(tempTest[i] == resTest[i] )
        testSignature.append(testF[index[0][0]])

    signatureDatabase = [] #this contains signatures of all songs
    database = [] #this stores all the names of song input
    databaseSignals = [] #this stores all time domain signals of songs in database
    databaseFs = []#this stores the sampling freqency of songs in database
    
    
    for filename in glob.glob('song-*.wav'): #it will read all the .wav file starting with "song-"
        signal , fs = sf.read(filename)
        database.append(filename) # just stores the names of songs
        databaseSignals.append(signal) # stores the raw data of song in time domain
        databaseFs.append(fs) # stores the sampling rate of songs obtained from the "read()"
        
        # Sxx gives 2D array of x-axis as time and y-axis frequency values at the corrosponding time
        f, t, Sxx = spectrogram(signal, fs, nperseg = fs//2)

        #it finds the max value in each colomn of 2D array and returns 1D array of max values from each colomn
        res = np.amax(Sxx, axis = 0) #axis = 0 means find from colomn and 1 means from rows

        SongSignature=[] #this holds the signature of induvidual song
        temp = Sxx.T #will transpose the array so colomn of frequency is now row of frequencies

        #this gets the index of highest frequency present at the perticular time
        for i in range(len(res)):
            index = np.where(temp[i] == res[i])
            SongSignature.append(f[index[0][0]])
            
        signatureDatabase.append(SongSignature)
    

    normList = [] #it stores all norm comparision values with whole database
    for i in range(len(signatureDatabase)):
        norm1 = np.linalg.norm(np.subtract(testSignature, signatureDatabase[i]), ord =1)
        normList.append(norm1)

    # this had indices in accending order of norms 
    closestSongIndex = np.argsort(normList)
    
    # prints the first 5 closest norm1 values and songs name
    for i in closestSongIndex[:5]:
        print(int(normList[i]),database[i])
        
    # plots the spectrum of original, first closest and second closest
    plt.specgram(testInput, Fs = testFs)
    plt.title(testsongInput)
    plt.show()

    plt.specgram(databaseSignals[closestSongIndex[0]], Fs = databaseFs[closestSongIndex[0]])
    plt.title(database[closestSongIndex[0]])
    plt.show()
    plt.specgram(databaseSignals[closestSongIndex[1]], Fs = databaseFs[closestSongIndex[1]])
    plt.title(database[closestSongIndex[1]])
    plt.show()

###################  main  ###################
if __name__ == "__main__" :
    classifyMusic()
