from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QAudioProbe
from PyQt5.QtCore import Qt, QUrl
import csv
from PyQt5.QtWidgets import QFileDialog, QColorDialog,QMessageBox, QGraphicsPixmapItem
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys
import numpy as np
import pandas as pd
import random
import os
import wfdb
from functools import partial
from random import randint
from scipy.signal import resample
import Utility
import pydub
import playsound
import librosa
import librosa.display
import scipy
import mplwidget
from Classes import signal,window,AudioAnalyzer
import scipy.io.wavfile as wavfile

input_Signal=signal()
output_Signal=signal()
windowsaver=window()
timer=0
cinespeed=1
ispaused=False
player=0
iswav=False
savewindows=[]
isvisible=1
outSoundExist=0
ogFile=0
samplerate=0
def browseFiles(self,inputTime,outputTime,inputspectro,outputspectro,inputFrequency,outputFrequency,music,tabIndex,allSliders,allvalues):
    global timer
    global input_Signal
    global output_Signal
    global player
    global iswav
    global ogFile
    global samplerate
    global ispaused
    self.filename, _ = QFileDialog.getOpenFileName(None, 'Open the signal file', './',
                                                   filter="Raw Data(*.csv *.txt *.xls *.hea *.dat *.rec *.wav)")
    path = self.filename
    filetype = path[len(path) - 3:]
    outputTime.clear()
    clearall(inputTime,outputTime,inputspectro,outputspectro,inputFrequency,outputFrequency)
    Utility.resetSliders(allSliders,allvalues,tabIndex)
    output_Signal.clearAllArrays()
    input_Signal.clearAllArrays()
    if filetype == "dat":
        iswav=False
        if not player==0:
            player.pause()
        self.record = wfdb.rdrecord(path[:-4], channels=[1])
        temp_arr_y = self.record.p_signal
        temp_arr_y = np.concatenate(temp_arr_y)
        temp_arr_y = temp_arr_y[:3000]
        temp_arr_x=np.linspace(0,3,3000,endpoint=False)
        self.fsampling = self.record.fs
        samplerate=self.record.fs
        maxFreq=self.fsampling/2
        input_Signal.amp = temp_arr_y
        input_Signal.time = temp_arr_x
        output_Signal.amp = temp_arr_y
        output_Signal.time = temp_arr_x
    if filetype == "csv":
        iswav = False
        if not player==0:
            player.pause()
        try:
            dataframe = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return
        except pd.errors.ParserError:
            dataframe = pd.read_csv(path, header=None)

        input_Signal.amp = dataframe.iloc[:, 1]
        input_Signal.time = dataframe.iloc[:, 0]
        output_Signal.amp = dataframe.iloc[:, 1]
        output_Signal.time = dataframe.iloc[:, 0]
        samplerate=dataframe.iloc[0,2]

    if filetype== 'wav':
        iswav=True
        y, sr = librosa.load(path, sr=None,duration=100)
        # Set up the audio player
        audio_content = QMediaContent(QUrl.fromLocalFile(path))
        ogFile=audio_content
        if not player==0:
            player.pause()
        player = QMediaPlayer(self)
        player.setMedia(audio_content)
        # Connect the audio probe to analyze audio data
        probe = AudioAnalyzer(player)
        probe.setSource(player)
        player.play()
        # Normalize the audio signal to the range [-1, 1] for better visualization
        samplerate=sr
        y2=y / np.max(np.abs(y))
        # Calculate the time values for each sample
        time = np.arange(0, len(y)) / sr
        # time =np.linspace(0,4,len(y),endpoint=False)
        input_Signal.time=time
        input_Signal.amp=y2
        output_Signal.time=time
        output_Signal.amp=y2


    setranges(inputTime,outputTime)

    input_Signal.time_domain_reference = inputTime.plot(input_Signal.time, input_Signal.amp)
    output_Signal.time_domain_reference = outputTime.plot(output_Signal.time,output_Signal.amp)
    # if timer==0:
    #     setQtimer(self,inputTime,outputTime)


    input_freq, input_magnitude, input_phase = perform_fourier_transform(input_Signal.amp,input_Signal.time)
    output_freq, output_magnitude, output_phase = perform_fourier_transform(output_Signal.amp,output_Signal.time)
    input_Signal.setfrequencydetails(input_freq, input_magnitude, input_phase, iswav, music)
    output_Signal.setfrequencydetails(output_freq, output_magnitude, output_phase, iswav, music)
    Fs = np.max(output_Signal.frequency) * 2
    drawSpectroGram(inputspectro,outputspectro,input_Signal.amp,output_Signal.amp,Fs)

    setQtimer(self)


    input_Signal.freq_domain_reference=inputFrequency.plot(input_freq,input_magnitude)
    output_Signal.freq_domain_reference=outputFrequency.plot(output_freq, output_magnitude)

    inputFrequency.getViewBox().scaleBy((0.25,1))
    outputFrequency.getViewBox().scaleBy((0.25, 1))
    input_Signal.fileOpened=True
    output_Signal.fileOpened=True











def perform_fourier_transform(amp,time):
    # Perform the Fourier transform
    signal_fft = np.fft.fft(amp)
    sampleRate=1/(time[1]-time[0])
    # Calculate the frequency values
    n = len(signal_fft)
    freq = np.fft.fftfreq(n, 1 / sampleRate)

    # Calculate magnitude and phase
    magnitude = np.abs(signal_fft)
    phase = np.angle(signal_fft)

    return freq, magnitude, phase



# Add this function to your code
def perform_inverse_fourier_transform(freq, magnitude, phase):
    signal_fft = magnitude * np.exp(1j * phase)
    signal_amp = np.fft.ifft(signal_fft).real
    return signal_amp






def updatesound():
    global timer
    global input_Signal
    global output_Signal
    global cinespeed
    global player
    global iswav

    timer.setInterval(int(100/cinespeed))
    if iswav:
        if player.state()==QMediaPlayer.StoppedState:
            player.play()


def setQtimer(self):
    global timer
    timer = QtCore.QTimer(self)
    timer.setInterval(100)
    timer.timeout.connect(partial(updatesound))
    timer.start()


def drawSpectroGram(inputSpectro,outputSpectro,inputAmp,OutputAmp,Fs):
    inputSpectro.canvas.plot_specgram(inputAmp,Fs)
    outputSpectro.canvas.plot_specgram(OutputAmp, Fs)


def pauseplay():
    global ispaused
    global timer
    global player
    global iswav
    if ispaused == False:
        ispaused = True
        timer.stop()
        if iswav:
            player.pause()
    else:
        ispaused = False
        timer.start()
        if iswav:
            player.play()


def stops():
    global timer
    global ispaused
    global input_Signal
    global output_Signal
    global player
    global iswav
    if iswav:
        player.pause()
    timer.stop()
    ispaused = True
    reset()



def reset():
    global input_Signal
    global output_Signal
    global player
    global iswav
    if iswav:
        player.setPosition(0)
    input_Signal.x_values=[]
    input_Signal.y_values=[]
    input_Signal.time_domain_reference.setData([0],[0])

    output_Signal.x_values = []
    output_Signal.y_values = []
    output_Signal.time_domain_reference.setData([0], [0])


def zoom(string,inputTime,outputTime):
    if string=="-":
        inputTime.getViewBox().scaleBy((2,1.3))
        outputTime.getViewBox().scaleBy((2,1.3))
    else:
        inputTime.getViewBox().scaleBy((0.5, 0.75))
        outputTime.getViewBox().scaleBy((0.5, 0.75))



def changespeed(string):
    global cinespeed
    global player
    global iswav
    if string == "down":
        if cinespeed == 0.25:
            return

        cinespeed=cinespeed/2
        if iswav:
            player.setPlaybackRate(cinespeed)
    else:
        if cinespeed == 4:
            return

        cinespeed = cinespeed / 0.5
        if iswav:
            player.setPlaybackRate(cinespeed)



def clearall(inputTime,outputTime,inputspectro,outputspectro,inputFrequency,outputFrequency):
    global input_Signal
    global output_Signal
    inputTime.clear()
    outputTime.clear()
    inputspectro.canvas.axes.cla()
    outputspectro.canvas.axes.cla()
    inputFrequency.clear()
    outputFrequency.clear()
    input_Signal.x_values=[]
    input_Signal.y_values=[]
    output_Signal.x_values = []
    output_Signal.y_values = []

def setranges(inputTime,outputTime):
    global input_Signal
    global output_Signal

    inputTime.setLimits(xMin=0, xMax=max(input_Signal.time), yMin=min(input_Signal.amp), yMax=max(input_Signal.amp))
    outputTime.setLimits(xMin=0, xMax=max(output_Signal.time), yMin=min(output_Signal.amp), yMax=max(output_Signal.amp))
def heightslider(rectangleradio, hammingradio, hanningradio, gaussRadio, windowvisual,lengthcontrol, alphacontrol,tabindex,heightValue,alphaValue,ismusic):
    height = lengthcontrol.value()
    heightValue.setText(str(height))
    alpha = alphacontrol.value()
    alphaValue.setText(str(alpha))
    adaptchange(rectangleradio, hammingradio, hanningradio, gaussRadio, windowvisual,tabindex,ismusic,height,alpha)
def adaptchange(rectangleradio, hammingradio, hanningradio, gaussRadio, windowvisual,tabindex,ismusic,height,alpha):
    global windowsaver
    global input_Signal
    global iswav
    windowsaver.emptywindowdata()
    # if not tabindex==0:
    Wdata=0
    currentArray=input_Signal.allArrays[tabindex].copy()
    for i in range(len(currentArray)):
        value=len(currentArray[i])
        Wdata = makeawindow(value, rectangleradio.isChecked(), hammingradio.isChecked(), hanningradio.isChecked(),
                                gaussRadio.isChecked(), alpha)
        WdataArray = np.zeros(len(Wdata))
        for j in range(len(Wdata)):
            WdataArray[j] = Wdata[j] * height
        windowsaver.windowdata.append(WdataArray)

    if Wdata is not None:
        drawwindow(Wdata, windowvisual)





def addwindow(inputFrequency,tabindex,ismusic):
    global windowsaver
    global input_Signal
    global output_Signal
    global savewindows
    removewindows(inputFrequency)
    currentArray=output_Signal.allArrays[tabindex].copy()
    for i in range(len(windowsaver.windowdata)):

        window=windowsaver.windowdata[i]
        t = np.linspace(min(currentArray[i]), max(currentArray[i]), len(window))
        # print(t.shape,window.shape)
        refrence = inputFrequency.plot(t, window, pen="r")
        savewindows.append(refrence)






def removewindows(inputFrequency):
    global windowsaver
    global input_Signal
    global output_Signal
    global savewindows
    savewindows=Utility.emptyarray(savewindows)
    inputFrequency.clear()
    input_Signal.freq_domain_reference = inputFrequency.plot(input_Signal.frequency, input_Signal.magnitude)


def drawwindow(data,windowvisual):
    windowvisual.clear()
    t=np.linspace(0,1,len(data))
    windowvisual.plot(t,data)

def modifyOutput(allSliders,i,outputTime,outputspectro,outputFrequency,allValues,tabindex,music):
    global output_Signal
    global input_Signal
    global windowsaver
    global outSoundExist
    global ispaused
    valChanged = allSliders[i].value() / 10
    allValues[i].setText(str(valChanged))
    # else:
     #send the start and finish of each part of the 4 parts and prepare needed values to change the output
    multipliedSignal,posindexstart,posindexend,negindexstart,negindexend=Utility.makeArrays(input_Signal.allStarts,input_Signal.allEnds,input_Signal.magnitude,windowsaver.windowdata,valChanged,i,tabindex)
    # send the prepared values to change the output
    changeOutput(i,posindexstart, posindexend,multipliedSignal,negindexstart, negindexend)
    #set the freq and magnitude of the output freq
    output_Signal.freq_domain_reference.setData(output_Signal.frequency, output_Signal.magnitude)
    newsignalamp = perform_inverse_fourier_transform(output_Signal.frequency, output_Signal.magnitude,
                                                     output_Signal.phase)
    outputTime.setLimits(xMin=0, xMax=max(output_Signal.time), yMin=min(output_Signal.amp),
                         yMax=max(output_Signal.amp))
    output_Signal.fixdynamicdrawing(newsignalamp)
    outputTime.setLimits(xMin=0, xMax=max(output_Signal.time), yMin=min(output_Signal.amp),
                         yMax=max(output_Signal.amp))
    outputspectro.canvas.axes.cla()
    Fs = np.max(output_Signal.frequency) * 2
    outputspectro.canvas.plot_specgram(newsignalamp, Fs)
    outputTime.clear()
    outputTime.plot(output_Signal.time,output_Signal.amp)
    if outSoundExist:
        position = player.position()
        player.setMedia(QMediaContent())
        normalized_signal = np.int16(newsignalamp / np.max(np.abs(newsignalamp)) * 32767)
        wavfile.write('modifiedOutput.wav', int(Fs), normalized_signal)
        audio_content = QMediaContent(QUrl.fromLocalFile('modifiedOutput.wav'))
        player.setMedia(audio_content)
        player.setPosition(position)
        if not ispaused:
            player.play()



def changeOutput(i,posindexstart, posindexend,multipliedSignal,negindexstart, negindexend):
    global output_Signal
    # if not i==3:
    for k in range(posindexstart, posindexend):
        output_Signal.magnitude[k] = multipliedSignal[0][k - posindexstart]
    # print(len(multipliedSignal[0]),len(multipliedSignal[1]))
    # print(posindexstart,posindexend,negindexstart,negindexend)
    for j in range(negindexstart, negindexend):
        output_Signal.magnitude[j] = multipliedSignal[1][j - negindexstart]


#     else:
#             for k in range(posindexstart, posindexend - 1):
#                 output_Signal.magnitude[k] = multipliedSignal[0][k - posindexstart]
#             for j in range(negindexstart, negindexend):
#                 output_Signal.magnitude[j] = multipliedSignal[1][j - negindexstart]
def toggleSpectroHide(inputspectro,outputspectro):
    global isvisible
    if isvisible:
        isvisible=False
        inputspectro.setVisible(False)
        outputspectro.setVisible(False)
        inputspectro.canvas.draw()
        outputspectro.canvas.draw()

    else:
        isvisible = True
        inputspectro.setVisible(True)
        outputspectro.setVisible(True)
        inputspectro.canvas.draw()
        outputspectro.canvas.draw()




def updatenames(musicButton,soundTitle1,soundTitle2,soundTitle3,soundTitle4,tabIndex,allSliders,allValues,outputTime, outputspectro, outputFrequency,inputFrequency):
    global output_Signal
    global input_Signal
    global iswav
    if output_Signal.fileOpened:
        Utility.resetOutput(outputTime, outputspectro, outputFrequency, musicButton.isChecked(), output_Signal, input_Signal, iswav)
        removewindows(inputFrequency)
        Utility.resetSliders(allSliders, allValues, tabIndex)
    if not musicButton.isChecked():
        soundTitle1.setText("Bumblebee")
        soundTitle2.setText("Dolphin")
        soundTitle3.setText("Hawk")
        soundTitle4.setText("Birds")
    else:
        soundTitle1.setText("Bass")
        soundTitle2.setText("drums")
        soundTitle3.setText("Piano")
        soundTitle4.setText("Violin")






def changeTrack(isInput):
    global outSoundExist
    global iswav
    global player
    global output_Signal
    global ogFile
    global ispaused
    if iswav:
        if isInput:
            if not outSoundExist:
                return
            else:
                position = player.position()
                player.setMedia(ogFile)
                player.setPosition(position)
                player.play()
        else:
            outSoundExist=True
            position=player.position()
            Fs = int(np.max(output_Signal.frequency) * 2)
            player.setMedia(QMediaContent())
            normalized_signal = np.int16(output_Signal.amp / np.max(np.abs(output_Signal.amp)) * 32767)
            wavfile.write('modifiedOutput.wav',Fs,normalized_signal)
            audio_content = QMediaContent(QUrl.fromLocalFile('modifiedOutput.wav'))
            player.setMedia(audio_content)
            player.setPosition(position)
            if not ispaused:
                player.play()

def changeTab(tabIndex,allSliders,allValues,outputTime, outputspectro, outputFrequency,inputFrequency,ismusic):
    global output_Signal
    global input_Signal
    global iswav
    if input_Signal.fileOpened:
        Utility.resetOutput(outputTime, outputspectro, outputFrequency,ismusic,output_Signal,input_Signal,iswav)
        Utility.resetSliders(allSliders,allValues,tabIndex)
        removewindows(inputFrequency)


def makeawindow(value,rectangleradio,hammingradio,hanningradio,gaussRadio,alphavallue):


    if rectangleradio:
            Wdata=np.kaiser(value,0)
    elif hammingradio:
            Wdata = np.hamming(value)
    elif hanningradio:
            Wdata = np.hanning(value)
    elif gaussRadio:
        Wdata = scipy.signal.windows.gaussian(value, alphavallue/ 10, sym=True)
    return Wdata














