import numpy as np
import random
from pyqtgraph import PlotWidget
from PyQt5.QtMultimedia import QAudioProbe
import math
class signal():
    def __init__(self):
        self.tabindex=0
        self.fileOpened=False
        # variables that belong to time domain
        self.amp=[]
        self.time=[]
        self.x_values = [] #used for dynamic drawing of time domain
        self.y_values = [] #used for dynamic drawing of time domain
        self.time_domain_reference=0
        # variables that belong to freq domain
        self.frequency = []
        self.magnitude = []  # not sure of the y-axis in frequency domain must check
        self.phase=[]
        self.freq_domain_reference = 0
        self.ranges10=[]
        self.magnitude10=[]
        #Variables for Misc stuff that are not under a certain thing
        self.collectsound=[]
        self.startIndex=[]
        self.endIndex=[]
        # Variables for medical stuff
        self.ECGArray=[]
        #store arrays
        self.allArrays=[]
        self.allStarts=[]
        self.allEnds=[]





    def addValue(self):
        if len(self.y_values) == 0:
            self.y_values.append(self.amp[0])
        else:
            if len(self.amp) > len(self.y_values):
                self.y_values.append(self.amp[len(self.y_values)])

        if len(self.x_values) == 0:
            self.x_values.append(self.time[0])
        else:
            if len(self.time) > len(self.x_values):
                self.x_values.append(self.time[len(self.x_values)])



    def calc10Ranges(self):
        self.clear10Ranges()
        length=len(self.frequency)
        length=length/20

        for i in range(20):
            self.ranges10.append(self.fillarray(self.frequency,int(length*i),int((i+1)*length)))
            self.magnitude10.append(self.fillarray(self.magnitude, int(length * i), int((i + 1) * length)))
            self.startIndex.append(int(length * i))
            self.endIndex.append(int((i + 1) * length))
        self.allStarts.append((self.startIndex.copy()))
        self.allEnds.append((self.endIndex.copy()))


    def fixdynamicdrawing(self,newamp):
        self.amp = newamp
        lengthdrawn=len(self.y_values)
        for i in range(lengthdrawn):
            self.x_values[i]=(self.time[i])
            self.y_values[i]=(self.amp[i])

        self.time_domain_reference.setData(self.x_values,self.y_values)





    def setfrequencydetails(self,freq,magn,pha,iswav,ismu):
        self.frequency=freq
        self.magnitude=magn
        self.phase=pha
        self.calc10Ranges()
        self.readysoundsarray(iswav,ismu)
        self.medicalArray(iswav)
        self.allArrays=[self.ranges10,self.collectsound,self.ECGArray]




    def fillarray(self,freq,start,end):
        returnarray=[]
        for i in range(start,end):
            returnarray.append(freq[i])

        return (returnarray)



    def readysoundsarray(self,iswav,ismu):
        self.clearsoundpoints()
        startM = [0, 700, 2000, 7000]
        endM = [700, 2000, 7000,15000]
        startA = [100, 700, 2500, 4000]
        endA = [700, 2500, 4000, 7000]
        if iswav:
            if ismu:
                for i in range(8):
                    if i < 4:
                        filtered_arrayS = self.frequency[self.frequency >= startM[i]]
                        filtered_arrayE = self.frequency[self.frequency <= endM[i]]

                        # Find the index of the closest value in the filtered array
                        result_indexS = np.argmin(np.abs(filtered_arrayS - startM[i]))
                        result_indexE = np.argmin(np.abs(filtered_arrayE - endM[i]))
                        # Adjust the result index to the original array
                        original_indexS = np.where(self.frequency == filtered_arrayS[result_indexS])[0][0]
                        original_indexE = np.where(self.frequency == filtered_arrayE[result_indexE])[0][0]
                        self.startIndex.append(original_indexS)
                        self.endIndex.append(original_indexE)
                        self.collectsound.append(self.fillarray(self.frequency,original_indexS ,original_indexE))
                    else:
                        filtered_arrayS = self.frequency[self.frequency >= (-1*endM[i-4])]
                        filtered_arrayE = self.frequency[self.frequency <= (-1*startM[i-4])]


                        # Find the index of the closest value in the filtered array
                        result_indexS = np.argmin(np.abs(filtered_arrayS - (-1*endM[i-4])))
                        result_indexE = np.argmin(np.abs(filtered_arrayE - (-1 * startM[i - 4])))

                        # Adjust the result index to the original array
                        original_indexS = np.where(self.frequency == filtered_arrayS[result_indexS])[0][0]
                        original_indexE = np.where(self.frequency == filtered_arrayE[result_indexE])[0][0]
                        self.startIndex.append(original_indexS)
                        if i==4:
                            lastIndex=len(self.frequency)
                            self.endIndex.append(lastIndex)
                            self.collectsound.append(self.fillarray(self.frequency, original_indexS, lastIndex))
                        else:
                            self.endIndex.append(original_indexE)
                            self.collectsound.append(self.fillarray(self.frequency, original_indexS, original_indexE))



            elif not ismu:
                for i in range(8):
                    if i < 4:
                        filtered_arrayS = self.frequency[self.frequency >= startA[i]]
                        filtered_arrayE = self.frequency[self.frequency <= endA[i]]

                        # Find the index of the closest value in the filtered array
                        result_indexS = np.argmin(np.abs(filtered_arrayS - startA[i]))
                        result_indexE = np.argmin(np.abs(filtered_arrayE - endA[i]))

                        # Adjust the result index to the original array
                        original_indexS = np.where(self.frequency == filtered_arrayS[result_indexS])[0][0]
                        original_indexE = np.where(self.frequency == filtered_arrayE[result_indexE])[0][0]
                        self.startIndex.append(original_indexS)
                        self.endIndex.append(original_indexE)
                        self.collectsound.append(self.fillarray(self.frequency,original_indexS ,original_indexE))
                    else:
                        filtered_arrayS = self.frequency[self.frequency >= (-1*endA[i-4])]
                        filtered_arrayE = self.frequency[self.frequency <= (-1*startA[i-4])]

                        # Find the index of the closest value in the filtered array
                        result_indexS = np.argmin(np.abs(filtered_arrayS - (-1*endA[i-4])))
                        result_indexE = np.argmin(np.abs(filtered_arrayE -(-1*startA[i-4])))

                        # Adjust the result index to the original array
                        original_indexS = np.where(self.frequency == filtered_arrayS[result_indexS])[0][0]
                        original_indexE = np.where(self.frequency == filtered_arrayE[result_indexE])[0][0]
                        self.startIndex.append(original_indexS)
                        self.endIndex.append(original_indexE)
                        self.collectsound.append(self.fillarray(self.frequency,original_indexS,original_indexE))

            self.allStarts.append((self.startIndex.copy()))
            self.allEnds.append((self.endIndex.copy()))

    def medicalArray(self,iswav):
        self.clearmedical()
        start = [10, 0, 0, 50]
        end = [100,8, 4, 100]
        if not iswav:
            for i in range(8):
                if i < 4:
                    filtered_arrayS = self.frequency[self.frequency >= start[i]]
                    filtered_arrayE = self.frequency[self.frequency <= end[i]]
                    # Find the index of the closest value in the filtered array
                    result_indexS = np.argmin(np.abs(filtered_arrayS - start[i]))
                    result_indexE = np.argmin(np.abs(filtered_arrayE - end[i]))
                    # Adjust the result index to the original array
                    original_indexS = np.where(self.frequency == filtered_arrayS[result_indexS])[0][0]
                    original_indexE = np.where(self.frequency == filtered_arrayE[result_indexE])[0][0]
                    self.startIndex.append(original_indexS)
                    self.endIndex.append(original_indexE)
                    self.ECGArray.append(self.fillarray(self.frequency, original_indexS, original_indexE))
                else:
                    filtered_arrayS = self.frequency[self.frequency >= (-1 * end[i - 4])]
                    filtered_arrayE = self.frequency[self.frequency <= (-1 * start[i - 4])]
                    # Find the index of the closest value in the filtered array
                    result_indexS = np.argmin(np.abs(filtered_arrayS - (-1 * end[i - 4])))
                    result_indexE = np.argmin(np.abs(filtered_arrayE - (-1 * start[i - 4])))

                    # Adjust the result index to the original array
                    original_indexS = np.where(self.frequency == filtered_arrayS[result_indexS])[0][0]
                    original_indexE = np.where(self.frequency == filtered_arrayE[result_indexE])[0][0]
                    self.startIndex.append(original_indexS)
                    if i == 5 or i==6:
                        lastIndex = len(self.frequency)
                        self.endIndex.append(lastIndex)
                        self.ECGArray.append(self.fillarray(self.frequency, original_indexS, lastIndex))
                    else:
                        self.endIndex.append(original_indexE)
                        self.ECGArray.append(self.fillarray(self.frequency, original_indexS, original_indexE))
            self.allStarts.append((self.startIndex.copy()))
            self.allEnds.append((self.endIndex.copy()))




    def clear10Ranges(self):
        self.startIndex=[]
        self.endIndex=[]
        self.ranges10=[]
        self.magnitude10=[]

    def clearsoundpoints(self):
            self.startIndex=[]
            self.endIndex = []
            self.collectsound=[]

    def clearmedical(self):
        self.ECGArray=[]
        self.startIndex = []
        self.endIndex = []

    def clearAllArrays(self):
        self.allArrays=[]
        self.allStarts=[]
        self.allEnds=[]

class AudioAnalyzer(QAudioProbe):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data = []

    def processBuffer(self, buffer):
        # Convert the buffer to a numpy array
        data = np.frombuffer(buffer.data(), dtype=np.int16)
        self.audio_data.extend(data)











class window():
    def __init__(self):
        self.type=0
        self.windowdata=[]

    def emptywindowdata(self):
        self.windowdata=[]





