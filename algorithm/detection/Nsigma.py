import numpy as np

class Nsigma:
    modelname = 'NsigmaModel'

    def __init__(self,N,series):
        self.N = N
        self.series = series

    def getmodelname(self,modelname):
        return modelname

    def thresholding(self):
        mean = np.mean(self.series)
        std = np.std(self.series)
        threshold = mean + self.N * std
        return threshold

    def anomalylocation(self):
        threshold = self.thresholding()
        anomalylist = [x for x in range(len(self.series)) if self.series[x]>=threshold]
        return anomalylist

    def normallocation(self):
        threshold = self.thresholding()
        normallist = [x for x in range(len(self.series)) if self.series[x] <= threshold]
        return normallist

