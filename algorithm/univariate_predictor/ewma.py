import numpy as np
import pandas as pd
import datetime
from .PredictModel import PredictModel
import matplotlib.pyplot as plt

class EWMA_class:

    def __init__(self,df,metric,start,end,windows,alpha):
        super().__init__()
        self.name = 'EWMA'
        self.df = df
        self.alpha = alpha
        self.start = start
        self.end = end
        self.windows = windows
        self.metric = metric


    #利用一组数据预测最后一个数据值
    def ewma_predict(self,X):
        s = [X[0]]
        for i in range(1,len(X)-1):
            temp = self.alpha * s[-1] + ( 1 - self.alpha ) * X[i]
            s.append(temp)
        return s[-1]

    def predictpoint(self,t):
        delta = datetime.timedelta(hours=self.windows)
        start = t - delta
        end = t
        X = self.df[(self.df['timestamp']>=start)&(self.df['timestamp']<=end)][self.metric].tolist()
        tp = self.ewma_predict(X)
        return tp

    def predict(self):
        # starttime = datetime.datetime.strptime(self.start, '%Y-%m-%d %H:%M:%S')
        # endtime = datetime.datetime.strptime(self.end, '%Y-%m-%d %H:%M:%S')
        starttime = self.start
        endtime = self.end
        testdata = self.df[(self.df['timestamp']>=starttime)&(self.df['timestamp']<=endtime)]['timestamp'].tolist()
        testpredictseries = []
        for date in testdata:
            testpredictseries.append(self.predictpoint(date))
        dic = {'timestamp':testdata,'predictdata':testpredictseries}
        predf = pd.DataFrame(dic)
        return predf

