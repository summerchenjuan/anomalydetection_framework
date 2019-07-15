import numpy as np
import pandas as pd
import datetime,time
from .PredictModel import PredictModel
from esconn import esinteracton
import matplotlib.pyplot as plt

class EWMA_class:

    def __init__(self,nodename,metric,start,end,windows=3,alpha=0.7,df=None):
        super().__init__()
        self.name = 'EWMA'
        self.df = df
        self.alpha = alpha
        self.start = start
        self.end = end
        self.windows = windows
        self.metric = metric
        self.nodename = nodename

    def createdata(self):
        delta = datetime.timedelta(hours=self.windows)
        start = self.start - delta
        end = self.end
        starttime = int(round(time.mktime(start.timetuple())) * 1000)
        endtime = int(round(time.mktime(end.timetuple())) * 1000)
        self.df = esinteracton.search_nodename_timestamp_dataframe(nodename=self.nodename, starttime=starttime,
                                                                   endtime=endtime, metrics=[self.metric])


    #利用一组数据预测最后一个数据值
    def ewma_predict(self,X):
        s = [X[0]]
        i = 1
        for i in range(len(X)):
            if (np.isnan(X[i])):
                pass
            else:
                s = [X[i]]
                break
        for i in range(1,len(X)):
            if (np.isnan(X[i])):
                temp = s[-1]
            else:
                temp = self.alpha * s[-1] + ( 1 - self.alpha ) * X[i]
            s.append(temp)
        return s[-1]

    def predictpoint(self,t):
        delta = datetime.timedelta(hours=self.windows)
        start = t - delta
        end = t
        X = self.df[(self.df['timestamp']>=start)&(self.df['timestamp']<end)][self.metric].tolist()
        tp = self.ewma_predict(X)
        return tp

    def predict(self):
        starttime = self.start
        endtime = self.end
        testdata = self.df[(self.df['timestamp']>=starttime)&(self.df['timestamp']<=endtime)]['timestamp'].tolist()
        testpredictseries = []
        for date in testdata:
            testpredictseries.append(self.predictpoint(date))
        dic = {'timestamp':testdata,self.metric:testpredictseries}
        predf = pd.DataFrame(dic)
        return predf

