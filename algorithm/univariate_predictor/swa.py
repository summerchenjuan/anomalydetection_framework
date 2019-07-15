import numpy as np
import pandas as pd
import datetime,time
from .PredictModel import PredictModel
import matplotlib.pyplot as plt
from esconn import esinteracton

class SWA_class:

    def __init__(self,nodename,metric,start,end,windows=3,df=None):
        # super().__init__()
        self.name = 'SWA'
        self.nodename = nodename
        self.df = df
        self.start = start
        self.end = end
        self.windows = windows
        self.metric = metric

    def createdata(self):
        delta = datetime.timedelta(hours=self.windows)
        start = self.start - delta
        end = self.end
        starttime = int(round(time.mktime(start.timetuple())) * 1000)
        endtime = int(round(time.mktime(end.timetuple()))*1000)
        self.df = esinteracton.search_nodename_timestamp_dataframe(nodename=self.nodename,starttime=starttime,endtime=endtime,metrics=[self.metric])

    #利用一组数据预测后一个数据值
    def swa_predict(self,X):
        s = [X[0]]
        i = 1
        for i in range(len(X)):
            if(np.isnan(X[i])):
               pass
            else:
               s = [X[i]]
               break
        for i in range(i+1,len(X)):
            if(np.isnan(X[i])):
                temp = (s[-1]*(i)+s[-1])/(i+1)
            else:
                temp = (s[-1]*(i)+X[i])/(i+1)
            s.append(temp)
        return s[-1]

    def predictpoint(self,t):
        delta = datetime.timedelta(hours=self.windows)
        start = t - delta
        end = t
        X = self.df[(self.df['timestamp']>=start)&(self.df['timestamp']<end)][self.metric].tolist()
        tp = self.swa_predict(X)
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
        # delta = datetime.timedelta(hours=self.windows)
        # start = self.start - delta
        # end = self.start
        # X = self.df[(self.df['timestamp'] >= start) & (self.df['timestamp'] < end)][self.metric].tolist()
        # testpredictseries.append(self.swa_predict(X))
        # for date in testdata[1:]:
        #     a = self.df[(self.df['timestamp'] >= end) & (self.df['timestamp'] < date)][self.metric].tolist()
        #     if(a):
        #         X.pop(0)
        #         X.extend(a)
        #     testpredictseries.append(self.swa_predict(X))
        dic = {'timestamp':testdata,self.metric:testpredictseries}
        predf = pd.DataFrame(dic)
        return predf

