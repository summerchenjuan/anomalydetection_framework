import numpy as np
import pandas as pd
import datetime,time
from esconn import esinteracton
from sklearn.linear_model import LinearRegression

class LR_class:

    def __init__(self,nodename,metric,start,end,windows=3,df=None):
        # super().__init__()
        self.name = 'LR'
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
    def lr_predict(self,X):
        """
        :param X:
        :return:
        """
        X = np.array(X)
        X=np.delete(X, np.where(np.isnan(X)))
        tx = np.array(range(0,len(X))).reshape(-1,1)
        X = X.reshape(-1,1)
        regr = LinearRegression().fit(tx, X)
        s = regr.predict([[len(X)]])
        s = s.flatten()
        return s[0]

    def predictpoint(self,t):
        delta = datetime.timedelta(hours=self.windows)
        start = t - delta
        end = t
        X = self.df[(self.df['timestamp']>=start)&(self.df['timestamp']<end)][self.metric].tolist()
        tp = self.lr_predict(X)
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

