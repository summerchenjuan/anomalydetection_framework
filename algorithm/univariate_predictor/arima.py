# Arima
import itertools
import warnings
import statsmodels.api as sm
from esconn import esinteracton
import datetime,time
import numpy as np
import pandas as pd


class Arima_class():
    # pdq_range:{pmin:?,pmax:?,dmin:?,dmax:?,qmin:?,qmax:?}
    def __init__(self,nodename,metric,start,end,windows=3,df=None,
                 pdq_range={'pmin':0,'pmax':3,'dmin':0,'dmax':3,'qmin':0,'qmax':3},
                 seasonal_para=12):
        self.nodename = nodename
        self.metric = metric
        self.start = start
        self.end = end
        self.windows = windows
        self.pdq_range = pdq_range
        pmin = pdq_range['pmin']
        pmax = pdq_range['pmax']
        dmin = pdq_range['dmin']
        dmax = pdq_range['dmax']
        qmin = pdq_range['qmin']
        qmax = pdq_range['qmax']
        p = range(pmin, pmax)
        d = range(dmin, dmax)
        q = range(qmin, qmax)
        self.pdq = list(itertools.product(p, d, q))
        self.seasonal_pdq = [(x[0], x[1], x[2], seasonal_para)
                             for x in list(itertools.product(p, d, q))]


    def createdata(self):
        delta = datetime.timedelta(hours=self.windows)
        start = self.start - delta
        end = self.end
        starttime = int(round(time.mktime(start.timetuple())) * 1000)
        endtime = int(round(time.mktime(end.timetuple()))*1000)
        self.df = esinteracton.search_nodename_timestamp_dataframe(nodename=self.nodename,starttime=starttime,endtime=endtime,metrics=[self.metric])


 #利用一组数据预测后一个数据值
    def arima_predict(self,X):
        X = np.array(X)
        X = np.delete(X, np.where(np.isnan(X)))
        # SARIMAX_model = []
        # warnings.filterwarnings("ignore")
        # for param in self.pdq:
        #     for param_seasonal in self.seasonal_pdq:
        #         try:
        #             mod = sm.tsa.statespace.SARIMAX(X,
        #                                             order=param,
        #                                             seasonal_order=param_seasonal,
        #                                             enforce_stationarity=False,
        #                                             enforce_invertibility=False)
        #             results = mod.fit(disp=False)
        #             SARIMAX_model.append([param, param_seasonal, results.aic])
        #         except:
        #             continue
        # SARIMAX_model = np.array(SARIMAX_model)
        # aicmin = np.argmin(SARIMAX_model[:, 2])
        # param = SARIMAX_model[aicmin, 0]
        # param_seasonal = SARIMAX_model[aicmin, 1]
        param = (0,2,2)
        param_seasonal = (0,2,0,12)
        mod = sm.tsa.statespace.SARIMAX(X,
                                        order=param,
                                        seasonal_order=param_seasonal,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        result_model = mod.fit(disp=False)
        pre = result_model.get_prediction(start=len(X))
        pre_value = pre.predicted_mean
        return pre_value[0]

    def predictpoint(self,t):
        delta = datetime.timedelta(hours=self.windows)
        start = t - delta
        end = t
        X = self.df[(self.df['timestamp']>=start)&(self.df['timestamp']<end)][self.metric].tolist()
        tp = self.arima_predict(X)
        return tp

    def predict(self):
        starttime = self.start
        endtime = self.end
        testdata = self.df[(self.df['timestamp']>=starttime)&(self.df['timestamp']<=endtime)]['timestamp'].tolist()
        testpredictseries = []
        for date in testdata:
            # print(date)
            testpredictseries.append(self.predictpoint(date))
        dic = {'timestamp':testdata,self.metric:testpredictseries}
        predf = pd.DataFrame(dic)
        return predf

if __name__ == '__main__':
    dic = {'pmin': 0, 'pmax': 2, 'dmin': 0, 'dmax': 2, 'qmin': 0, 'qmax': 2}
    m = Arima(dic, 2)
    print(m.pdq)
    print(m.seasonal_pdq)

    data = pd.read_csv('D://IHEP/PycharmProject/Predictor/Arima/international-airline-passengers.csv', engine='python',
                       skipfooter=2)
    data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m-%d')
    data.set_index(['Month'], inplace=True)
    train_data = data['1949-01-01':'1959-12-01']
    result = m.fit(train_data)
    pre_value = m.forecast(result_model=result, start='1961-01-01', end='1963-12-01')
    print(type(pre_value))
    print(pre_value[pre_value[0]])

