import pandas as pd
from keras import Sequential
from keras.layers import Dense,LSTM
import datetime
import numpy as np

class LSTM_class:
    def __init__(self,df,metric,start,end,train_start,train_end):
        self.df = df
        self.metric = metric
        self.start = start
        self.end = end
        self.train_start = train_start
        self.train_end = train_end

    def series_to_supervised(self):
        data = [self.df[self.metric][i] for i in range(1, len(self.df))]
        data.append(self.df[self.metric][len(self.df) - 1])
        self.df['data'] = data
        return self.df


    def createdata(self):
        starttime = datetime.datetime.strptime(self.train_start, '%Y-%m-%d %H:%M:%S')
        endtime = datetime.datetime.strptime(self.train_end, '%Y-%m-%d %H:%M:%S')
        start = datetime.datetime.strptime(self.start, '%Y-%m-%d %H:%M:%S')
        end = datetime.datetime.strptime(self.end, '%Y-%m-%d %H:%M:%S')
        traindf = self.df[(self.df['timestamp'] >= starttime) & (self.df['timestamp'] <= endtime)]
        traindf = pd.DataFrame(traindf,columns=[self.metric,'data'])
        traindata = traindf.values
        traindata_X = traindata[:,:-1]
        traindata_Y = traindata[:,-1]
        testdf = self.df[(self.df['timestamp'] >= start) & (self.df['timestamp'] <= end)]
        testdf = pd.DataFrame(testdf,columns=[self.metric,'data'])
        testdata = testdf.values
        testdata_X = testdata[:,:-1]
        testdata_Y = testdata[:,-1]
        testtime =  self.df[(self.df['timestamp'] >= start) & (self.df['timestamp'] <= end)]['timestamp'].tolist()
        Train_data_X = np.array(traindata_X)
        Train_data_X = Train_data_X.reshape(Train_data_X.shape[0],1,Train_data_X.shape[1])
        Test_data_X = np.array(testdata_X)
        Test_data_X = Test_data_X.reshape(Test_data_X.shape[0], 1, Test_data_X.shape[1])
        return {'Train_data_X':Train_data_X,'Test_data_X':Test_data_X,'Train_data_Y':traindata_Y,'Test_data_Y':testdata_Y,'testtime':testtime}


    def predict(self):
        self.series_to_supervised()
        data = self.createdata()
        Train_data_X = data['Train_data_X']
        Train_data_Y = data['Train_data_Y']
        Test_data_X = data['Test_data_X']
        Test_data_Y = data['Test_data_Y']
        testtime = data['testtime']
        model = Sequential()
        model.add(LSTM(48, input_shape=(Train_data_X.shape[1], Train_data_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        history = model.fit(Train_data_X, Train_data_Y, epochs=5, batch_size=72,
                            validation_data=(Test_data_X, Test_data_Y), verbose=2,
                            shuffle=False)
        yhat = model.predict(Test_data_X)
        yhat = np.array(yhat)
        yhat = yhat.reshape(len(yhat))
        print(yhat)
        dic = {'timesatmp':testtime,'predictdata':yhat}
        predf = pd.DataFrame(dic)
        return predf
