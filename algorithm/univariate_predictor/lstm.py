import pandas as pd
from keras import Sequential
from keras.layers import Dense,LSTM
from keras import backend
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator

class LSTM_class:
    def __init__(self,df,metric,start,end,train_start,train_end,timesteps):
        self.df = df
        self.metric = metric
        self.start = start
        self.end = end
        self.train_start = train_start
        self.train_end = train_end
        self.timesteps = timesteps

    def create_data(self):
        metricdata = [self.df[self.metric][i] for i in range(0, len(self.df))]
        scaler = MinMaxScaler(feature_range=(0, 1))
        print(np.array(metricdata).reshape(-1, 1))
        metricdata = scaler.fit_transform(np.array(metricdata).reshape(-1, 1))
        print(metricdata)
        metricdata = np.array(metricdata).flatten()
        print(metricdata)
        print(scaler)
        timestamps = [self.df['timestamp'][i] for i in range(0, len(self.df))]
        dataset = []
        columns = ['timestamp']
        columns.extend([('input' + str(i)) for i in range(1, self.timesteps + 1)])
        columns.extend(['output1'])
        for i in range(len(metricdata) - self.timesteps):
            timestamp = [timestamps[i + self.timesteps]]
            inputdata = metricdata[i:i + self.timesteps]
            outdata = [metricdata[i + self.timesteps]]
            timestamp.extend(inputdata)
            timestamp.extend(outdata)
            dataset.append(timestamp)
        self.df = pd.DataFrame(dataset, columns=columns)
        return scaler


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
        traindf = pd.DataFrame(traindf)
        traindata = traindf.values
        traindata_X = traindata[:,1:-1]
        traindata_Y = traindata[:,-1]
        testdf = self.df[(self.df['timestamp'] >= start) & (self.df['timestamp'] <= end)]
        testdf = pd.DataFrame(testdf)
        testdata = testdf.values
        testdata_X = testdata[:,1:-1]
        testdata_Y = testdata[:,-1]
        testtime =  self.df[(self.df['timestamp'] >= start) & (self.df['timestamp'] <= end)]['timestamp'].tolist()
        Train_data_X = np.array(traindata_X)
        Train_data_X = Train_data_X.reshape(Train_data_X.shape[0],Train_data_X.shape[1],1)
        Test_data_X = np.array(testdata_X)
        Test_data_X = Test_data_X.reshape(Test_data_X.shape[0], Test_data_X.shape[1],1)
        return {'Train_data_X':Train_data_X,'Test_data_X':Test_data_X,'Train_data_Y':traindata_Y,'Test_data_Y':testdata_Y,'testtime':testtime}

    # def createdata(self):
    #     starttime = datetime.datetime.strptime(self.train_start, '%Y-%m-%d %H:%M:%S')
    #     endtime = datetime.datetime.strptime(self.train_end, '%Y-%m-%d %H:%M:%S')
    #     start = datetime.datetime.strptime(self.start, '%Y-%m-%d %H:%M:%S')
    #     end = datetime.datetime.strptime(self.end, '%Y-%m-%d %H:%M:%S')
    #     traindf = self.df[(self.df['timestamp'] >= starttime) & (self.df['timestamp'] <= endtime)]
    #     traindf = pd.DataFrame(traindf,columns=[self.metric,'data'])
    #     traindata = traindf.values
    #     traindata_X = traindata[:,1:-1]
    #     traindata_Y = traindata[:,-1]
    #     testdf = self.df[(self.df['timestamp'] >= start) & (self.df['timestamp'] <= end)]
    #     testdf = pd.DataFrame(testdf,columns=[self.metric,'data'])
    #     testdata = testdf.values
    #     testdata_X = testdata[:,:-1]
    #     testdata_Y = testdata[:,-1]
    #     testtime =  self.df[(self.df['timestamp'] >= start) & (self.df['timestamp'] <= end)]['timestamp'].tolist()
    #     Train_data_X = np.array(traindata_X)
    #     Train_data_X = Train_data_X.reshape(Train_data_X.shape[0],1,Train_data_X.shape[1])
    #     Test_data_X = np.array(testdata_X)
    #     Test_data_X = Test_data_X.reshape(Test_data_X.shape[0], 1, Test_data_X.shape[1])
    #     return {'Train_data_X':Train_data_X,'Test_data_X':Test_data_X,'Train_data_Y':traindata_Y,'Test_data_Y':testdata_Y,'testtime':testtime}


    def predict(self):
        scaler = self.create_data()
        data = self.createdata()
        Train_data_X = data['Train_data_X']
        Train_data_Y = data['Train_data_Y']
        Test_data_X = data['Test_data_X']
        Test_data_Y = data['Test_data_Y']
        testtime = data['testtime']
        model = Sequential()
        model.add(LSTM(128, input_shape=(Train_data_X.shape[1], Train_data_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        # history = model.fit(Train_data_X, Train_data_Y, epochs=5, batch_size=128,
        #                     validation_data=(Test_data_X, Test_data_Y), verbose=2,
        #                     shuffle=False)
        history = model.fit(Train_data_X, Train_data_Y, epochs=10, batch_size=128,verbose=2, shuffle=False)
        model.save('D://my_model.h5')
        backend.clear_session()
        model = load_model('D://my_model.h5')
        yhat = model.predict(Test_data_X)
        print(np.array(yhat).reshape(-1, 1))
        yhat = scaler.inverse_transform(np.array(yhat).reshape(-1, 1))
        yhat = yhat.reshape(len(yhat))
        dic = {'timesatmp':testtime,'predictdata':yhat}
        predf = pd.DataFrame(dic)
        return predf


class LSTM_mulnodename_class:
    def __init__(self,df=None,metric=None,test_start=None,test_end=None,train_start=None,train_end=None,timesteps=12,nodelists=None,
                 path='D://IHEP/chenj/anomaly/sqlsave/modelpath/mymodel.h5',epochs=10,batch_size=128,verbose=2,validation_split=0.1,shuffle=True):
        self.df = df
        self.metric = metric
        self.test_start = test_start
        self.test_end = test_end
        self.train_start = train_start
        self.train_end = train_end
        self.timesteps = timesteps
        self.path = path
        self.nodelists = nodelists
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = validation_split
        self.shuffle = shuffle


#将预测问题转化为有监督问题
    def create_train_data(self):
        dataset = []
        columns = ['timestamp']
        columns.extend([('input' + str(i)) for i in range(1, self.timesteps + 1)])
        columns.extend(['output1'])
        headindex = 0
        for nodename in self.nodelists:
            dfone = self.df[self.df['nodename'] == nodename]
            metricdata = [dfone[self.metric][headindex+i] for i in range(0, len(dfone))]
            scaler = MinMaxScaler(feature_range=(0, 1))
            # print(np.array(metricdata).reshape(-1, 1))
            metricdata = scaler.fit_transform(np.array(metricdata).reshape(-1, 1))
            metricdata = np.array(metricdata).flatten()
            timestamps = [dfone['timestamp'][headindex+i] for i in range(0, len(dfone))]
            for i in range(len(metricdata) - self.timesteps):
                timestamp = [timestamps[i + self.timesteps]]
                inputdata = metricdata[i:i + self.timesteps]
                outdata = [metricdata[i + self.timesteps]]
                timestamp.extend(inputdata)
                timestamp.extend(outdata)
                dataset.append(timestamp)
            headindex = headindex + len(dfone)
        self.df = pd.DataFrame(dataset, columns=columns)

    def create_test_data(self):
        dataset = []
        columns = ['timestamp']
        columns.extend([('input' + str(i)) for i in range(1, self.timesteps + 1)])
        columns.extend(['output1'])
        metricdata = [self.df[self.metric][i] for i in range(0, len(self.df))]
        scaler = MinMaxScaler(feature_range=(0, 1))
        metricdata = scaler.fit_transform(np.array(metricdata).reshape(-1, 1))
        metricdata = np.array(metricdata).flatten()
        timestamps = [self.df['timestamp'][i] for i in range(0, len(self.df))]
        for i in range(len(metricdata) - self.timesteps):
            timestamp = [timestamps[i + self.timesteps]]
            inputdata = metricdata[i:i + self.timesteps]
            outdata = [metricdata[i + self.timesteps]]
            timestamp.extend(inputdata)
            timestamp.extend(outdata)
            dataset.append(timestamp)
        self.df = pd.DataFrame(dataset, columns=columns)
        return scaler

    def createtraindata(self):
        traindf = self.df[(self.df['timestamp'] >= self.train_start) & (self.df['timestamp'] <= self.train_end)]
        traindf = pd.DataFrame(traindf)
        traindata = traindf.values
        traindata_X = traindata[:,1:-1]
        traindata_Y = traindata[:,-1]
        Train_data_X = np.array(traindata_X)
        Train_data_X = Train_data_X.reshape(Train_data_X.shape[0],Train_data_X.shape[1],1)
        return {'Train_data_X':Train_data_X,'Train_data_Y':traindata_Y}

    def createtestdata(self):
        testdf = self.df[(self.df['timestamp'] >= self.test_start) & (self.df['timestamp'] <= self.test_end)]
        testdf = pd.DataFrame(testdf)
        testdata = testdf.values
        testdata_X = testdata[:,1:-1]
        testdata_Y = testdata[:,-1]
        testtime =  self.df[(self.df['timestamp'] >= self.test_start) & (self.df['timestamp'] <= self.test_end)]['timestamp'].tolist()
        Test_data_X = np.array(testdata_X)
        Test_data_X = Test_data_X.reshape(Test_data_X.shape[0], Test_data_X.shape[1],1)
        return {'Test_data_X':Test_data_X,'Test_data_Y':testdata_Y,'testtime':testtime}


    def train(self):
        self.create_train_data()
        data = self.createtraindata()
        Train_data_X = data['Train_data_X']
        Train_data_Y = data['Train_data_Y']
        model = Sequential()
        model.add(LSTM(128, input_shape=(Train_data_X.shape[1], Train_data_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        # history = model.fit(Train_data_X, Train_data_Y, epochs=5, batch_size=128,
        #                     validation_data=(Test_data_X, Test_data_Y), verbose=2,
        #                     shuffle=False)
        history = model.fit(Train_data_X, Train_data_Y, epochs=self.epochs, batch_size=self.batch_size,verbose=self.verbose,
                            validation_split=self.validation_split,shuffle=self.shuffle)
        return model


    def predict(self,model):
        """

        :param model:
        :return:
        """
        scaler = self.create_test_data()
        data = self.createtestdata()
        Test_data_X = data['Test_data_X']
        testtime = data['testtime']
        yhat = model.predict(Test_data_X)
        yhat = scaler.inverse_transform(np.array(yhat).reshape(-1, 1))
        yhat = yhat.reshape(len(yhat))
        dic = {'timesatmp': testtime, 'predictdata': yhat}
        predf = pd.DataFrame(dic)
        return predf

#保存为hdf5文件
    def save(self,model):
        """
        保存训练好的模型
        :param
        model:train中训练好的模型
        """
        model.save(self.path)


    def load(self):
        """
        从hdf5文件加载模型
        :return: 加载的模型
        """
        backend.clear_session()
        model = load_model(self.path)
        return model


class LSTM_mul_class:
    def __init__(self, df, nodelists,metrics, premetrics, test_start=None, test_end=None, train_start=None, train_end=None,
                 timesteps=12,
                 path='D://IHEP/chenj/anomaly/sqlsave/modelpath/my_mul_model.h5', epochs=10, batch_size=128, verbose=2,
                 validation_split=0.1, shuffle=True):
        self.df = df
        self.metrics = metrics
        self.premetrics = premetrics
        self.test_start = test_start
        self.test_end = test_end
        self.train_start = train_start
        self.train_end = train_end
        self.timesteps = timesteps
        self.path = path
        self.nodelists = nodelists
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = validation_split
        self.shuffle = shuffle


    def create_test_data(self):
        dataset = []
        columns = ['timestamp']
        columns.extend([('input' + str(i)) for i in range(1, self.timesteps + 1)])
        columns.extend(['output1'])
        metricdata = [self.df[self.metrics][i] for i in range(0, len(self.df))]
        scaler = MinMaxScaler(feature_range=(0, 1))
        metricdata = scaler.fit_transform(np.array(metricdata).reshape(-1, 1))
        metricdata = np.array(metricdata).flatten()
        timestamps = [self.df['timestamp'][i] for i in range(0, len(self.df))]
        for i in range(len(metricdata) - self.timesteps):
            timestamp = [timestamps[i + self.timesteps]]
            inputdata = metricdata[i:i + self.timesteps]
            outdata = [metricdata[i + self.timesteps]]
            timestamp.extend(inputdata)
            timestamp.extend(outdata)
            dataset.append(timestamp)
        self.df = pd.DataFrame(dataset, columns=columns)
        return scaler


    def createtestdata(self):
        testdf = self.df[(self.df['timestamp'] >= self.test_start) & (self.df['timestamp'] <= self.test_end)]
        testdf = pd.DataFrame(testdf)
        testdata = testdf.values
        testdata_X = testdata[:, 1:-1]
        testdata_Y = testdata[:, -1]
        testtime = self.df[(self.df['timestamp'] >= self.test_start) & (self.df['timestamp'] <= self.test_end)][
            'timestamp'].tolist()
        Test_data_X = np.array(testdata_X)
        Test_data_X = Test_data_X.reshape(Test_data_X.shape[0], Test_data_X.shape[1], 1)
        return {'Test_data_X': Test_data_X, 'Test_data_Y': testdata_Y, 'testtime': testtime}

    def create_train_data_onenode(self,nodename):
        """
        对单个nodename进行model训练阶段需要的输入和输出数据的构建
        :param nodename:
        :return:
        """
        train_num = 0
        Train_Data_X = []
        Train_Data_Y = []
        dfone = self.df[self.df['nodename'] == nodename]
        data = self.getTrainData(dfone)
        targets = self.getTraintarget(dfone)
        scaler1 = MinMaxScaler(feature_range=(0, 1))
        data = scaler1.fit_transform(data)
        scaler2 = MinMaxScaler(feature_range=(0, 1))
        targets = scaler2.fit_transform(targets)
        data_generator = TimeseriesGenerator(data, targets, length=self.timesteps, batch_size=1)
        for i in range(len(data_generator)):
            X, Y = data_generator[i]
            if (train_num == 0):
                Train_Data_X = X
                Train_Data_Y = Y
                train_num = train_num + 1
            else:
                Train_Data_X = np.concatenate((Train_Data_X, X))
                Train_Data_Y = np.concatenate((Train_Data_Y, Y))
        return  Train_Data_X,Train_Data_Y

    def create_train_data(self):
        """
        将每个nodename中进行lstm 训练阶段输入和输出数据进行合并
        :return:

        """
        Train_Data_X ,Train_Data_Y = self.create_train_data_onenode(self.nodelists[0])
        if(len(self.nodelists) > 0):
            for nodename in self.nodelists[1:]:
                X,Y = self.create_train_data_onenode(nodename)
                Train_Data_X = np.concatenate((Train_Data_X, X))
                Train_Data_Y = np.concatenate((Train_Data_Y, Y))
        return Train_Data_X, Train_Data_Y

    def train(self,Train_data_X,Train_data_Y):
        model = Sequential()
        model.add(LSTM(128, input_shape=(Train_data_X.shape[1], Train_data_X.shape[2])))
        model.add(Dense(Train_data_X.shape[2]))
        model.compile(loss='mae', optimizer='adam')
        history = model.fit(Train_data_X, Train_data_Y, epochs=self.epochs, batch_size=self.batch_size,
                            verbose=self.verbose,
                            validation_split=self.validation_split, shuffle=self.shuffle)
        return model


    def predict(self, model):
        """
        :param model:
        :return:
        """
        scaler = self.create_test_data()
        data = self.createtestdata()
        Test_data_X = data['Test_data_X']
        testtime = data['testtime']
        yhat = model.predict(Test_data_X)
        yhat = scaler.inverse_transform(np.array(yhat).reshape(-1, 1))
        yhat = yhat.reshape(len(yhat))
        dic = {'timesatmp': testtime, 'predictdata': yhat}
        predf = pd.DataFrame(dic)
        return predf

    # 保存为hdf5文件
    def save(self, model):
        """
        保存训练好的模型
        :param
        model:train中训练好的模型
        """
        model.save(self.path)

    def load(self):
        """
        从hdf5文件加载模型
        :return: 加载的模型
        """
        backend.clear_session()
        model = load_model(self.path)
        return model

    def getTrainData(self,dataframe):
        data_df = dataframe[self.metrics]
        return data_df.values

    def getTraintarget(self,dataframe):
        target_df =  dataframe[self.premetrics]
        return target_df.values






