import pandas as pd
from keras import Sequential
from keras.layers import Dense,LSTM
from keras import backend
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
from esconn import esinteracton
import time
from keras.optimizers import Adam
from algorithm import model_setting
from keras.callbacks import ReduceLROnPlateau
from matplotlib import  pyplot
from algorithm.model_setting import PATH
from algorithm.TrainModel import TrainModel

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
                 path='D://IHEP/chenj/anomaly/sqlsave/modelpath/my_mul_es_model.h5',epochs=10,batch_size=128,verbose=2,validation_split=0.1,shuffle=True):
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
                 path='D://IHEP/chenj/anomaly/sqlsave/modelpath/my_mul_es_model.h5', epochs=10, batch_size=128, verbose=2,
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
        scaler.fit([[0],[150000000]])#bytes
        metricdata = scaler.transform(np.array(metricdata).reshape(-1, 1))
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


#keras LSTM
# keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
#                   kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
#                   unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
#                   activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
#                   bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1,
#                   return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)

# class LSTM_mul_es_class:
#
#     def __init__(self,metrics,premetrics, modelname=None,test_start=None, test_end=None, train_start=None, train_end=None,nodename=None,nodelists=None,
#                  timesteps=12,epochs=100, batch_size=128, verbose=1,validation_split=0.1,seed=42, shuffle=True):
#         self.metrics = metrics
#         self.premetrics = premetrics
#         self.test_start = test_start
#         self.test_end = test_end
#         self.train_start = train_start
#         self.train_end = train_end
#         self.timesteps = int(timesteps)
#         self.path = PATH+modelname+'.h5'
#         self.nodelists = nodelists
#         self.epochs = int(epochs)
#         self.batch_size = int(batch_size)
#         self.verbose = int(verbose)
#         self.validation_split = float(validation_split)
#         self.seed = int(seed)
#         self.shuffle = shuffle
#         self.nodename = nodename
#
#     def create_train_data_onenode(self,nodename,scalerdata,scalertargets):
#         """
#         对单个nodename进行model训练阶段需要的输入和输出数据的构建
#         :param nodename:
#         :return:
#         """
#         train_num = 0
#         Train_Data_X = []
#         Train_Data_Y = []
#         starttime = int(round(time.mktime(self.train_start.timetuple()))*1000)
#         endtime = int(round(time.mktime(self.train_end.timetuple()))*1000)
#         dfone = esinteracton.search_nodename_timestamp_dataframe_miss(nodename=nodename,starttime=starttime,endtime=endtime,metrics=self.metrics)
#         dfonet = esinteracton.search_nodename_timestamp_dataframe_miss(nodename=nodename,starttime=starttime,endtime=endtime,metrics=self.premetrics)
#         data = self.getTrainData(dfone)
#         targets = self.getTrainTarget(dfonet)
#         data = scalerdata.transform(data)
#         targets = scalertargets.transform(targets)
#         data_generator = TimeseriesGenerator(data, targets, length=self.timesteps, batch_size=128)
#         for i in range(len(data_generator)):
#             X, Y = data_generator[i]
#             index_nanx = np.where(np.isnan(X))[0]
#             index_nany = np.where(np.isnan(Y))[0]
#             index_nan = list(set(index_nanx).union(set(index_nany)))
#             X = np.delete(X, index_nan, axis=0)
#             Y = np.delete(Y, index_nan, axis=0)
#             if (train_num == 0):
#                 Train_Data_X = X
#                 Train_Data_Y = Y
#                 train_num = train_num + 1
#             else:
#                 Train_Data_X = np.concatenate((Train_Data_X, X))
#                 Train_Data_Y = np.concatenate((Train_Data_Y, Y))
#         return  Train_Data_X,Train_Data_Y
#
#     def setscaler(self,metriclist):
#         """
#         根据metrics和premetrics设置归一化标准
#         :return: data和targets的归一化
#         """
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         scalerfitdata = []
#         mindata = []
#         maxdata = []
#         for metric in metriclist:
#             mindata.append(model_setting.METRIC_MIN_MAX.get(metric).get('min'))
#             maxdata.append(model_setting.METRIC_MIN_MAX.get(metric).get('max'))
#         scalerfitdata.append(mindata)
#         scalerfitdata.append(maxdata)
#         scaler.fit(scalerfitdata)
#         return scaler
#
#     def create_train_data(self):
#         """
#         将每个nodename中进行lstm 训练阶段输入和输出数据进行合并
#         :return:
#
#         """
#         scalerdata = self.setscaler(self.metrics)
#         scalertargets = self.setscaler(self.premetrics)
#         Train_Data_X ,Train_Data_Y = self.create_train_data_onenode(self.nodelists[0],scalerdata,scalertargets)
#         if(len(self.nodelists) > 0):
#             for nodename in self.nodelists[1:]:
#                 X,Y = self.create_train_data_onenode(nodename,scalerdata,scalertargets)
#                 Train_Data_X = np.concatenate((Train_Data_X, X))
#                 Train_Data_Y = np.concatenate((Train_Data_Y, Y))
#         return Train_Data_X, Train_Data_Y
#
#
#     def fit(self,model,Train_data_X,Train_data_Y):
#         model.fit(Train_data_X,Train_data_Y,epochs=self.epochs, batch_size=self.batch_size,
#                             verbose=self.verbose,
#                             validation_split=self.validation_split, shuffle=self.shuffle)
#         return model
#
#     def buildmodel(self):
#         backend.clear_session()
#         model = Sequential()
#         model.add(LSTM(units=128, input_shape=(self.timesteps, len(self.metrics)),return_sequences=True))
#         model.add(LSTM(units=128))
#         model.add(Dense(len(self.premetrics)))
#         adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#         model.compile(loss='mae', optimizer=adam, metrics=['mae','accuracy'])
#         #model.compile(loss='mape', optimizer=adam, metrics=['mape', 'accuracy'])
#         return model
#
#     def train(self,Train_data_X,Train_data_Y):
#         reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
#         model = self.buildmodel()
#         Train_data_X,Test_data_X,Train_data_Y,Test_data_Y = train_test_split(
#             Train_data_X,Train_data_Y,test_size=self.validation_split,random_state=self.seed)
#
#         history = model.fit(Train_data_X, Train_data_Y, epochs=self.epochs, batch_size=self.batch_size,
#                             verbose=self.verbose,
#                             validation_data=(Test_data_X,Test_data_Y), shuffle=self.shuffle)
#                             # callbacks=[reduce_lr]
#         self.save(model)
#         pyplot.plot(history.history['loss'], label='train')
#         pyplot.plot(history.history['val_loss'], label='test')
#         pyplot.legend()
#         pyplot.show()
#         return model
#
#     def train_batch(self):
#         model = self.buildmodel()
#         return 0
#
#
#     def test(self,Test_data_X,Test_data_Y):
#         return 0
#
#     def create_test_data_onenode(self, nodename, scalerdata):
#         """
#         离线测试集的构建
#         对要进行预测的nodename进行预测阶段需要的输入和输出数据的构建
#         :param nodename:str 预测的node
#         :return:
#         """
#         test_num = 0
#         Test_Data_X = []
#         Test_Time = []
#         starttime = int(round(time.mktime(self.test_start.timetuple())) * 1000)
#         endtime = int(round(time.mktime(self.test_end.timetuple())) * 1000)
#         dfone = esinteracton.search_nodename_timestamp_dataframe(nodename=nodename, starttime=starttime,
#                                                                       endtime=endtime, metrics=self.metrics)
#         data = self.getTestData(dfone)
#         targets = self.getTestTargets(dfone)
#         Test_Time = self.getTestTime(dfone)
#         data = scalerdata.transform(data)
#         data_generator = TimeseriesGenerator(data, targets, length=self.timesteps, batch_size=128)
#         for i in range(len(data_generator)):
#             X, Y = data_generator[i]
#             if (test_num == 0):
#                 Test_Data_X = X
#                 test_num = test_num + 1
#             else:
#                 Test_Data_X = np.concatenate((Test_Data_X, X))
#         s=np.argwhere(np.isnan(Test_Data_X))
#         return Test_Data_X,Test_Time
#
#     def predict_test(self,model):
#         """
#         测试集上的预测
#         :param model:
#         :return:
#         """
#         scalerdata = self.setscaler(self.metrics)
#         scalertargets = self.setscaler(self.premetrics)
#         Test_Data_X,Test_Time = self.create_test_data_onenode(nodename = self.nodename,scalerdata= scalerdata)
#         yhat = model.predict(Test_Data_X)
#         yhat = scalertargets.inverse_transform(yhat)
#         predf = pd.DataFrame(yhat,columns=self.premetrics)
#         predf['timestamp'] = Test_Time
#         return predf
#
#     def create_test_online_data_onenode(self,nodename):
#         #在线的逻辑
#         t = self.test_start
#         # 设置queue1为空队列
#         #while(t<self.test_end):
#             #if (len(queue1)<self.timesteps):
#             #取t前timesteps*5min+3min个数据 es交互
#                 #if(len(数据)>timesteps):
#                    #取后len个数据
#                 #elif(==):
#                    #取数据
#                 #elif(<)
#                    #是否有时间戳间隔过大的，若有，前向补齐
#                    #否则取前timesteps*5min+8min到3min中的数据的后一个
#                 #将数据存入队列
#              #elif：
#                #删除对头，队尾加t-5min 至t之间的数据
#          #转化成Test_data_X,
#          #归一化，预测
#          #计算偏差
#          #if(偏差队列长度小于windows)：
#            #直接入
#          #else(删除再入)
#         return 0
#
#
#
#     # 保存为hdf5文件
#     def save(self, model):
#         """
#         保存训练好的模型
#         :param
#         model:train中训练好的模型
#         """
#         model.save(self.path)
#
#     def load(self):
#         """
#         从hdf5文件加载模型
#         :return: 加载的模型
#         """
#         backend.clear_session()
#         model = load_model(self.path)
#         return model
#
#     def getTrainData(self,dataframe):
#         data_df = dataframe[self.metrics]
#         return data_df.values
#
#     def getTrainTarget(self,dataframe):
#         target_df =  dataframe[self.premetrics]
#         return target_df.values
#
#     def getTestData(self,dataframe):
#         data_df = dataframe[self.metrics]
#         return data_df.values
#
#     def getTestTargets(self,dataframe):
#         data_df = dataframe[self.premetrics]
#         return data_df.values
#
#     def getTestTime(self,dataframe):
#         """
#         生成time的时间戳
#         :param dataframe:
#         :return:
#         """
#         time = dataframe['timestamp'].tolist()
#         time = time[self.timesteps:]
#         return time



class LSTM_mul_es_class(TrainModel):
    def __init__(self,nodelists=None,metrics=None,train_start=None,train_end=None,test_start=None,test_end=None,nodename=None,
                 modelname=None,premetrics=None,
                 timesteps=12,epochs=100, batch_size=128, verbose=1,validation_split=0.1,seed=42, shuffle=True):
        super(LSTM_mul_es_class,self).__init__(nodelists,metrics,train_start,train_end,test_start,test_end,nodename)
        self.premetrics = premetrics
        self.timesteps = int(timesteps)
        self.path = PATH+modelname+'.h5'
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.verbose = int(verbose)
        self.validation_split = float(validation_split)
        self.seed = int(seed)
        self.shuffle = shuffle

    def create_train_data_onenode(self,nodename,scalerdata,scalertargets):
        """
        对单个nodename进行model训练阶段需要的输入和输出数据的构建
        :param nodename:
        :return:
        """
        train_num = 0
        Train_Data_X = []
        Train_Data_Y = []
        starttime = int(round(time.mktime(self.train_start.timetuple()))*1000)
        endtime = int(round(time.mktime(self.train_end.timetuple()))*1000)
        dfone = esinteracton.search_nodename_timestamp_dataframe_miss(nodename=nodename,starttime=starttime,endtime=endtime,metrics=self.metrics)
        dfonet = esinteracton.search_nodename_timestamp_dataframe_miss(nodename=nodename,starttime=starttime,endtime=endtime,metrics=self.premetrics)
        data = self.getTrainData(dfone)
        targets = self.getTrainTarget(dfonet)
        data = scalerdata.transform(data)
        targets = scalertargets.transform(targets)
        data_generator = TimeseriesGenerator(data, targets, length=self.timesteps, batch_size=self.batch_size)
        for i in range(len(data_generator)):
            X, Y = data_generator[i]
            index_nanx = np.where(np.isnan(X))[0]
            index_nany = np.where(np.isnan(Y))[0]
            index_nan = list(set(index_nanx).union(set(index_nany)))
            X = np.delete(X, index_nan, axis=0)
            Y = np.delete(Y, index_nan, axis=0)
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
        scalerdata = self.setscaler(self.metrics)
        scalertargets = self.setscaler(self.premetrics)
        Train_Data_X ,Train_Data_Y = self.create_train_data_onenode(self.nodelists[0],scalerdata,scalertargets)
        for nodename in self.nodelists[1:]:
            X,Y = self.create_train_data_onenode(nodename,scalerdata,scalertargets)
            Train_Data_X = np.concatenate((Train_Data_X, X))
            Train_Data_Y = np.concatenate((Train_Data_Y, Y))
        return Train_Data_X, Train_Data_Y


    def fit(self,model,Train_data_X,Train_data_Y):
        model.fit(Train_data_X,Train_data_Y,epochs=self.epochs, batch_size=self.batch_size,
                            verbose=self.verbose,
                            validation_split=self.validation_split, shuffle=self.shuffle)
        return model

    def buildmodel(self):
        backend.clear_session()
        model = Sequential()
        model.add(LSTM(units=128, input_shape=(self.timesteps, len(self.metrics)),return_sequences=True))
        model.add(LSTM(units=128))
        model.add(Dense(len(self.premetrics)))
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='mae', optimizer=adam, metrics=['mae','accuracy'])
        #model.compile(loss='mape', optimizer=adam, metrics=['mape', 'accuracy'])
        return model

    def train(self,Train_data_X,Train_data_Y):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
        model = self.buildmodel()
        Train_data_X,Test_data_X,Train_data_Y,Test_data_Y = train_test_split(
            Train_data_X,Train_data_Y,test_size=self.validation_split,random_state=self.seed)
        history = model.fit(Train_data_X, Train_data_Y, epochs=self.epochs, batch_size=self.batch_size,
                            verbose=self.verbose,
                            validation_data=(Test_data_X,Test_data_Y), shuffle=self.shuffle)
                            # callbacks=[reduce_lr]
        self.save(model)
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
        return model

    def train_batch(self):
        model = self.buildmodel()
        return 0


    def test(self,Test_data_X,Test_data_Y):
        return 0

    def create_test_data_onenode(self, nodename, scalerdata):
        """
        离线测试集的构建
        对要进行预测的nodename进行预测阶段需要的输入和输出数据的构建
        :param nodename:str 预测的node
        :return:
        """
        test_num = 0
        Test_Data_X = []
        Test_Time = []
        starttime = int(round(time.mktime(self.test_start.timetuple())) * 1000)
        endtime = int(round(time.mktime(self.test_end.timetuple())) * 1000)
        dfone = esinteracton.search_nodename_timestamp_dataframe(nodename=nodename, starttime=starttime,
                                                                      endtime=endtime, metrics=self.metrics)
        data = self.getTestData(dfone)
        targets = self.getTestTargets(dfone)
        Test_Time = self.getTestTime(dfone)
        data = scalerdata.transform(data)
        data_generator = TimeseriesGenerator(data, targets, length=self.timesteps, batch_size=128)
        for i in range(len(data_generator)):
            X, Y = data_generator[i]
            if (test_num == 0):
                Test_Data_X = X
                test_num = test_num + 1
            else:
                Test_Data_X = np.concatenate((Test_Data_X, X))
        s=np.argwhere(np.isnan(Test_Data_X))
        return Test_Data_X,Test_Time

    def predict_test(self,model):
        """
        测试集上的预测
        :param model:
        :return:
        """
        scalerdata = self.setscaler(self.metrics)
        scalertargets = self.setscaler(self.premetrics)
        Test_Data_X,Test_Time = self.create_test_data_onenode(nodename = self.nodename,scalerdata= scalerdata)
        yhat = model.predict(Test_Data_X)
        yhat = scalertargets.inverse_transform(yhat)
        predf = pd.DataFrame(yhat,columns=self.premetrics)
        predf['timestamp'] = Test_Time
        return predf

    def create_test_online_data(self,nodename):
        #在线的逻辑
        t = self.test_start
        # 设置queue1为空队列
        #while(t<self.test_end):
            #if (len(queue1)<self.timesteps):
            #取t前timesteps*5min+3min个数据 es交互
                #if(len(数据)>timesteps):
                   #取后len个数据
                #elif(==):
                   #取数据
                #elif(<)
                   #是否有时间戳间隔过大的，若有，前向补齐
                   #否则取前timesteps*5min+8min到3min中的数据的后一个
                #将数据存入队列
             #elif：
               #删除对头，队尾加t-5min 至t之间的数据
         #转化成Test_data_X,
         #归一化，预测
         #计算偏差
         #if(偏差队列长度小于windows)：
           #直接入
         #else(删除再入)
        return 0



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

    def getTrainTarget(self,dataframe):
        target_df =  dataframe[self.premetrics]
        return target_df.values

    def getTestData(self,dataframe):
        data_df = dataframe[self.metrics]
        return data_df.values

    def getTestTargets(self,dataframe):
        data_df = dataframe[self.premetrics]
        return data_df.values

    def getTestTime(self,dataframe):
        """
        生成time的时间戳
        :param dataframe:
        :return:
        """
        time = dataframe['timestamp'].tolist()
        time = time[self.timesteps:]
        return time


