import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from algorithm import model_setting

class TrainModel:
    def __init__(self,nodelists=None,metrics=None,train_start=None,train_end=None,test_start=None,test_end=None,nodename=None,modelname=None):
        self.nodelists = nodelists
        self.nodename = nodename
        self.metrics = metrics
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.modelname = modelname

    def setscaler(self,metriclist):
        """
        根据metriclist设置归一化标准
        :return: data和targets的归一化
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scalerfitdata = []
        mindata = []
        maxdata = []
        for metric in metriclist:
            mindata.append(model_setting.METRIC_MIN_MAX.get(metric).get('min'))
            maxdata.append(model_setting.METRIC_MIN_MAX.get(metric).get('max'))
        scalerfitdata.append(mindata)
        scalerfitdata.append(maxdata)
        scaler.fit(scalerfitdata)
        return scaler

    def create_train_data_onenode(self,**kwargs):
        """
        对单个nodename进行model训练阶段需要的输入和输出数据的构建
        :param nodename:
        :return:
        """
        pass

    def create_test_data_onenode(self, **kwargs):
       pass


    def create_train_data(self):
        pass

    def create_test_date(self):
        pass

    def buildmodel(self,**kwargs):
        pass

    def train(self,**kwargs):
       pass

    def train_batch(self):
       pass

    def test(self,**kwargs):
       pass

    def predict_test(self,model):
        """
        测试集上的预测
        :param model:
        :return:
        """
        pass


    def save(self, model):
        """
        保存训练好的模型
        """
        pass

    def load(self):
        """
        加载模型
        """
        pass

    def getTrainData(self,dataframe):
        data_df = dataframe[self.metrics]
        return data_df.values
