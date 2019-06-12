import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
from os import listdir
import time

class Iforest:
    def __init__(self,df,nodelists,metrics):
        self.df = df
        self.nodelists = nodelists
        self.metrics = metrics

    def create_train_data(self):
        data_df = self.df[self.metrics]
        return data_df.values

    def train(self,Train_data):
        """
        训练模型
        :param
        Train_data:array(n_samples,n_features)训练数据
        :return:model
        """
        model = IsolationForest(n_estimators=100, max_samples=256, contamination=0.01, behaviour='new',verbose=1)
        model.fit(Train_data)
        return model
