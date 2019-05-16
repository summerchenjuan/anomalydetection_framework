from .models import PriSample
from django.core import serializers
import datetime
import pandas as pd

class modelsql:

    def __init__(self,model):
        self.model = model

    #根据nodename和metric选择要可视化的数据
    def select_visualize_data(self,nodename,metric):
        lists = self.model.objects.filter(nodename = nodename).values('nodename','timestamp',metric)
        edatasets = [['timestamp',metric]]
        for i in range(len(lists)):
            edataset = []
            edataset.append(lists[i]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'))
            edataset.append(lists[i][metric])
            edatasets.append(edataset)
        return edatasets


    def select_data(self,nodename,start,end):
        starttime = datetime.datetime.strptime(start,'%Y-%m-%d %H:%M:%S')
        endtime = datetime.datetime.strptime(end,'%Y-%m-%d %H:%M:%S')
        lists = list(self.model.objects.filter(timestamp__range = (starttime,endtime),nodename = nodename).values())
        df = pd.DataFrame(lists)
        return df


