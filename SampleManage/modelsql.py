from django.core import serializers
import datetime
import pandas as pd
from django.core import serializers
import json
class modelsql:

    def __init__(self,model):
        self.model = model

    #根据nodename和metric选择要可视化的数据
    def select_visualize_data(self,nodename,metric):
        # testlist = self.model.objects.filter(nodename = nodename)
        # testdata = serializers.serialize('json',testlist,fields=('nodename','timestamp',metric))
        lists = self.model.objects.filter(nodename = nodename).values('nodename','timestamp',metric)
        edatasets = [['timestamp',metric]]
        for i in range(len(lists)):
            edataset = []
            edataset.append(lists[i]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'))
            edataset.append(lists[i][metric])
            edatasets.append(edataset)
        return edatasets

    def select_visualize_anomaly(self,nodename,metric):
        lists = self.model.objects.filter(nodename=nodename,label=-1).values('nodename', 'timestamp', metric)
        edatasets = []
        for i in range(len(lists)):
            edataset = []
            edataset.append(lists[i]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'))
            edataset.append(lists[i][metric])
            edatasets.append(edataset)
        return edatasets

    def select_data(self,nodename,start,end):
        # starttime = datetime.datetime.strptime(start,'%Y-%m-%d %H:%M:%S')
        # endtime = datetime.datetime.strptime(end,'%Y-%m-%d %H:%M:%S')
        lists = list(self.model.objects.filter(timestamp__range = (start,end),nodename = nodename).values())
        df = pd.DataFrame(lists)
        return df

    def select_data_nodelists(self,nodelists,start=None,end=None):
        listall = []
        if ((start == None)|(end == None)):
            for nodename in nodelists:
                lists = list(self.model.objects.filter(nodename = nodename).values())
                listall.extend(lists)
            df = pd.DataFrame(listall)
        else:
            for nodename in nodelists:
                lists = list(self.model.objects.filter(nodename = nodename,timestamp__range = (start,end)).values())
                listall.extend(lists)
            df = pd.DataFrame(listall)
        return df


class Dedimensionsql:
    def save_to_Dedimension(self,name,method,nodelists,metrics,path):
        return 0


