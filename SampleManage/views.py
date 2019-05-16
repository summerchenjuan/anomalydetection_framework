from django.shortcuts import render
from django.contrib import  messages
from django.http import HttpResponse,HttpResponseRedirect
import os
import Common_Sql.sql_setting
import pandas as  pd
from sqlalchemy import create_engine
from django.contrib.auth.decorators import  login_required
from  django.template import loader
import json
from .forms import PriSampleForm
from .forms import VisualizeForm
from .modelsql import modelsql
from algorithm.univariate_predictor.ewma import ewma
from algorithm.univariate_predictor.lstm import LSTM_class
import datetime
from algorithm.evaluation.evaluation import ME,MAE,MAPE,MPE,RMSE
import numpy as np
from .models import PriSample
from django.core import serializers
# from .format import DecimalEncoder
# Create your views here.


def index(request):
    return render(request, 'SampleManage/index.html')

@login_required
def upload(request):
    if request.method == "GET":
        return render(request,"SampleManage/index.html")
    else:
        #获取上传的样本文件，并对其进行分析
        myFile = request.FILES.get("uploadfile",None)
        #若上传空文件，则..
        if not myFile:
            return HttpResponse("no files for upload!")
        #过滤文件类型
        if not myFile.name.endswith('.csv'):
            return HttpResponse("请选择.csv文件")
        #处理文件
        file_data = pd.read_csv(myFile)
        s = file_data.columns.values
        print(s)
        print(type(s))
        if not ((s[0] == 'nodename')&(s[1] == 'timestamp')):
            return HttpResponse("文件输入格式错误")
        file_data['timestamp'] = pd.to_datetime(file_data['timestamp'],errors='coerce',exact=False,infer_datetime_format=False)
        if (file_data['timestamp'].isnull().any()):
            print(file_data['timestamp'][0])
            return HttpResponse("存在错误的时间戳")
        print(type(file_data['timestamp'][0]))
        database_url = Common_Sql.sql_setting.DATABASE_URL
        engine = create_engine(database_url,echo=False)
        entries = []
        repeat = []
        for e in file_data.T.to_dict().values():
            if(PriSample.objects.filter(nodename=e['nodename'],timestamp=e['timestamp']).exists()):
                repeat.append([e['nodename'],e['timestamp']])
            else :
                entries.append(PriSample(**e))
                PriSample.objects.bulk_create(entries)

            # for e in ds.T.to_dict().values():
            #    PriSample.objects.get_or_create(metrics=e['metrics'],nodename=e['nodename'],timestamp=e['timestamp'],value=e['value'])

            #way2利用pandas to_sql批量导入数据
            # ds.to_sql("sample",con=engine,if_exists='replace')


        return render(request,'SampleManage/uploadsuccess.html',{'repeat':repeat})

def resultechart(request):
    return render(request,'SampleManage/resultechart.html')


def testform(request):
        return render(request,'SampleManage/testform.html')

#可视化元数据
def visualize(request):
    list = []
    nodename = 'alimds.ihep.ac.cn'
    metric = ''
    obj = VisualizeForm()
    if request.method == "POST":
        nodename = request.POST.get('nodename')
        # metric = request.POST.get('metric')
        obj = VisualizeForm(request.POST,request.FILES)
        if obj.is_valid():
            metric = obj.clean()['metrics']
            list = modelsql(PriSample).select_visualize_data(nodename=nodename,metric=metric)
        else:
            errors = obj.errors
        return render(request,'SampleManage/visualize.html',{'list':json.dumps(list),'nodename':nodename,'metric':metric,'form':obj,})
    else:
        return render(request,'SampleManage/visualize.html',{'list':json.dumps(list),'nodename':nodename,'metric':metric,'form':obj,})

#提供给数据人工打标的功能
def retag(request,pathtest):
    # nodename = request.POST.get('nodename')
    list = str.split(pathtest,'/')
    nodename = list[0]
    timestamp = list[1]
    timestamp = datetime.datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S')
    sample = PriSample.objects.get(nodename=nodename,timestamp=timestamp)
    if(request.method == 'GET'):
        form = PriSampleForm(instance=sample)
        context = {'pathtest':pathtest,'form':form}
        return render(request,'SampleManage/retag.html',context)
    if(request.method == 'POST'):
        form = PriSampleForm(request.POST,instance=sample)
        form.save()
        return HttpResponse('保存成功')

def savetag(request):
    testform = PriSampleForm(request.POST,request.FILES)
    print(testform)
    if(testform.is_valid()):
        s = testform.clean()
        print(s)
        sample =  PriSample.objects.get(nodename=s['nodename'],timestamp=s['timestamp'])
        sample.label = s['label']
        sample.save(commit=True)
        return HttpResponse('保存成功')
    else:
        return HttpResponse('保存失败')


def predict(request):
    x = []
    y = []
    z = []
    e = []
    nodename = 'alimds.ihep.ac.cn'
    obj = VisualizeForm()
    content = {'nodename':nodename,
                'form':obj,
                'x':json.dumps(x),
                'y':json.dumps(y),
                'z':json.dumps(z),
                'e':json.dumps(e),
               }
    if(request.method == 'POST'):
        obj = VisualizeForm(request.POST, request.FILES)
        if obj.is_valid():
            metric = obj.clean()['metrics']
            start = '2018-12-01 00:00:00'
            end = '2018-12-31 00:00:00'
            starttime = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            endtime = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
            listdf = modelsql(PriSample).select_data(nodename=nodename,start='2018-11-01 00:00:00',end='2018-12-31 00:00:00')

            #选择预测算法
            e = ewma(df=listdf,alpha=0.3,start=start,end=end,windows=1,metric=metric)
            #e = LSTM_class(df=listdf,start=start,end=end,train_start='2018-11-01 00:00:00',train_end='2018-11-30 23:59:59',metric=metric)
            predf = e.predict()
            x_ = listdf[(listdf['timestamp']>=start)&(listdf['timestamp']<=end)]['timestamp'].tolist()
            for i in range(len(x_)):
                x.append(x_[i].strftime('%Y-%m-%d %H:%M:%S'))
            y = listdf[(listdf['timestamp']>=start)&(listdf['timestamp']<=end)][metric].tolist()
            z = predf['predictdata'].tolist()
            e = np.fabs( np.array(y) - np.array(z))
            e = e.tolist()
            me = ME(y,z)
            mae = MAE(y,z)
            rmse = RMSE(y,z)
            mpe = MPE(y,z)
            mape = MAPE(y,z)
            content = {'nodename':nodename,'form':obj,'x':json.dumps(x),'y':json.dumps(y),'z':json.dumps(z),'e':json.dumps(e),'metric':metric,'me':me,'mae':mae,'rmse':rmse,'mpe':mpe,'mape':mape}
        else:
            errors = obj.errors
    return render(request,'SampleManage/predict_visualize.html',content)
