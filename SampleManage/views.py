from django.shortcuts import render
from django.http import HttpResponse
import Common.sql_setting
import pandas as  pd
from sqlalchemy import create_engine
from django.contrib.auth.decorators import  login_required
import json
from .forms import PriSampleForm,VisualizeForm,VisualizeESForm,PredictVisForm,TrainForm,MulvisualizeForm,PredictVisesForm,TagForm,DetectVisesForm
from .modelsql import modelsql
from algorithm.univariate_predictor.lstm import LSTM_mul_es_class
from algorithm.detection.Iforest import Iforest_class
from algorithm.model_setting import  Metrics
from algorithm.univariate_predictor.ewma import EWMA_class
from algorithm.univariate_predictor.swa import SWA_class
from algorithm.univariate_predictor.linearregression import  LR_class
from algorithm.univariate_predictor.arima import Arima_class
from algorithm.univariate_predictor.evaluation import evaluation
from algorithm.univariate_predictor import evaluation
import numpy as np
from .models import PriSample,ProSample,PredictMethod,DetectMethod
from algorithm.detection.Nsigma import Nsigma
from algorithm.detection.Qfunction import Qfunction
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from esconn import esinteracton
import datetime,time
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
        file_data.replace('Array', np.nan, inplace=True)
        file_data = file_data.astype(object).where(pd.notnull(file_data), None)
        print(file_data)
        s = file_data.columns.values
        print(s)
        print(type(s))
        if not ((s[0] == 'nodename')&(s[1] == 'timestamp')):
            return HttpResponse("文件输入格式错误")
        file_data['timestamp'] = pd.to_datetime(file_data['timestamp'],errors='coerce',exact=False,infer_datetime_format=False)
        if (file_data['timestamp'].isnull().any()):
            print(file_data['timestamp'][0])
            return HttpResponse("存在错误的时间戳")
        if(file_data[file_data.duplicated('timestamp')].empty == False):
            return HttpResponse("存在重复的时间戳")
        print(type(file_data['timestamp'][0]))
        database_url = Common.sql_setting.DATABASE_URL
        engine = create_engine(database_url,echo=False)
        entries = []
        repeat = []
        testnum = 0
        for e in file_data.T.to_dict().values():
            if(PriSample.objects.filter(nodename=e['nodename'],timestamp=e['timestamp']).exists()):
                repeat.append([e['nodename'],e['timestamp']])
            else :
                testnum = testnum + 1
                entries.append(PriSample(**e))
        print(testnum)
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
    anomalylist = []
    nodename = ''
    metric = ''
    obj = VisualizeForm()
    if request.method == "POST":
        obj = VisualizeForm(request.POST,request.FILES)
        if obj.is_valid():
            nodename = obj.clean()['nodenames']
            metric = obj.clean()['metrics']
            modeldata = modelsql(PriSample)
            list = modeldata.select_visualize_data(nodename=nodename,metric=metric)
            anomalylist = modeldata.select_visualize_anomaly(nodename=nodename,metric=metric)
        else:
            errors = obj.errors
        return render(request,'SampleManage/visualize.html',{'list':json.dumps(list),'anomalylist':json.dumps(anomalylist),'nodename':nodename,'metric':metric,'form':obj,})
    else:
        return render(request,'SampleManage/visualize.html',{'list':json.dumps(list),'anomalylist':json.dumps(anomalylist),'nodename':nodename,'metric':metric,'form':obj,})

def visualizees(request):
    edatasets = []
    nodename = ''
    metric = ''
    obj = VisualizeESForm()
    if request.method == "POST":
        obj = VisualizeESForm(request.POST,request.FILES)
        if obj.is_valid():
            nodename = obj.clean()['nodenames']
            metric = obj.clean()['metrics']
            start = obj.clean()['start']
            end = obj.clean()['end']
            metriclist = [metric]
            start = int(round(time.mktime(start.timetuple()))*1000)
            end = int(round(time.mktime(end.timetuple())) * 1000)
            source = esinteracton.search_nodename_timestamp_dataframe(nodename=nodename,starttime=start,endtime=end,metrics=metriclist)
            edatasets = [['timestamp', metric]]
            for i in range(len(source)):
                edataset = []
                edataset.append(source['timestamp'][i].strftime('%Y-%m-%d %H:%M:%S'))
                edataset.append(source[metric][i])
                edatasets.append(edataset)
        else:
            errors = obj.errors
        return render(request,'SampleManage/visualizees.html',{'list':json.dumps(edatasets),'nodename':nodename,'metric':metric,'form':obj,})
    else:
        return render(request,'SampleManage/visualizees.html',{'list':json.dumps(edatasets),'nodename':nodename,'metric':metric,'form':obj,})

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

def retages(request,pathtest):
    # nodename = request.POST.get('nodename')
    list = str.split(pathtest,'/')
    nodename = list[0]
    timestamp = list[1]
    timestamp_ = datetime.datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S')
    form = TagForm()
    if(request.method == 'GET'):
        timestamp = int(round(time.mktime(timestamp_.timetuple())) * 1000)
        result = esinteracton.search_isexist(index="mlganglia_agg",
                                             query_json=esinteracton.search_nodename_timestamp_mantag_queryjson(nodename=nodename,timestamp=timestamp))
        if (result == 0):
            form = TagForm(initial={'nodename':nodename,'timestamp':timestamp_,'mantag':'0'})
        elif(result == 1):
            form = TagForm(initial={'nodename':nodename,'timestamp':timestamp_,'mantag':'1'})
        context = {'pathtest':pathtest,'form':form}
        return render(request,'SampleManage/retages.html',context)
    if(request.method == 'POST'):
        obj =TagForm(request.POST, request.FILES)
        if obj.is_valid():
            nodename = obj.clean()['nodename']
            timestamp_ = obj.clean()['timestamp']
            timestamp = int(round(time.mktime(timestamp_.timetuple())) * 1000)
            mantag = obj.clean()['mantag']
            result = esinteracton.search_isexist(index="mlganglia_agg",
                                             query_json=esinteracton.search_nodename_timestamp_mantag_queryjson(nodename=nodename,timestamp=timestamp))

            # 搜索结果如果和mantag一致，则不做变化
            if(mantag == str(result)):
                print(nodename, timestamp_, mantag)
                pass
            else:
                isexist = esinteracton.search_isexist(index="mlganglia_agg",
                                                      query_json=esinteracton.is_search_nodename_timestamp_queryjson(nodename=nodename,timestamp=timestamp))
                if(isexist == 0):
                    #如果mlganglia_agg不存在该document，则从search_ganglia查找数据新建
                    mdata = esinteracton.search_bulk(index="search_ganglia",
                                                     query_json=esinteracton.search_nodename_timestamp_queryjson(nodename=nodename,starttime=timestamp,endtime=timestamp,metrics=['time']))
                    esinteracton.insert_bulk(index="mlganglia_agg",mdata=mdata)
                else:
                    pass
                #进行更新操作
                mdata = esinteracton.search_bulk(index="mlganglia_agg",
                                                 query_json=esinteracton.search_nodename_timestamp_queryjson(
                                                     nodename=nodename, starttime=timestamp, endtime=timestamp,
                                                     metrics=['time']))
                if(mantag == '0'):
                    esinteracton.upadte_bulk(index="mlganglia_agg",mdata=mdata,docs={'tags':''})
                else:
                    esinteracton.upadte_bulk(index="mlganglia_agg",mdata=mdata,docs={'tags': 'mantag'})
            print(nodename,timestamp_,mantag)
        return HttpResponse('保存成功')




def predict(request):
    x = []
    y = []
    z = []
    e = []
    anomalypoints = []
    nodename = ''
    metric = ''
    obj = PredictVisForm()
    content = {
               'nodename':nodename,
               'form':obj,
               'x':json.dumps(x),
               'y':json.dumps(y),
               'z':json.dumps(z),
               'e':json.dumps(e),
               'anomalypoints':json.dumps(anomalypoints),
               'metric':metric,
               }
    if(request.method == 'POST'):
        obj = PredictVisForm(request.POST, request.FILES)
        if obj.is_valid():
            nodename = obj.clean()['nodenames']
            metric = obj.clean()['metrics']
            teststart = obj.clean()['teststart']
            testend = obj.clean()['testend']
            # start = '2018-12-01 00:00:00'
            # end = '2018-12-31  23:59:59'
            # starttime = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            # endtime = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
            listdf = modelsql(PriSample).select_data(nodename=nodename,start='2018-11-01 00:00:00',end='2018-12-31 00:00:00')
            prelistdf = modelsql(ProSample).select_data(nodename=nodename,start='2018-11-01 00:00:00',end='2018-12-31 00:00:00' )
            #选择预测方法
            # ewma
            e = EWMA_class(df=prelistdf,alpha=0.3,start=teststart,end=testend,windows=1,metric=metric)
            predf = e.predict()

            #swa
            # e = SWA_class(df=prelistdf,start=teststart,end=testend,windows=1,metric=metric)
            # predf = e.predict()

            #e = LSTM_class(df=prelistdf,start=start,end=end,train_start='2018-11-01 00:00:00',train_end='2018-11-30 23:59:59',metric=metric,timesteps=12)


            # e = LSTM_mulnodename_class(df=prelistdf,metric=metric,test_start=teststart,test_end=testend)
            # model = e.load()
            # predf = e.predict(model)

            x_ = listdf[(listdf['timestamp']>=teststart)&(listdf['timestamp']<=testend)]['timestamp'].tolist()
            for i in range(len(x_)):
                x.append(x_[i].strftime('%Y-%m-%d %H:%M:%S'))
            y = listdf[(listdf['timestamp']>=teststart)&(listdf['timestamp']<=testend)][metric].tolist()
            z = predf['predictdata'].tolist()

            e = np.fabs(np.array(y) - np.array(z))
            e = e.tolist()
            detemodel = Nsigma(N=10,series=e)
            anomalylist = Nsigma.anomalylocation(detemodel)

            anomaly_x = [x[l] for l in anomalylist]
            anomaly_y = [y[l] for l in anomalylist]
            # anomalymarkpoints = []
            anomalypoints = []
            for i  in range(len(anomaly_x)):
                dic = {'coord':[anomaly_x[i],anomaly_y[i]],'value':anomaly_y[i]}
                # anomalymarkpoints.append(dic)
                # print(anomaly_x[i],anomaly_y[i])
                anomalypoints.append([anomaly_x[i],anomaly_y[i]])
                PriSample.objects.filter(nodename=nodename,timestamp=anomaly_x[i]).update(label=-1)

            eva = evaluation.evaluation(y,z)
            eva.deleteNan()
            evamethod = {}
            evamethod['me'] = eva.ME()
            evamethod['mae'] = eva.MAE()
            evamethod['rmse'] = eva.RMSE()
            evamethod['mpe'] = eva.MPE()
            evamethod['mape'] = eva.MAPE()
            content = {'nodename':nodename,
                       'metric': metric,
                       'form':obj,
                       'x':json.dumps(x),
                       'y':json.dumps(y),
                       'z':json.dumps(z),
                       'e':json.dumps(e),
                       # 'anomalymarkpoints':json.dumps(anomalymarkpoints),
                       'anomalypoints':json.dumps(anomalypoints),
                       'evamethod':evamethod,
                       }
        else:
            errors = obj.errors
    return render(request,'SampleManage/predict_visualize.html',content)


def train(request):
    context = {}
    form = TrainForm()
    context = {'form':form}
    if (request.method == 'POST'):
        form = TrainForm(request.POST, request.FILES)
        if(form.is_valid()):
            nodelists = form.clean()['nodename']
            metrics = form.clean()['metrics']
            premetrics = form.clean()['premetrics']
            trainstart = form.clean()['trainstart']
            trainend = form.clean()['trainend']
            alogrithm = form.clean()['algorithm']
            parameters = form.clean()['parameters']
            print(trainstart,type(trainstart))
            # if (alogrithm == '1'):
            #     df = modelsql(ProSample).select_data_nodelists(nodelists=nodelists, start=trainstart, end=trainend)
            #     modelclass = LSTM_mul_class(df=df, metrics=metrics, premetrics=premetrics,nodelists=nodelists)
            #     Train_data_X, Train_data_Y = modelclass.create_train_data()
            #     model = modelclass.train(Train_data_X=Train_data_X,Train_data_Y=Train_data_Y)
            #     # modelclass.save(model)
            #
            # elif(alogrithm == '2'):
            #     df = modelsql(ProSample).select_data_nodelists(nodelists=nodelists, start=trainstart, end=trainend)
            #     modelclass = Iforest(df=df,metrics=metrics,nodelists=nodelists)
            #     Train_Data = modelclass.create_train_data()
            #     model = modelclass.train(Train_Data)

            if(alogrithm == 'LSTM'):
                modelname = time.strftime("%Y%m%d%H%M%S", time.localtime())
                thiskw = {"metrics": metrics, "premetrics": premetrics, "nodelists": nodelists,
                          "train_start": trainstart, "train_end": trainend, 'modelname': modelname}
                if(parameters == 'default'):
                    # modelclass = LSTM_mul_es_class(metrics=metrics, premetrics=premetrics, nodelists=nodelists,train_start=trainstart,train_end=trainend,modelname=modelname)
                    modelclass = LSTM_mul_es_class(**thiskw)
                else:
                    parameterlist = parameters.split(',')
                    for param in parameterlist:
                         param = param.split(':')
                         thiskw[param[0]] = param[1]
                    modelclass = LSTM_mul_es_class(**thiskw)
                Train_data_X, Train_data_Y = modelclass.create_train_data()
                model = modelclass.train(Train_data_X=Train_data_X, Train_data_Y=Train_data_Y)
                thiskw['train_start'] = thiskw['train_start'].strftime("%Y%m%d%H%M%S")
                thiskw['train_end'] = thiskw['train_end'].strftime("%Y%m%d%H%M%S")
                param = json.dumps(thiskw)
                pre = PredictMethod(methodname='LSTM+'+thiskw['modelname'],algorithm='LSTM',paramvalue=param,path=thiskw['modelname'] )
                pre.save()
                # modelclass.save(model)

                # modelclass = LSTM_mul_es_class(metrics=metrics, premetrics=premetrics, nodelists=nodelists,
                #                 #                                train_start=trainstart, train_end=trainend)
                #                 # nodename = modelclass.nodelists[0]
                #                 # Train_data_X, Train_data_Y = modelclass.create_train_data_onenode(nodename)
                #                 # model = modelclass.train(Train_data_X=Train_data_X,Train_data_Y=Train_data_Y)
                #                 # for nodename in modelclass.nodelists[1:]:
                #                 #      Train_data_X,Train_data_Y = modelclass.create_train_data_onenode(nodename)
                #                 #      model = modelclass.fit(model=model,Train_data_X=Train_data_X, Train_data_Y=Train_data_Y)
                #                 # modelclass.save(model)

            if(alogrithm == 'Iforest'):
                modelname = time.strftime("%Y%m%d%H%M%S", time.localtime())
                thiskw = {"metrics": metrics, "nodelists": nodelists,
                          "train_start": trainstart, "train_end": trainend, 'modelname': modelname}
                if (parameters == 'default'):
                    modelclass = Iforest_class(**thiskw)
                else:
                    parameterlist = parameters.split(',')
                    for param in parameterlist:
                        param = param.split(':')
                        thiskw[param[0]] = param[1]
                    modelclass = Iforest_class(**thiskw)
                Train_data = modelclass.create_train_data()
                model = modelclass.train(Train_data=Train_data)
                thiskw['train_start'] = thiskw['train_start'].strftime("%Y%m%d%H%M%S")
                thiskw['train_end'] = thiskw['train_end'].strftime("%Y%m%d%H%M%S")
                param = json.dumps(thiskw)
                pre = DetectMethod(methodname='Iforest+' + thiskw['modelname'], algorithm='Iforest', paramvalue=param,
                                    path=thiskw['modelname'])
                pre.save()




    return render(request,'SampleManage/train.html',context)


def mulvisualize(request):

    nodelists = []
    points = []
    anomalypoints = []
    form = MulvisualizeForm()
    context = {'nodelists': json.dumps(nodelists), 'points': json.dumps(points),
               'anomalypoints': json.dumps(anomalypoints),'form':form}

    if (request.method == 'POST'):
        form = MulvisualizeForm(request.POST,request.FILES)
        if(form.is_valid()):
            nodelists = form.clean()['nodename']
            metrics = form.clean()['metric']
        # nodelists = ['alimds.ihep.ac.cn']
        # # bytes_in_value, bytes_out_value,\
        # # cpu_idle_value, disk_free_value,\
        # # load_fifteen_value, load_five_value,\
        # # load_one_value, mem_buffers_value,\
        # # mem_cached_value, mem_free_value,\
        # # pkts_in_value, pkts_out_value,\
        # # proc_run_value, proc_total_value, \
        # # swap_free_value\
        # metrics = ['nodename', 'timestamp', 'label',
        #            'bytes_in_value', 'bytes_out_value', 'cpu_idle_value',
        #            'disk_free_value', 'load_fifteen_value', 'load_five_value',
        #            'load_one_value', 'mem_buffers_value', 'mem_cached_value',
        #            'mem_free_value', 'pkts_in_value', 'pkts_out_value',
        #            'proc_run_value', 'proc_total_value', 'swap_free_value'
        #            ]
            metrics.insert(0,'label')
            metrics.insert(0,'timestamp')
            metrics.insert(0,'nodename')
            print(metrics)
            df = modelsql(PriSample).select_data_nodelists(nodelists)
            df.dropna(axis=0,how='any',inplace=True)
            ds = df[metrics]
            lists = ds.values
            metricsvalue_list = lists[:1000,3:]
            label = lists[:1000,2]
            label = np.array(label)
            label = label.reshape(-1)
            metricsvalue_list = np.array(metricsvalue_list)

            #归一化
            scaler = MinMaxScaler(feature_range=(0, 1))
            metricsvalue_list = scaler.fit_transform(metricsvalue_list)

            tsne = TSNE(n_components=2,verbose=1)
            X_embedded = tsne.fit_transform(metricsvalue_list)
            print(type(X_embedded))
            data = lists[:1000,:3]
            data = np.concatenate((data,X_embedded),axis=1)
            decrease_dimension_data = pd.DataFrame(data,columns=['nodename','timestamp','dimen1','dimen2'])
            print(decrease_dimension_data.head())
            points = []
            anomalypoints = []
            for i in range(len(label)):
                if (label[i] == -1):
                    anomalypoints.append(X_embedded[i].tolist())
                else:
                    points.append(X_embedded[i].tolist())
            print(X_embedded[0])
            print(X_embedded.shape)
            context = {'nodelists':json.dumps(nodelists),'points':json.dumps(points),'anomalypoints':json.dumps(anomalypoints),'form':form}
    return render(request,'SampleManage/mulvisualize.html',context)


def predictes(request):
    x = []
    y = []
    z = []
    nodename = ''
    metric = ''
    obj = PredictVisesForm()
    content = {
        'nodename': nodename,
        'form': obj,
        'x': json.dumps(x),
        'y': json.dumps(y),
        'z': json.dumps(z),
        'metric': metric,
    }
    if (request.method == 'POST'):
        obj = PredictVisesForm(request.POST, request.FILES)
        if obj.is_valid():
            nodename = obj.clean()['nodename']
            teststart = obj.clean()['predictstart']
            testend = obj.clean()['predictend']
            metric = obj.clean()['metric']
            predictmethod = obj.clean()['predictmethod']

            #预测方式为LSTM
            if('LSTM' in predictmethod):
                preparam = PredictMethod.objects.filter(methodname=predictmethod).values('paramvalue')
                preparam = preparam[0]['paramvalue']
                preparam = json.loads(preparam)
                thiskw = {'nodename': nodename, 'test_start': teststart, 'test_end': testend}
                thiskw['metrics'] = preparam['metrics']
                thiskw['premetrics'] = preparam['premetrics']
                thiskw['modelname'] = preparam['modelname']
                metrics = preparam['metrics']
                premetric = preparam['premetrics']
                if ('timesteps' in preparam):
                    thiskw['timesteps'] = preparam['timesteps']
                else:
                    pass
                print(type(preparam), type(preparam['metrics']), preparam['metrics'])
                e = LSTM_mul_es_class(**thiskw)
                # e = LSTM_mul_es_class(nodename=nodename,metrics=metrics,premetrics=premetric,test_start=teststart,test_end=testend,modelname='my_mul_es_model_all_1-3yue_12_2lstm_all')
                model = e.load()
                predf = e.predict_test(model)

            elif (predictmethod == 'MA'):
                e = SWA_class(nodename=nodename,metric=metric,start=teststart,end=testend)
                e.createdata()
                predf = e.predict()

            elif(predictmethod == 'EWMA'):
                e = EWMA_class(nodename=nodename, metric=metric, start=teststart,end=testend)
                e.createdata()
                predf = e.predict()

            elif(predictmethod == 'LR'):
                e = LR_class(nodename=nodename,metric=metric,start=teststart,end=testend)
                e.createdata()
                predf = e.predict()

            elif(predictmethod == 'ARIMA'):
                e = Arima_class(nodename=nodename,metric=metric,start=teststart,end=testend)
                e.createdata()
                predf = e.predict()

            x_ = predf['timestamp'].tolist()
            visteststart = x_[0]
            visteststart = int(round(time.mktime(visteststart.timetuple())) * 1000)
            vistestend = x_[-1]
            vistestend = int(round(time.mktime(vistestend.timetuple())) * 1000)

            for i in range(len(x_)):
                x.append(x_[i].strftime('%Y-%m-%d %H:%M:%S'))
            trudf = esinteracton.search_nodename_timestamp_dataframe(nodename=nodename, starttime=visteststart,
                                                                     endtime=vistestend, metrics=Metrics)
            y = trudf[metric].tolist()
            z = predf[metric].tolist()
            eva = evaluation.evaluation(y, z)
            eva.deleteNan()
            evamethod = {}
            evamethod['me'] = eva.ME()
            evamethod['mae'] = eva.MAE()
            evamethod['rmse'] = eva.RMSE()
            evamethod['mpe'] = eva.MPE()
            evamethod['mape'] = eva.MAPE()
            content = {'nodename': nodename,
                       'metric': metric,
                       'form': obj,
                       'x': json.dumps(x),
                       'y': json.dumps(y),
                       'z': json.dumps(z),
                       'evamethod': evamethod,
                       }
        else:
            errors = obj.errors
    return render(request, 'SampleManage/predict_visualizees.html', content)


def detectes(request):
    x = []
    y = []
    z = []
    e = []
    anomalypoints = []
    anomalyscore = []
    nodename = ''
    metric = ''
    obj = DetectVisesForm()
    content = {
               'nodename':nodename,
               'form':obj,
               'x':json.dumps(x),
               'y':json.dumps(y),
               'z':json.dumps(z),
               'e':json.dumps(e),
               'anomalypoints':json.dumps(anomalypoints),
               'anomalyscore':json.dumps(anomalyscore),
               'metric':metric,
               }
    if(request.method == 'POST'):
        obj = DetectVisesForm(request.POST, request.FILES)
        if obj.is_valid():
            nodename = obj.clean()['nodename']
            teststart = obj.clean()['detectstart']
            testend = obj.clean()['detectend']
            metric = obj.clean()['metric']
            predictmethod  = obj.clean()['predictmethod']

            if(predictmethod == 'None'):
                detectmethod = obj.clean()['detectmethod']
                if('Iforest' in detectmethod):
                    detectparam = DetectMethod.objects.filter(methodname=detectmethod).values('paramvalue')
                    detectparam = detectparam[0]['paramvalue']
                    detectparam = json.loads(detectparam)
                    thiskw = {'nodename':nodename, 'test_start':teststart,'test_end':testend}
                    thiskw['metrics'] = detectparam['metrics']
                    thiskw['modelname'] = detectparam['modelname']
                    detclass = Iforest_class(**thiskw)
                    model = detclass.load()
                    detdf = detclass.predict_test(model=model)
                    anomalyscore = detclass.anomalyscore(model=model)
                    anomalylist = detclass.anomalylocation(model=model)
                    x_ = detdf['timestamp'].tolist()
                    for i in range(len(x_)):
                        x.append(x_[i].strftime('%Y-%m-%d %H:%M:%S'))
                    visteststart = int(round(time.mktime(teststart.timetuple())) * 1000)
                    vistestend = int(round(time.mktime(testend.timetuple())) * 1000)
                    trudf = esinteracton.search_nodename_timestamp_dataframe(nodename=nodename, starttime=visteststart,
                                                                             endtime=vistestend, metrics=Metrics)
                    y = trudf[metric].tolist()
                    z = ['-']*len(y)
                    e = ['-']*len(y)


            elif('LSTM' in predictmethod):
                preparam = PredictMethod.objects.filter(methodname=predictmethod).values('paramvalue')
                preparam = preparam[0]['paramvalue']
                preparam = json.loads(preparam)
                thiskw = {'nodename':nodename, 'test_start':teststart, 'test_end':testend}
                thiskw['metrics'] = preparam['metrics']
                thiskw['premetrics'] = preparam['premetrics']
                thiskw['modelname'] = preparam['modelname']
                metrics = preparam['metrics']
                premetric = preparam['premetrics']
                if('timesteps' in preparam):
                    thiskw['timesteps'] = preparam['timesteps']
                else:
                    pass
                print(type(preparam),type(preparam['metrics']),preparam['metrics'])
                preclass = LSTM_mul_es_class(**thiskw)
                #preclass = LSTM_mul_es_class(nodename=nodename,metrics=metrics,premetrics=premetric,test_start=teststart,test_end=testend,modelname='my_mul_es_model_all_1-3yue_12_2lstm_all')
                model = preclass.load()
                predf = preclass.predict_test(model)
                x_ = predf['timestamp'].tolist()
                visteststart = x_[0]
                visteststart = int(round(time.mktime(visteststart.timetuple()))*1000)
                vistestend = x_[-1]
                vistestend = int(round(time.mktime(vistestend.timetuple()))*1000)

                for i in range(len(x_)):
                    x.append(x_[i].strftime('%Y-%m-%d %H:%M:%S'))
                trudf = esinteracton.search_nodename_timestamp_dataframe(nodename=nodename,starttime=visteststart,endtime=vistestend,metrics=Metrics)
                y = trudf[metric].tolist()


                # # Nsigma 单维度的Nsigma
                # detemodel = Nsigma(N=10, series=e)
                # anomalylist = Nsigma.anomalylocation(detemodel)

                # #Qfunction 单维度
                # qe = (np.array(y) - np.array(z)) / np.array(z)
                # detemodel = Qfunction(scorethreshold=0.99999,series=qe,windows=12*24*7)
                # anomalylist = Qfunction.anomalylocation(detemodel)

                #Qfunciton多维度1
                # onemetric_tru = trudf[premetric[0]].tolist()
                # onemetric_pre = predf[premetric[0]].tolist()
                # qe = (np.array(onemetric_tru) - np.array(onemetric_pre)) / np.array(onemetric_pre)
                # detemodel = Qfunction(scorethreshold=1, series=qe, windows=12 * 24 * 7)
                # anomalylist = Qfunction.anomalylocation(detemodel)
                # print(len(anomalylist))
                # for metric_ in premetric[1:]:
                #     onemetric_tru = trudf[metric_].tolist()
                #     onemetric_pre = predf[metric_].tolist()
                #     qe = (np.array(onemetric_tru) - np.array(onemetric_pre)) / np.array(onemetric_pre)
                #     detemodel = Qfunction(scorethreshold=1, series=qe, windows=12 * 24 * 7)
                #     anomalylist_ = Qfunction.anomalylocation(detemodel)
                #     print(len(anomalylist_))
                #     #anomalylist = list(set(anomalylist_).intersection(set(anomalylist)))
                #     anomalylist = list(set(anomalylist_).union(set(anomalylist)))
                #     print(len(anomalylist))


                #Qfuncyion多维度2
                trunp = []
                prenp = []
                for metric_ in premetric:
                    trunp.append(trudf[metric_].tolist())
                    prenp.append(predf[metric_].tolist())
                trunp = np.transpose(np.array(trunp))
                prenp = np.transpose(np.array(prenp))
                print(trunp.shape,prenp.shape)
                qe = []
                for i in range(trunp.shape[0]):
                    errors = np.linalg.norm(trunp[i] - prenp[i]) / np.linalg.norm(prenp[i])
                    if(np.linalg.norm(trunp[i])>=np.linalg.norm(trunp[i])):
                        qe.append(errors)
                    else:
                        qe.append(-errors)
                print(qe)
                detemodel = Qfunction(scorethreshold=1,series=qe,windows=12*24*7)
                anomalylist = Qfunction.anomalylocation(detemodel)
                print(len(anomalylist))
                if(metric in premetric):
                   z = predf[metric].tolist()
                else:
                   z = [0]*len(y)
                # e = np.fabs(np.array(y) - np.array(z))/np.array(y)
                e = np.fabs(np.array(y) - np.array(z))
                e = e.tolist()
            anomaly_x = [x[l] for l in anomalylist]
            anomaly_y = [y[l] for l in anomalylist]
            # anomalymarkpoints = []
            anomalypoints = []
            for i in range(len(anomaly_x)):
                dic = {'coord': [anomaly_x[i], anomaly_y[i]], 'value': anomaly_y[i]}
                anomalypoints.append([anomaly_x[i], anomaly_y[i]])
            content = {'nodename': nodename,
                       'metric': metric,
                       'form': obj,
                       'x': json.dumps(x),
                       'y': json.dumps(y),
                       'z': json.dumps(z),
                       'e': json.dumps(e),
                       'anomalypoints': json.dumps(anomalypoints),
                       'anomalyscore':json.dumps(anomalyscore),
                       }
        else:
            errors = obj.errors
    return render(request,'SampleManage/detect_visualizees.html',content)