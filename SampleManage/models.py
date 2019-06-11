from django.db import models

# Create your models here.
#metrics
# bytes_in_value, bytes_out_value,\
# cpu_idle_value, disk_free_value,\
# load_fifteen_value, load_five_value,\
# load_one_value, mem_buffers_value,\
# mem_cached_value, mem_free_value,\
# pkts_in_value, pkts_out_value,\
# proc_run_value, proc_total_value, \
# swap_free_value\

#原始样本数据库
class PriSample(models.Model):
    nodename = models.CharField(max_length=100)
    timestamp = models.DateTimeField(max_length=100)
    bytes_in_value = models.FloatField(max_length=20,null=True,blank=True)
    bytes_out_value = models.FloatField(max_length=20,null=True,blank=True)
    cpu_idle_value = models.FloatField(max_length=20,null=True,blank=True)
    disk_free_value = models.FloatField(max_length=20,null=True,blank=True)
    load_fifteen_value = models.FloatField(max_length=20,null=True,blank=True)
    load_five_value = models.FloatField(max_length=20,null=True,blank=True)
    load_one_value = models.FloatField(max_length=20,null=True,blank=True)
    mem_buffers_value = models.FloatField(max_length=20,null=True,blank=True)
    mem_cached_value = models.FloatField(max_length=20,null=True,blank=True)
    mem_free_value = models.FloatField(max_length=20,null=True,blank=True)
    pkts_in_value = models.FloatField(max_length=20,null=True,blank=True)
    pkts_out_value = models.FloatField(max_length=20,null=True,blank=True)
    proc_run_value = models.FloatField(max_length=20,null=True,blank=True)
    proc_total_value = models.FloatField(max_length=20,null=True,blank=True)
    swap_free_value = models.FloatField(max_length=20,null=True,blank=True)
    STATUS_NORMAL = 1
    STATUS_ANOMALY = -1
    STATUS_UNKNOW = 0
    STATUS_ITEMS = (
        (STATUS_NORMAL, '正常'),
        (STATUS_ANOMALY, '异常'),
        (STATUS_UNKNOW, '未标记'),
    )
    label = models.IntegerField(default=STATUS_UNKNOW, choices=STATUS_ITEMS)

    class Meta:
        db_table = 'sampledb'
        ordering = ['nodename','timestamp']
        unique_together = [['nodename','timestamp']]

#预处理样本数据库
class ProSample(models.Model):
    nodename = models.CharField(max_length=100)
    timestamp = models.DateTimeField(max_length=100)
    bytes_in_value = models.FloatField(max_length=20,null=True,blank=True)
    bytes_out_value = models.FloatField(max_length=20,null=True,blank=True)
    cpu_idle_value = models.FloatField(max_length=20,null=True,blank=True)
    disk_free_value = models.FloatField(max_length=20,null=True,blank=True)
    load_fifteen_value = models.FloatField(max_length=20,null=True,blank=True)
    load_five_value = models.FloatField(max_length=20,null=True,blank=True)
    load_one_value = models.FloatField(max_length=20,null=True,blank=True)
    mem_buffers_value = models.FloatField(max_length=20,null=True,blank=True)
    mem_cached_value = models.FloatField(max_length=20,null=True,blank=True)
    mem_free_value = models.FloatField(max_length=20,null=True,blank=True)
    pkts_in_value = models.FloatField(max_length=20,null=True,blank=True)
    pkts_out_value = models.FloatField(max_length=20,null=True,blank=True)
    proc_run_value = models.FloatField(max_length=20,null=True,blank=True)
    proc_total_value = models.FloatField(max_length=20,null=True,blank=True)
    swap_free_value = models.FloatField(max_length=20,null=True,blank=True)
    STATUS_NORMAL = 1
    STATUS_ANOMALY = -1
    STATUS_UNKNOW = 0
    STATUS_ITEMS = (
        (STATUS_NORMAL, '正常'),
        (STATUS_ANOMALY, '异常'),
        (STATUS_UNKNOW, '未标记'),
    )
    label = models.IntegerField(default=STATUS_UNKNOW, choices=STATUS_ITEMS)
    class Meta:
        db_table = 'prosampledb'
        ordering = ['nodename','timestamp']
        unique_together = [['nodename','timestamp']]


#Metrics
class Metric(models.Model):
    metric = models.CharField(max_length=30)
    class Meta:
        db_table = 'metric'

#Nodenames
class Nodename(models.Model):
    nodename = models.CharField(max_length=60)
    class Meta:
        db_table = 'nodename'

#Algorithm
#包括所有预测和检测算法，算法ID,算法名称，算法类型（预测/检测）,是否需要提前训练，参数
class Algorithm(models.Model):
    algorithmid = models.IntegerField(primary_key=True)
    algorithmname = models.CharField(max_length=50)

    PREDICT_ALGOTITHM = 0
    DETECT_ALGORITHM = 1
    TYPE_ITEMS = (
        (PREDICT_ALGOTITHM, '预测'),
        (DETECT_ALGORITHM,'检测')
    )
    algorithmtype = models.IntegerField(default=PREDICT_ALGOTITHM, choices=TYPE_ITEMS)

    TRAIN_NO = 0
    TRAIN_YES = 1
    ISTRAIN_ITEMS = (
        (TRAIN_NO,'无需训练模型'),
        (TRAIN_YES,'需训练模型')
    )
    istrain = models.IntegerField(default=TRAIN_NO,choices=ISTRAIN_ITEMS)

    algorithmparameter = models.CharField( max_length=300)

    class Meta:
        db_table = 'algorithm'

class PredictMethod(models.Model):
    methodname = models.CharField(max_length=50)
    algorithm = models.CharField(max_length=50)
    paramvalue = models.CharField(max_length=300)
    path = models.CharField(max_length=100)
    class Meta:
        db_table = 'predictmethod'

class Mulvisualize(models.Model):
    dedimensionname = models.CharField(max_length=50,primary_key=True)
    dedimensionalgorithm = models.CharField(max_length=50)
    dedimensionnodelist = models.CharField(max_length=1000)
    dedimensionmetrics = models.CharField(max_length=1000)
    dedimensionalparamvalue = models.CharField(max_length=300)
    dedimensionpath = models.CharField(max_length=100)
    class Meta:
        db_table = 'dedimension'