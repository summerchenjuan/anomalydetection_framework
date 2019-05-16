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
class PriSample(models.Model):
    nodename = models.CharField(max_length=100)
    timestamp = models.DateTimeField(max_length=100)
    bytes_in_value = models.FloatField(max_length=20,null=True)
    bytes_out_value = models.FloatField(max_length=20,null=True)
    cpu_idle_value = models.FloatField(max_length=20,null=True)
    disk_free_value = models.FloatField(max_length=20,null=True)
    load_fifteen_value = models.FloatField(max_length=20,null=True)
    load_five_value = models.FloatField(max_length=20,null=True)
    load_one_value = models.FloatField(max_length=20,null=True)
    mem_buffers_value = models.FloatField(max_length=20,null=True)
    mem_cached_value = models.FloatField(max_length=20,null=True)
    mem_free_value = models.FloatField(max_length=20,null=True)
    pkts_in_value = models.FloatField(max_length=20,null=True)
    pkts_out_value = models.FloatField(max_length=20,null=True)
    proc_run_value = models.FloatField(max_length=20,null=True)
    proc_total_value = models.FloatField(max_length=20,null=True)
    swap_free_value = models.FloatField(max_length=20,null=True)
    # bytes_in_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
    # bytes_out_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
    # cpu_idle_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
    # disk_free_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
    # load_fifteen_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
    # load_five_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
    # load_one_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
    # mem_buffers_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
    # mem_cached_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
    # mem_free_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
    # pkts_in_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
    # pkts_out_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
    # proc_run_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
    # proc_total_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
    # swap_free_value = models.DecimalField(max_digits=15,decimal_places=2,null=True)
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


class Metric(models.Model):
    metric = models.CharField(max_length=30)
    class Meta:
        db_table = 'metric'

# class Anomaly(models.Model):
#     STATUS_NORMAL = 1
#     STATUS_ANOMALY = -1
#     STATUS_UNKNOW = 0
#     STATUS_ITEMS = (
#         (STATUS_NORMAL,'正常'),
#         (STATUS_ANOMALY,'异常'),
#         (STATUS_UNKNOW,'未标记'),
#     )
#     label = models.IntegerField(default=STATUS_UNKNOW,choices=STATUS_ITEMS)
#     nodename = models.CharField(max_length=100)
#     timestamp = models.DateTimeField()
#     class Meta:
#         db_table = 'anomaly'
#         ordering = ['nodename', 'timestamp']