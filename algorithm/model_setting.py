'''
有关于异常检测各模型的公共配置
'''

#所有Metric
Metrics = ['bytes_in_value', 'bytes_out_value', 'cpu_idle_value',
'disk_free_value', 'load_fifteen_value', 'load_five_value',
'load_one_value', 'mem_buffers_value', 'mem_cached_value',
'mem_free_value', 'pkts_in_value', 'pkts_out_value',
'proc_run_value', 'proc_total_value', 'swap_free_value'
]

#所有Metric的最大值最小值配置
METRIC_MIN_MAX = {
    'bytes_in_value':{
        'min':0,
        'max':150000000,
    },
    'bytes_out_value':{
        'min':0,
        'max':150000000,
    },
    'cpu_idle_value':{
        'min':0,
        'max':100,
    },
    'disk_free_value':{
        'min':0,
        'max':500,
    },
    'load_fifteen_value':{
        'min':0,
        'max':150,
    },
    'load_five_value':{
        'min':0,
        'max':150,
    },
    'load_one_value':{
        'min':0,
        'max':150,
    },
    'mem_buffers_value':{
        'min':0,
        'max':50000000,
    },
    'mem_cached_value':{
        'min':0,
        'max':1000000,
    },
    'mem_free_value':{
        'min':0,
        'max':100000000,
    },
    'pkts_in_value':{
        'min':0,
        'max':300000,
    },
    'pkts_out_value':{
        'min':0,
        'max':300000,
    },
    'proc_run_value':{
        'min':0,
        'max':300,
    },
    'proc_total_value':{
        'min':0,
        'max':2000,
    },
    'swap_free_value':{
        'min':0,
        'max':50000000,
    }
}

#model保存path
PATH = 'D://IHEP/chenj/anomaly/sqlsave/modelpath/es/'
