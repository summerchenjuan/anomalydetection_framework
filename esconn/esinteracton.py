from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd
import datetime
import time
import numpy as np
# 与es的数据交互


# 连接es
def esconn():
    es = Elasticsearch(['esheader01.ihep.ac.cn'], http_auth=('elastic', 'mine09443'), timeout=3600)
    return es


def search_nodename_timestamp(nodename, starttime, endtime, metrics):
    '''
    通过nodename/timestamps/metrics 获取数据
    :param nodename:str nodename
    :param starttime:datetime
    :param endtime:datetime
    :param metrics:list
    :return: mdata list, 每一个元素是一个document
    mdata e.g.
   [
     {
      '_id': 'J1HRUWsBKDKewOy4y_UW',
       '_source': {'time': '1560445803', 'load_five_value': 401.53, ...., '@timestamp': '2019-06-13T17:10:03.000Z',...},
       '_type': 'doc',
       '_score': 8.54739,
       '_index': 'ganglia_agg-2019.06.07-000012'
       },
     {
       '_id': 'klXaUWsBKDKewOy48XbO',
        '_source': {'time': '1560446391', 'load_five_value': 400.93,...., '@timestamp': '2019-06-13T17:19:51.000Z',... },
        '_type': 'doc',
         '_score': 8.579774,
         '_index': 'ganglia_agg-2019.06.07-000012'
         },

       ...
    ]
    '''
    es = esconn()
    query_json = search_nodename_timestamp_queryjson(nodename=nodename, starttime=starttime, endtime=endtime,metrics=metrics)
    queryData = es.search(index='search_ganglia', scroll='5m', timeout='3s', size=1000, body=query_json)
    mdata = queryData.get("hits").get("hits")
    if not mdata:
        print('empty!')
    scroll_id = queryData["_scroll_id"]
    total = queryData["hits"]["total"]
    for i in range(int(total / 1000)):
        es.transport.send_get_body_as = 'POST'
        # res = es.scroll(scroll_id=scroll_id, scroll='5m')  # scroll参数必须指定否则会报错
        res = es.scroll(body={'scroll':'5m','scroll_id':scroll_id})
        mdata += res["hits"]["hits"]
    return mdata

def search_bulk(index,query_json):
    """
    :param index: 要查询的index
    :param query_json: DSL语句
    :return: 查询到的mdata
    """
    es = esconn()
    queryData = es.search(index=index, scroll='5m', timeout='3s', size=1000, body=query_json)
    mdata = queryData.get("hits").get("hits")
    if not mdata:
        print('empty!')
    scroll_id = queryData["_scroll_id"]
    total = queryData["hits"]["total"]
    for i in range(int(total / 1000)):
        #res = es.scroll(scroll_id=scroll_id, scroll='5m')
        es.transport.send_get_body_as = 'POST'
        res = es.scroll(body={'scroll': '5m', 'scroll_id': scroll_id})
        mdata += res["hits"]["hits"]
    return mdata

def search_isexist(index,query_json):
    """
    查看是否查找到某document
    :return: 0 没有查找到满足条件的document，
              1 反之
    """
    es = esconn()
    queryData = es.search(index=index, scroll='5m', timeout='3s', size=1000, body=query_json)
    mdata = queryData.get("hits").get("hits")
    if not mdata:
        return 0
    return 1


def insert_bulk(index,mdata):
    """
    批量写入
    :param index:
    :param mdata:
    :return:
    """
    es = esconn()
    # 批量写入
    ACTION=[]
    print(mdata)
    for unit in mdata:
        #根据nodename和time创建key
        id=unit['_source']['nodename'].replace(".ihep.ac.cn","")+"_"+str(unit['_source']['time'])
        print(id)
        action = {
            "_index":index,
            "_type":"doc",
            "_id":id,
            "_source":unit['_source'],
            "_op_type":"index"
        }
        ACTION.append(action)
        if len(ACTION)>=500:
            success, _ = bulk(es, ACTION,index=index)
            ACTION=[]
    if len(ACTION):
        success, _ = bulk(es, ACTION,index=index)


def upadte_bulk(index,mdata,docs):
    es = esconn()
    ACTION = []
    for unit in mdata:
        # 根据nodename和time计算出key
        id = unit['_source']['nodename'].replace(".ihep.ac.cn", "") + "_" + str(unit['_source']['time'])
        action = {
            "_index": index,
            "_type": "doc",
            "_id": id,
            "doc": docs,
            "_op_type": "update"
        }
        ACTION.append(action)
        if len(ACTION) >= 500:
            success, _ = bulk(es, ACTION, index=index)
            ACTION = []
    if len(ACTION):
        success, _ = bulk(es, ACTION, index=index)

# 向已存在的documents中加field

# 各query_json
def search_nodename_timestamp_queryjson(nodename, starttime, endtime, metrics):
    '''
    :param nodename: str nodename
    :param starttime: datetime timestamp
    :param endtime:datetime timestamp
     :param metrics:list
    :return:query_json
    '''
    fields_metrics = ['nodename', '@timestamp']
    fields_metrics.extend(metrics)
    querynodename = "nodename:" + nodename
    query_json = {
        "_source": fields_metrics,
        "query": {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "query": querynodename,
                        }
                    },
                    {
                        "range": {
                            "@timestamp": {
                                "gte": starttime,
                                "lte": endtime,
                                "format": "epoch_millis"
                            }
                        },
                    },
                ],
                "filter": [],
                "should": [],
                "must_not": [],
            }
        },
        "sort": {
            "@timestamp": {
                "order": "asc"
            }
        }
    }
    # fields_metrics = ['nodename', '@timestamp']
    # fields_metrics.extend(metrics)
    # query_nodename = "nodename:" + nodename
    # query_json = {
    #     "_source": fields_metrics,
    #     "query": {
    #         "bool": {
    #             "must": [
    #                 {
    #                     "query_string": {
    #                         "query": query_nodename
    #                     }
    #                 },
    #                 {
    #                     "range": {
    #                         "@timestamp": {
    #                             "gte": starttime,
    #                             "lte": endtime,
    #                             "format": "epoch_millis"
    #                         }
    #                     },
    #                 },
    #             ],
    #             "filter": [],
    #             "should": [],
    #             "must_not": []
    #         },
    #         "sort": {
    #             "@timestamp": {
    #                 "order": "asc"
    #             }
    #         }
    #     }
    # }
    return query_json



def is_search_nodename_timestamp_queryjson(nodename,timestamp):
    """
        查看异常候选库（mlganglia-agg）中是否存在 某timestamp的nodename
        :param nodename:
        :param timestamp:
        :return: 0 没有查找到满足条件的document，index中没有该条数据
                  1 反之，有该条数据
        """
    querynodename = "nodename:" + nodename
    query_json = {
        "query": {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "query": querynodename,
                        }
                    },
                    {
                        "range": {
                            "@timestamp": {
                                "gte": timestamp,
                                "lte": timestamp,
                                "format": "epoch_millis"
                            }
                        },
                    },
                ],
                "filter": [],
                "should": [],
                "must_not": [],
            }
        },
    }
    return query_json

def search_nodename_timestamp_mantag_queryjson(nodename,timestamp):
    """
    查看timestamp的nodename是否被标记为了异常
    :param nodename:
    :param timestamp:
    :return:
    """
    querynodename = "nodename:" + nodename
    query_json = {
        "query": {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "query": querynodename,
                        }
                    },
                    {
                        "range": {
                            "@timestamp": {
                                "gte": timestamp,
                                "lte": timestamp,
                                "format": "epoch_millis"
                            }
                        },
                    },
                    {
                       "term":{
                           "tags":"mantag"
                       }
                    }
                ],
                "filter": [],
                "should": [],
                "must_not": [],
            }
        },
    }
    return query_json

# 解析获取后的数据
def mdata_dataframe(mdata):
    '''
    将从es查询获取的数据解析为DataFrame形式
    :param mdata:从es查询获取的数据
    :return:DataFrame类型
    '''
    sourcediclist = [dic['_source'] for dic in mdata]
    source = pd.DataFrame(sourcediclist)
    source.rename(columns={'@timestamp': 'timestamp'}, inplace=True)
    source['timestamp'] = source['timestamp'].map(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ") + datetime.timedelta(hours=8))
    return source


def search_nodename_timestamp_dataframe(nodename, starttime, endtime, metrics):
    mdata = search_nodename_timestamp(nodename, starttime, endtime, metrics)
    source = mdata_dataframe(mdata)
    return source


def search_nodename_timestamp_dataframe_miss(nodename, starttime, endtime, metrics):
    mdata = search_nodename_timestamp(nodename, starttime, endtime, metrics)
    source = mdata_dataframe(mdata)
    #对前后两个时间戳进行比较，若两时间戳存在缺失数据(即两时间戳超过10分钟)，则将该行数据某metirc修改为np.nan  在后续对该两条时间戳的样本同缺失样本处理方式一样删除
    for i in range(len(source)-1):
        if((source['timestamp'][i+1]-source['timestamp'][i]).total_seconds() >= 600):
            source[metrics[0]][i] = np.nan
            source[metrics[0]][i+1] = np.nan
            print(source['timestamp'][i])
            print(source['timestamp'][i+1])
    return source