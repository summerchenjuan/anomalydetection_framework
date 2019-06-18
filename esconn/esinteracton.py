from elasticsearch import Elasticsearch
import pandas as pd


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
    print(total)
    for i in range(int(total / 1000)):
        res = es.scroll(scroll_id=scroll_id, scroll='5m')  # scroll参数必须指定否则会报错
        mdata += res["hits"]["hits"]
    return mdata


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


# 解析获取后的数据
def mdata_dataframe(mdata):
    '''
    将从es查询获取的数据解析为DataFrame形式
    :param mdata:
    :return:
    '''
    sourcediclist = [dic['_source'] for dic in mdata]
    source = pd.DataFrame(sourcediclist)
    source.rename(columns={'@timestamp': 'timestamp'}, inplace=True)
    return source

def search_nodename_timestamp_dataframe(nodename, starttime, endtime, metrics):
    mdata = search_nodename_timestamp(nodename, starttime, endtime, metrics)
    source = mdata_dataframe(mdata)
    return source