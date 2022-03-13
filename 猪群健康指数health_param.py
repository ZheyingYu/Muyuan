# 数据包
import sys
import re
from pyhive import hive
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import scipy.stats as stats
from sklearn.preprocessing import KBinsDiscretizer
import requests
import json
import time

from graphframes import *

from matplotlib import pyplot as plt
from DBoperation import *

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"/data/datamining/envir/SimHei.ttf", size=30) #中文字体


pd.set_option('display.max_columns', None) #显示所有列
sys.path.append('/data/datamining/Py_functions/get_data')
from get_data_func import * ##取数function  

currenttime =   time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))   # "2020-10-01 16:46:11" 
currenttime = '2021-07-01'

mysql_engine = create_engine('mysql+pymysql://barview:barview@10.106.10.17:13306/my_bar_patrol')

myanacon = create_engine(
    'hive://sz210428470@10.106.20.15:10000/myana',
    connect_args={'configuration': {'hive.exec.reducers.max': '123'},'password':'xsj123!','auth':'CUSTOM'},
)

mydwcon = create_engine(
    'hive://sz210428470@10.106.20.15:10000/mydw',
    connect_args={'configuration': {'hive.exec.reducers.max': '123'},'password':'xsj123!','auth':'CUSTOM'},
)

mysacon = create_engine(
    'hive://sz210428470@10.106.20.15:10000/mysa',
    connect_args={'configuration': {'hive.exec.reducers.max': '123'},'password':'xsj123!','auth':'CUSTOM'},
)


# 原始表格
currentbatch_sql = '''SELECT fbatchno, ffieldid, fareaid, funitcount FROM myana.datamining_DWD_BI  WHERE fsegmentid = 13906 AND fstate = 1'''
currentbatch_df = pd.read_sql_query(currentbatch_sql, myanacon)

def get_factors(currentbatch_df):
    '''健康因素取数'''
    
    # 来源批次数，来源场区数，来源单元数
    source_discrete_sql = '''SELECT fbatchno, count(source_batchno) AS source_batchnum, count(DISTINCT ffieldid) AS source_fieldnum, sum(funitcount) AS source_unitnum 
    FROM 
    (SELECT A.fbatchno,A.source_batchno, B.ffieldid, B.fareaid, B.funitcount
    FROM 
    (SELECT a.fbatchno, b.fbatchno AS source_batchno
    FROM 
    (SELECT*FROM myana.datamining_DWD_BI WHERE fsegmentid = 13906 AND fstate = 1) a
    LEFT JOIN myana.ybb_batch_retrospect_change_08 b
    ON a.fbatchno = b.to_fbatchno 
    AND b.fsegmentid  = 13905 AND b.to_segmentid = 13906) A
    LEFT JOIN myana.datamining_DWD_BI B 
    ON A.source_batchno = B.fbatchno)
    GROUP BY fbatchno'''

    source_discrete_df = pd.read_sql_query(source_discrete_sql, myanacon)
    
    #来源批次死亡率
    source_deathrate_sql = '''
    SELECT fbatchno, sum(1-0.01*liverate)/(count(DISTINCT source_batchno)) AS source_deathrate
    FROM 
    (SELECT * FROM 
    (SELECT a.fbatchno, b.fbatchno AS source_batchno
    FROM 
    (SELECT*FROM myana.datamining_DWD_BI WHERE fsegmentid = 13906 AND fstate = 1) a
    LEFT JOIN myana.ybb_batch_retrospect_change_08 b
    ON a.fbatchno = b.to_fbatchno 
    AND b.fsegmentid  = 13905 AND b.to_segmentid = 13906) A
    LEFT JOIN (SELECT fbatchno AS source_batchno2, liverate FROM myana.datamining_DWD_SE_nurse) B 
    ON A.source_batchno = B.source_batchno2 )
    GROUP BY fbatchno
    '''
    source_deathrate_df = pd.read_sql_query(source_deathrate_sql, myanacon)
    
    #来源批次初均重
    source_weight_sql = '''
    SELECT fbatchno, avg(avg_fweight) AS source_weight
    FROM 
    (
    SELECT a.fbatchno, b.fbatchno AS source_batchno, b.avg_fweight
    FROM 
    (SELECT*FROM myana.datamining_DWD_BI WHERE fsegmentid = 13906 AND fstate = 1) a
    LEFT JOIN myana.ybb_batch_retrospect_change_08 b
    ON a.fbatchno = b.to_fbatchno AND b.fsegmentid  = 13905 AND b.to_segmentid = 13906
    ) 
    GROUP BY fbatchno 

    '''
    source_weight_df =  pd.read_sql_query(source_weight_sql, myanacon)
    
    #来源批次不合格重, 不合格率 (即现批次转入的不合格重和不合格率，fchangetype = 3表示转入)
    source_bad_sql = '''
    SELECT t.fbatchno
    ,sum(CASE WHEN t.fqualified = 1 then  t.`fqty`  ELSE 0 END)/sum(t.`fqty`) AS source_badrate
    ,sum(CASE WHEN t.fqualified = 1 THEN t.fweight ELSE 0 END)/sum(CASE WHEN t.fqualified = 1 then  t.`fqty`  ELSE 0 END) AS source_badweightavg
    FROM (
        SELECT a.fbatchno,b.`fqty` ,b.`fqualified`, b.`fweight` FROM
            (SELECT*FROM myana.datamining_DWD_BI WHERE fsegmentid = 13906 AND fstate = 1) a
            LEFT JOIN mydw.`source_my_sc_changelog` b  ON a.fbatchno = b.fbatchno AND b.fchangetype = 3 
        ) t
    GROUP BY fbatchno;
    '''
    source_bad_df =  pd.read_sql_query(source_bad_sql, myanacon)
    
    
    
    #关联表格
    currentbatch_df = currentbatch_df.merge(source_discrete_df, how = 'left', on = 'fbatchno')
    currentbatch_df = currentbatch_df.merge(source_weight_df, how = 'left', on = 'fbatchno')
    currentbatch_df = currentbatch_df.merge(source_deathrate_df, how = 'left', on = 'fbatchno')
    currentbatch_df = currentbatch_df.merge(source_bad_df, how = 'left', on = 'fbatchno')
    
    # 30天同期工段
    batch_firstdate_sql = '''
    SELECT fbatchno, first_changedate, last_changedate, source_ffieldid, ffieldid
    FROM 
    (
    SELECT a.fbatchno, b.fbatchno AS source_batchno,b.`first_changedate` , b.`last_changedate` ,b.`ffieldid` AS source_ffieldid , b.`to_ffieldid` AS ffieldid ,row_number() over (distribute by a.fbatchno sort by b.first_changedate ASC) as num 
    FROM 
    (SELECT*FROM myana.datamining_DWD_BI WHERE fsegmentid = 13906 AND fstate = 1) a
    LEFT JOIN myana.ybb_batch_retrospect_change_08 b
    ON a.fbatchno = b.to_fbatchno AND b.fsegmentid  = 13905 AND b.to_segmentid = 13906
    ) 
    WHERE num = 1

    '''
    batch_firstdate =  pd.read_sql_query(batch_firstdate_sql, myanacon)

    batch_firstdate.first_changedate = pd.to_datetime(batch_firstdate.first_changedate, format = '%Y-%m-%d')
    batch_firstdate.last_changedate = pd.to_datetime(batch_firstdate.last_changedate, format = '%Y-%m-%d')
    
    sql_zhuanqun2 = "SELECT `fbatchno` , `to_fbatchno` , `first_changedate` ,`last_changedate` ,`ffieldid` ,`to_ffieldid` FROM myana.ybb_batch_retrospect_change_08 WHERE `to_segmentid` = 13906 AND `fsegmentid` = 13905 AND `first_changedate` > '2021-01-01'"
    zhuanqun2_data = query_data(method = 'inceptor', sql = sql_zhuanqun2)
    
    def calc_fieldnum(ID):
        '''对每个批次计算上游场区数量
            ID: str 批次号
        '''
        #时间参照
        firstdate = batch_firstdate.loc[batch_firstdate.fbatchno == ID].first_changedate
        #场区ID 
        fieldid = batch_firstdate.loc[batch_firstdate.fbatchno == ID].ffieldid.min()

        #时间区间
        up_time = firstdate + timedelta(days = 15)
        down_time = firstdate - timedelta(days = 15)

        #筛选出时间段的批次号
        #筛选条件：1. 第一次转入时间 < up 并且 最后一次转入时间 > down
        sametime_batch = zhuanqun2_data.loc[(zhuanqun2_data.first_changedate <= up_time.iloc[0].strftime('%Y-%m-%d')) & (zhuanqun2_data.last_changedate >= down_time.iloc[0].strftime('%Y-%m-%d'))]
        #筛选条件：2. 批次号都来自同一个场区
        sametime_batch2 = sametime_batch.loc[sametime_batch.to_ffieldid == fieldid]


        #检查是否空集
        if sametime_batch2.empty : 
            fieldnum = 0
        else:    
            #那么sametime_batch 里的 fbatchno 就是上游目标批次
            fieldnum = sametime_batch2.ffieldid.unique().shape[0]

        return fieldnum 
    
    
    #获得30天同期工段场区数据
    batch_list = list(currentbatch_df.iloc[:, 0])

    for batchid in batch_list: 
        if currentbatch_df.loc[currentbatch_df.fbatchno == batchid].source_batchnum.min() == 0:
            currentbatch_df.loc[currentbatch_df.fbatchno == batchid, '30dfieldnum'] = 0
        else: 
            fieldnum = calc_fieldnum(batchid)
            #更新到表格中
            currentbatch_df.loc[currentbatch_df.fbatchno == batchid, '30dfieldnum'] = fieldnum 
            
    return currentbatch_df




#计算健康指数
def calc_healthparam(currentbatch_df):
    '''健康指数 = a1*来源批次场区数+a2*来源初均重+a3*来源批次死亡率+a4*来源批次不合格率+a5*30天同期工段场区数 + a6*来源批次数 + a7*来源批次单元数 + a8*来源批次不合格率+ 函数常数项'''
    
    #离散：来源批次数
    data = currentbatch_df.fillna(0)

    #对于source_BadWeightAvg 做分箱（大于0或等于0）
    data.loc[data.source_badweightavg == 0.0, 'source_badweightavg'] = 0
    data.loc[data.source_badweightavg > 0.0, 'source_badweightavg'] = 1
    data.loc[data.source_unitnum >= 4.0, 'source_unitnum'] = 4.0
    data.loc[data.source_batchnum >= 4.0,'source_batchnum'] = 4.0

    X = data.loc[:, ['source_batchnum', 'source_unitnum', 'source_fieldnum', 'source_weight', 'source_deathrate', 'source_badrate', 'source_badweightavg', '30dfieldnum']]

    #转化呈数值型的数据，方便之后的离散
    discreteX = X.loc[:, ['source_batchnum', 'source_unitnum', 'source_badweightavg']]
    X.loc[:, ['source_batchnum', 'source_unitnum', 'source_badweightavg']] = discreteX.astype(str)

    #对离散值进行离散化
    X_onehot = pd.get_dummies(X)

    
    health_coeff = {'来源批次厂区数': 0.0244, '来源初均重':0.262, '来源批次死亡率':-0.208, '来源批次不合格率':0.0201,
                   '30天同期工段场区数': -0.171, '来源批次数 = 1': 0.0566, '来源批次数 = 2':0.0482, '来源批次数 = 3':-0.0035,
                   '来源批次数 = 4':-0.0986, '来源批次单元数 = 1': 0.3334, '来源批次单元数 = 2': 0.0219, '来源批次单元数 = 3':-0.1309,
                   '来源批次单元数 = 4': -0.2216, '来源不合格率大于0': 0.1234, '来源不合格率等于0': -0.1206, 'C':-0.2258}
    
    health_param = health_coeff['来源批次厂区数']*X_onehot.source_fieldnum + health_coeff['来源初均重']*X_onehot.source_weight + health_coeff['来源批次死亡率']*X_onehot.source_deathrate 
    + health_coeff['来源批次不合格率']*X_onehot.source_badrate + health_coeff['30天同期工段场区数']*X_onehot['30dfieldnum'] + health_coeff['来源批次数 = 1']*X_onehot.source_batchnum_1
    +health_coeff['来源批次数 = 2']*X_onehot.source_batchnum_2 + health_coeff['来源批次数 = 3']*X_onehot.source_batchnum_3 + health_coeff['来源批次数 = 4']*X_onehot.source_batchnum_4
    +health_coeff['来源批次单元数 = 1']*X_onehot['source_unitnum_1.0'] + health_coeff['来源批次单元数 = 2']*X_onehot['source_unitnum_2.0'] 
    + health_coeff['来源批次单元数 = 3']*X_onehot['source_unitnum_3.0']+health_coeff['来源批次单元数 = 4']*X_onehot['source_unitnum_4.0'] + health_coeff['来源不合格率等于0']*X_onehot['source_badweightavg_0.0'] 
    + health_coeff['来源不合格率大于0']*X_onehot['source_badweightavg_1.0']+ health_coeff['C']
                    
    currentbatch_df['health_param'] = -1*(X_onehot.source_batchnum_0.astype('int') -1)*health_param
    
    return currentbatch_df 


#代入方程
currentbatch_test = calc_healthparam(get_factors(currentbatch_df))
#保存
currentbatch_test.to_csv('test20210721.csv', index = False)
                              
                    
