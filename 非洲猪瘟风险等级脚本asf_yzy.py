import sys
import re
from pyhive import hive
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import scipy.stats as stats
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
import requests
import json
import time
from graphframes import *

from matplotlib import pyplot as plt

from DBoperation import *

from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"/data/datamining/envir/SimHei.ttf", size=30) #中文字体


pd.set_option('display.max_columns', None) #显示所有列
sys.path.append('/data/datamining/Py_functions/get_data') #新路径
from get_data_func import * ##取数function  


mysql_engine = create_engine('mysql+pymysql://barview:barview@10.106.10.17:13306/my_bar_patrol')

myanacon = create_engine(
    'hive://sz210428470@10.106.20.15:10000/myana',
    connect_args={'configuration': {'hive.exec.reducers.max': '123'},'password':'xsj123!','auth':'CUSTOM'},
)


mydwcon = create_engine(
    'hive://hive@10.106.56.41:10000/my_dw',
    connect_args={'configuration': {'hive.exec.reducers.max': '123'},'auth':'NONE'},
)

mysacon = create_engine(
    'hive://sz210428470@10.106.20.15:10000/mysa',
    connect_args={'configuration': {'hive.exec.reducers.max': '123'},'password':'xsj123!','auth':'CUSTOM'},
)



#主要脚本

 #星环链接形式
hiveConn = hive.Connection(host='10.106.20.15', port=10000, username='szchenye', password='szcy230#',
                                   database='mydw', auth='CUSTOM')
def str_to_list(x):
        '''辅助方程：字符串转列表'''
        data_list = x["funitnames"].replace("第","").replace("单元","").split(";")
        return data_list[0:-1]

def get_database():
    '''得到基础表'''
    
    #批次信息和中文信息表合并
    database_sql = '''SELECT a.*, b.fsegmentid, b.ffeeder_name, b.ffeeder_code, b.funitnames FROM (SELECT ffieldid,fareaid,fregionname,ffield,ffieldtypename FROM myana.datamining_DIM_AF) a 
    RIGHT JOIN (SELECT DISTINCT fsegmentid, ffieldid, fareaid, ffeeder_name, ffeeder_code, funitnames FROM myana.datamining_DWD_BI) b
    ON a.ffieldid = b.ffieldid AND a.fareaid = b.fareaid
    '''
    database = pd.read_sql(database_sql, hiveConn)

    #将funitnames 转化成单一数字字符，代表单元号
    database['funitnames'] = database.apply(lambda x: str_to_list(x), axis=1)
    database = database.reindex(
        database.index.repeat(database.funitnames.str.len())).assign(
        funitnames=np.concatenate(database.funitnames.values))
    
    #筛选条件：1. 生长场：出现后备段统一归在育肥；2.繁殖场有保育情况剔除
    database.loc[database.ffieldtypename == '生长场', 'fsegmentid'] = 13907
    database = database.drop(database.loc[(database.ffieldtypename == '繁殖场') & (database.fsegmentid == 13906)].index, axis = 0)
    
    database = database.reset_index()
    
    database2 = database.copy() #拷贝
    #新增一个字段便于后面的匹配，字段包括场区ID， 区域ID， 工段ID 和单元号
    database2['unit_link_id'] = database2.fareaid.astype(str) + '|' +  database2.ffieldid.astype(str) + '|' + database2.fsegmentid.astype(str) + '|' + database2.funitnames.astype(str)
    
    database['unit_link_id'] = database2.unit_link_id
    #输出
    return database
    
def get_batch_info():
    '''输出以批次号为单位的信息表'''
    #批次信息和中文信息表合并
    batch_info_sql = '''SELECT b.fbatchno, a.*, b.fsegmentid, b.ffeeder_name, b.ffeeder_code, b.funitnames  FROM (SELECT ffieldid,fareaid,fregionname,ffield,ffieldtypename FROM myana.datamining_DIM_AF) a 
    RIGHT JOIN myana.datamining_DWD_BI b
    ON a.ffieldid = b.ffieldid'''

    batch_info = pd.read_sql(batch_info_sql, hiveConn )

    #将funitnames 转化成单一数字字符，代表单元号
    batch_info['funitnames'] = batch_info.apply(lambda x: str_to_list(x), axis=1)
    batch_info = batch_info.reindex(
        batch_info.index.repeat(batch_info.funitnames.str.len())).assign(
        funitnames=np.concatenate(batch_info.funitnames.values))

    #筛选条件：1. 生长场：出现后备段统一归在育肥；2.繁殖场有保育情况剔除

    batch_info.loc[batch_info.ffieldtypename == '生长场', 'fsegmentid'] = 13907
    batch_info = batch_info.drop(batch_info.loc[(batch_info.ffieldtypename == '繁殖场') & (batch_info.fsegmentid == 13906)].index, axis = 0)

    #可以新增一个字段便于后面的匹配，字段包括场区ID， 区域ID， 工段ID 和单元号
    batch_info['unit_link_id'] = batch_info.fareaid.astype(str) + '|' +  batch_info.ffieldid.astype(str) + '|' + batch_info.fsegmentid.astype(str) + '|' + batch_info.funitnames.astype(str)

    #重置序号
    batch_info = batch_info.reset_index(drop = True) #drop参数会取代原来的index 
    
    return batch_info 

database = get_database()
batch_info = get_batch_info()



def get_asf_count(database):
    '''计算历史单元阳性总次数'''
    #送检信息
    sj_info_sql = '''
    SELECT batch_no, unit, pcr_result, pcrdl_result, date_format(sampling_time, 'yyyy-MM-dd') AS sample_time FROM myana.datamining_DWS_DS_antigen_UD
    '''
    sj_info = pd.read_sql(sj_info_sql,hiveConn)

    #送检表和上面批次信息表格根据批次号匹配，得到相应的unit_link_id 
    sj_info = sj_info.merge(batch_info.loc[:, ['fbatchno','funitnames', 'unit_link_id']], how = 'left', left_on = ['batch_no', 'unit'], right_on = ['fbatchno', 'funitnames'])
   
    # 非洲猪瘟阳性表格（2018至今），positive_num表示计算好阳性的总天数
    asf_data_sql = '''SELECT batch_no, unit, pcr_result, pcrdl_result, date_format(sampling_time, 'yyyy-MM-dd') AS sample_time 
    FROM myana.datamining_DWS_DS_antigen_UD 
    WHERE (concat_ws(',',pcrdl_result) LIKE '%A%' OR concat_ws(',',pcr_result) LIKE '%A%') and `sampling_time`>='2018-10-01'
    '''
    asf_data = pd.read_sql(asf_data_sql, hiveConn)
    #获得单独unit_link_id
    asf_data = asf_data.merge(batch_info.loc[:, ['fbatchno','funitnames', 'unit_link_id']], how = 'left', left_on = ['batch_no', 'unit'], right_on = ['fbatchno', 'funitnames']) 
    
    #计算每个单元的历史阳性次数，注意同一天算一次
    unit_id_list = list(asf_data.unit_link_id.dropna().unique())

    #建立新的dataframe 
    asf_count = pd.DataFrame(unit_id_list, columns = ['unit_link_id'])

    for unit_id in unit_id_list: 
        #计算天数
        num = asf_data.loc[asf_data.unit_link_id == unit_id].sample_time.unique().shape[0]
        #更新df
        asf_count.loc[asf_count.unit_link_id == unit_id, 'asf_total'] = num 
        
    # 合并到基础表格中 （按照unit_link_id)
    database = database.merge(asf_count, how = 'left', on = 'unit_link_id')
    
    return database

def get_asf_rate(database): 
    '''计算三轮非瘟比例'''
    # 每一轮检测的函数
    def asf_annal_round(starttime, endtime):
        '''输入起始时间和末尾时间
            输出每一轮的非洲猪瘟的检测结果。0 表示有送检未检出阳性，1表示有送检并且检出阳性
        '''
        #每一轮的送检表格
        antigen_sql = '''select * from myana.datamining_DWS_DS_antigen_UD where `sampling_time`>='{start}' and `sampling_time`<='{end}' '''
        antigen_sql = antigen_sql.format(start = starttime, end = endtime)
        antigen = pd.read_sql(antigen_sql, hiveConn)

        #每一轮的阳性检测表格
        asf_sql = '''select * from myana.datamining_DWS_DS_antigen_UD where `sampling_time`>='{start}' and `sampling_time`<='{end}' and (concat_ws(',',pcrdl_result) LIKE '%A%' OR concat_ws(',',pcr_result) LIKE '%A%') '''
        asf_sql = asf_sql.format(start = starttime, end = endtime)
        asf = pd.read_sql(asf_sql, hiveConn)


        #获得单独unit_link_id
        antigen = antigen.merge(batch_info.loc[:, ['fbatchno','funitnames', 'unit_link_id']], how = 'left', left_on = ['batch_no', 'unit'], right_on = ['fbatchno', 'funitnames']) 
        asf = asf.merge(batch_info.loc[:, ['fbatchno','funitnames', 'unit_link_id']], how = 'left', left_on = ['batch_no', 'unit'], right_on = ['fbatchno', 'funitnames']) 

        #对Unit_link_id 做计数统计
        antigen_count = pd.DataFrame(antigen.unit_link_id.value_counts().reset_index())
        antigen_count.columns = ['unit_link_id', 'antigen_total'] #重命名列名

        asf_count = pd.DataFrame(asf.unit_link_id.value_counts().reset_index())
        asf_count.columns = ['unit_link_id', 'asf_total'] #重命名列名

        #合并表格 
        round1 = antigen_count.merge(asf_count, how = 'left', on = 'unit_link_id')

        #合并好的表格中，但凡antigen_total和asf_total都有数据的，那么表示这一轮有检出阳性，记作1，否则就是有检查未检出阳性，记作0
        round1.loc[(round1.antigen_total.notnull()) & (round1.asf_total.notnull()), 'asf_round'] = 1
        round1['asf_round'].fillna(0, inplace = True)

        return round1
    
    #第一轮是否阳性
    asf_round1 = asf_annal_round('2018-10-01', '2019-09-30')
    #第二轮是否阳性
    asf_round2 = asf_annal_round('2019-10-01', '2020-09-30')
    #第三轮是否阳性
    asf_round3 = asf_annal_round('2020-09-30', '2030-09-30')

    database = database.merge(asf_round1.loc[:, ['unit_link_id', 'asf_round']], how = 'left', on = 'unit_link_id')
    database = database.merge(asf_round2.loc[:, ['unit_link_id', 'asf_round']], how = 'left', on = 'unit_link_id')
    database = database.merge(asf_round3.loc[:, ['unit_link_id', 'asf_round']], how = 'left', on = 'unit_link_id')
    
    database_copy = database.copy()
    
    #区分分子分母
    database_copy['numerator'] = database_copy.loc[:, ['asf_round_x', 'asf_round_y','asf_round']].sum(axis = 1) #分子 = 所有送检阳性的轮次
    database_copy['denominator'] = database_copy.loc[:, ['asf_round_x', 'asf_round_y','asf_round']].count(axis = 1) #分母， count() 计算所有非NAN的个数

    #比率计算函数
    def ratio_detail(x): 
        numerator = int(x.numerator)
        denominator = int(x.denominator)
        output = ''
        if denominator == 0: #防止存在分母为0的情况
            output = '0.0'
        else: 
            output = str(round(numerator / denominator,4)*100)
        return output + '%'+'(' + str(numerator) + '/' + str(denominator) + ')'

    #赋值
    database_copy['asf_ratio'] = database_copy.apply(lambda x: ratio_detail(x), axis = 1)

    database['asf_ratio'] = database_copy.asf_ratio
    
    return database.loc[:,['ffieldid', 'fareaid', 'fregionname', 'ffield', 'ffieldtypename',
       'fsegmentid', 'ffeeder_name', 'ffeeder_code', 'funitnames',
       'unit_link_id','asf_ratio']]

def get_asf_sorted(starttime, endtime): 
    '''输出每一轮按照时间序列排列好的送检报告，
    0表示未检出，其他数字表示检出序号'''
    #每一轮的送检表格
    antigen_sql = '''select * from myana.datamining_DWS_DS_antigen_UD where `sampling_time`>='{start}' and `sampling_time`<='{end}' '''
    antigen_sql = antigen_sql.format(start = starttime, end = endtime)
    antigen = pd.read_sql(antigen_sql, hiveConn)
    
    #每一轮的阳性检测表格
    asf_sql = '''select * from myana.datamining_DWS_DS_antigen_UD where `sampling_time`>='{start}' and `sampling_time`<='{end}' and (concat_ws(',',pcrdl_result) LIKE '%A%' OR concat_ws(',',pcr_result) LIKE '%A%') '''
    asf_sql = asf_sql.format(start = starttime, end = endtime)
    asf = pd.read_sql(asf_sql, hiveConn)
    
    #获得单独unit_link_id
    antigen = antigen.merge(batch_info.loc[:, ['fbatchno','funitnames', 'unit_link_id']], how = 'left', left_on = ['batch_no', 'unit'], right_on = ['fbatchno', 'funitnames']) 
    asf = asf.merge(batch_info.loc[:, ['fbatchno','funitnames', 'unit_link_id']], how = 'left', left_on = ['batch_no', 'unit'], right_on = ['fbatchno', 'funitnames']) 
    
    #去除缺失值
    antigen = antigen.loc[antigen.unit_link_id.notnull()]
    asf = asf.loc[asf.unit_link_id.notnull()]
    
    #建立新的表格作为排序好的表
    asf_sorted = asf.copy()
    
    #转化 asf 的sampling_time 为时间格式
    asf_sorted.sampling_time = asf_sorted.sampling_time.astype('datetime64')
    #按照时间排序 并且去重
    asf_sorted = asf_sorted.sort_values(by = 'sampling_time').drop_duplicates(subset=['unit_link_id'], keep='first', inplace=False)
    
    #添加序号字段（升序）
    asf_sorted['sort'] = range(1, asf_sorted.shape[0]+1)
    
    #序列重启
    asf_sorted = asf_sorted.reset_index(drop = True).loc[:,['unit_link_id','sort']]
    
    #现在结合年度送检表，填充序号字段，
    #0表示未检出，其他数字表示检出序号
    antigen = antigen.drop_duplicates(subset=['unit_link_id'], keep='first', inplace=False)
    asf_sorted2 = antigen.merge(asf_sorted, how = 'left', on = 'unit_link_id')
    asf_sorted2['sort'].fillna(0, inplace = True)

    return asf_sorted2.loc[:,['unit_link_id','sort']]


def asf_sort_avg(database): 
    '''输出添加好排序均值字段的数据表格'''
    #第一轮排序
    asf_sorted1 = get_asf_sorted('2018-10-01', '2019-09-30')
    #第二轮排序
    asf_sorted2 = get_asf_sorted('2019-10-01', '2020-09-30')
    #第三轮排序
    asf_sorted3 = get_asf_sorted('2020-09-30', '2030-09-10')
    
    #基础表合并
    database = database.merge(asf_sorted1, how = 'left', on = 'unit_link_id')
    database = database.merge(asf_sorted2, how = 'left', on = 'unit_link_id')
    database = database.merge(asf_sorted3, how = 'left', on = 'unit_link_id')
    
    # 重命名
    database = database.rename(columns = {'sort_x' : 'sort1', 'sort_y': 'sort2', 'sort':'sort3'})
    
    #拷贝
    database_copy = database.copy()
    
    #计算均值
    database_copy['sort_avg'] = database_copy.loc[:,['sort1', 'sort2','sort3']].mean(axis = 1, skipna = True)
    #填充缺失值
    database_copy['sort_avg'].fillna(0, inplace = True)
    #保留两位小数点
    database_copy = database_copy.round({'sort_avg': 2})
    
    #生成字段
    database_copy.loc[:,['sort1', 'sort2','sort3']] = database_copy.loc[:,['sort1', 'sort2','sort3']].fillna('未送检')
    database_copy.loc[:,['sort1', 'sort2','sort3']] = database_copy.loc[:,['sort1', 'sort2','sort3']].replace(to_replace= 0.0, value = '未检出')
    
    database_copy['sort_detail'] = database_copy.sort_avg.astype(str) + '(' + database_copy.sort1.astype(str) +',' +  database_copy.sort2.astype(str) + ',' +  database_copy.sort3.astype(str) + ')'
    
    database['sort_detail'] = database_copy.sort_detail
    
    return database.loc[:,['ffieldid', 'fareaid', 'fregionname', 'ffield', 'ffieldtypename',
       'fsegmentid', 'ffeeder_name', 'ffeeder_code', 'funitnames',
       'unit_link_id','sort_detail']]

def get_feeder_rate(database):
    '''计算饲养员处理的批次发病率'''
    feeder_sql = '''
    SELECT f1.*, f2.asf_batch FROM 
    (
    SELECT ffeeder_code, count(distinct fbatchno) as all_batch FROM 
    (SELECT a.*, b.fbatchno FROM (SELECT DISTINCT ffeeder_code FROM myana.datamining_DWD_BI) a 
    LEFT JOIN myana.datamining_DWD_BI b 
    ON a.ffeeder_code = b.ffeeder_code)
    group by ffeeder_code
    ) f1 
    LEFT JOIN 
    (SELECT A.ffeeder_code, count(distinct fbatchno) as asf_batch
    from (
    SELECT a.*, b.fbatchno FROM (SELECT DISTINCT ffeeder_code FROM myana.datamining_DWD_BI) a 
    LEFT JOIN myana.datamining_DWD_BI b 
    ON a.ffeeder_code = b.ffeeder_code ) A 
    LEFT JOIN myana.datamining_DWS_DS_antigen_UD B 
    on A.fbatchno = B.batch_no 
    WHERE concat_ws(',',pcrdl_result) LIKE '%A%' OR concat_ws(',',pcr_result) LIKE '%A%'
    group by A.ffeeder_code) f2
    ON f1.ffeeder_code = f2.ffeeder_code 
    '''
    feeder_df = pd.read_sql(feeder_sql, hiveConn)
    feeder_df = feeder_df.fillna(0)
    feeder_df['feeder_ratio'] = feeder_df.asf_batch / feeder_df.all_batch
    
    def get_feeder_detail(feeder_df):
        '''输出饲养历史批次发生比例'''
        #四舍五入
        feeder_df = feeder_df.round(4)
        feeder_df['feeder_ratio'] = feeder_df.feeder_ratio.apply(lambda x : format(x,'.2%'))
        #
        feeder_df['feeder_detail'] = feeder_df.feeder_ratio.astype(str) + '(' + feeder_df.asf_batch.astype(str) + '/' + feeder_df.all_batch.astype(str) + ')'
        return feeder_df 
    
    aa = get_feeder_detail(feeder_df)
    database = database.merge(aa.loc[:,['ffeeder_code', 'feeder_detail']], how = 'left', on = 'ffeeder_code')
    
    return database.loc[:,['ffieldid', 'fareaid', 'fregionname', 'ffield', 'ffieldtypename',
       'fsegmentid', 'ffeeder_name', 'ffeeder_code', 'funitnames',
       'unit_link_id','feeder_detail']]

def get_risk_level(database):
    '''输出风险划分等级'''
    d1 = get_asf_count(database)
    d2 = get_asf_rate(database)
    d3 = asf_sort_avg(database)
    d4 = get_feeder_rate(database)
    
    
    database['asf_total'] = d1.asf_total.fillna(0)
    database['asf_ratio'] = d2.asf_ratio
    database['sort_detail'] = d3.sort_detail
    database['feeder_detail'] = d4.feeder_detail
    
    #风险划分
    database = database.dropna(how = 'any') #删除缺失值
    database =  database.reset_index(drop = True)
    database_copy = database.copy()

    #将相应的字段都转变成数字
    database_copy['asf_ratio'] = database.asf_ratio.str.split('%').apply(lambda x: float(x[0])) #百分比已经乘以100了
    database_copy['sort_detail'] = database.sort_detail.str.split('(').apply(lambda x: float(x[0]))
    database_copy['feeder_detail'] = database.feeder_detail.str.split('%').apply(lambda x: float(x[0])) #百分比已经乘以100了

    
    def score_method3(data_part):
        '''计算风险分数
        非瘟发生顺序30%， 历史检测比例40%， 饲养员历史发生比例 20%，该单元历史检出阳性次数 10% '''
        #标准化
        scaler = MinMaxScaler()
        data_part_scalered = data_part.copy()
        data_part_scalered.loc[:,['asf_ratio','sort_detail','feeder_detail','asf_total']] = scaler.fit_transform(data_part_scalered.loc[:,['asf_ratio','sort_detail','feeder_detail','asf_total']])

        data_copy = data_part_scalered.copy()
        score = -data_part_scalered.sort_detail* 0.3 + data_part_scalered.asf_ratio * 0.4 + data_part_scalered.feeder_detail * 0.2 + data_part_scalered.asf_total * 0.1        
        return score
    
    d_notnull = database_copy.loc[(database_copy.asf_ratio != 0)|(database_copy.sort_detail != 0)|(database_copy.feeder_detail != 0)]
    score1 = score_method3(d_notnull)
    
    #得到划分标准
    top30 = np.percentile(score1, 70, axis = 0)
    top65 = np.percentile(score1, 35, axis = 0)
    
    database_copy.loc[(database_copy.asf_ratio != 0) | (database_copy.sort_detail != 0) | (database_copy.feeder_detail != 0), 'score'] = score1
    database_copy.score.fillna(0, inplace = True)

    database_copy.loc[database_copy.score >= top30, 'level'] = '高风险'
    database_copy.loc[(database_copy.score > top65)&(database_copy.score <= top30), 'level'] = '中风险'
    database_copy.loc[(database_copy.score <= top65) & (database_copy.score > 0), 'level'] = '低风险'
    database_copy.loc[database_copy.score == 0, 'level'] = '零风险'
    
    database_copy = database_copy.round({'score': 3})
    
    #更新原始表格
    database['score'] = database_copy.score * 100
    database['level'] = database_copy.level
    
    d_100 = database.sort_values(by = 'score', ascending = False)
    return d_100
    

#输出表格（按照业务方的格式）
def final_table(database, current_time): 
    '''加上当前时间字段'''
    
    database.rename(columns = {'fregionname': '区域名', 'ffield':'场区名', 'ffieldtypename':'场区类型', 
                               'fsegmentid':'工段', 'funitnames':'单元号', 'sort_detail':'非瘟发生平均顺序', 
                               'asf_ratio':'历史阳性检测比率', 'asf_total': '该单元历史检出阳性次数',
                              'ffeeder_name':'饲养员姓名', 'feeder_detail': '饲养员历史批次发生比例', 
                              'score': '风险分数', 'level':'风险等级'}, inplace = True)
        
    
    database['当前时间'] = current_time
    
    finaltable = database.loc[:, ['当前时间', '区域名', '场区名', '场区类型', '工段', '单元号', '非瘟发生平均顺序', '历史阳性检测比率', '该单元历史检出阳性次数', '饲养员姓名', '饲养员历史批次发生比例', '风险分数', '风险等级']]
    
    return finaltable

good_table = final_table(get_risk_level(get_database()), '2021-10-15')
good_table.to_csv('unit211015.csv', index = False)
