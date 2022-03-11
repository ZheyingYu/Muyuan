import sys
import re
from pyhive import hive
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import scipy.stats as stats
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
import requests
import json
import time
from matplotlib import pyplot as plt

from matplotlib.font_manager import FontProperties
font = FontProperties('simhei', size=20) #中文字体


pd.set_option('display.max_columns', None) #显示所有列
sys.path.append('/data/datamining/Py_functions/get_data')
from get_data_func import * ##取数function  


# 字段
# 脚本
# 所有批次基础表
hiveConn = hive.Connection(host='10.106.20.15', port=10000, username='szchenye', password='szcy230#',
                                               database='mydw', auth='CUSTOM')

batchbase_level_sql = '''SELECT b.*, a.fbatchno, a.fsegmentid FROM  myana.datamining_DWD_BI a 
INNER JOIN (SELECT ffieldid, ffield, fareaid, farea, fregionname, fbigregionname FROM myana.datamining_DIM_AF) b 
ON a.ffieldid = b.ffieldid WHERE a.fstate = 1 '''

batchbase_level_df = pd.read_sql(batchbase_level_sql,hiveConn)


#该场区母猪来源涉及场区个数
mom_source_fieldnum_sql = '''SELECT t.to_fbatchno,  t.to_ffieldid, t.to_segmentid, count(DISTINCT t.ffieldid) AS source_fieldnum FROM (
SELECT fbatchno, to_fbatchno, ffieldid, to_ffieldid, to_segmentid FROM myana.ybb_batch_retrospect_change_08
WHERE to_segmentid IN (13910, 13909, 21254,29535, 13908 )
AND last_changedate BETWEEN '2021-12-15' AND '2022-01-15') t 
GROUP BY t.to_fbatchno, t.to_ffieldid , t.to_segmentid '''

mom_source_fieldnum_df = pd.read_sql(mom_source_fieldnum_sql,hiveConn)

#该场区后备猪批次进群前是否检出蓝耳抗原
#筛选出后备批次在这一个月内检出蓝耳的场区
laner_houbei_batch = pcr_bi_rawdata.loc[(pcr_bi_rawdata.采样时间 >= '2021-12-15')&(pcr_bi_rawdata.采样时间 <= '2022-01-15')&(pcr_bi_rawdata.工段 == '后备')&(pcr_bi_rawdata.送检目的 == '转群监控')]
laner_houbei_batch['laner_houbei'] = 1
laner_houbei_batch = laner_houbei_batch.drop_duplicates(subset = ['批次编号'])

#一胎怀孕母猪批次（后备）配种的场区类型
houbei_field_sql = '''
SELECT DISTINCT A.fbatchno, B.ffieldtypename AS 配种场区id FROM (
SELECT a.fbatchno, a.ffieldid, a.to_ffieldid FROM myana.ybb_batch_retrospect_change_08 a 
inner JOIN 
(SELECT * FROM myana.datamining_DWD_BI WHERE fstate = 1 AND fsegmentid = 13908) b 
ON   a.fbatchno = b.fbatchno AND a.ffieldid = b.ffieldid 
WHERE a.to_segmentid = 29535 ) A 
LEFT JOIN  myana.datamining_DIM_AF B  ON A.to_ffieldid = B.ffieldid '''

houbei_field_df = pd.read_sql(houbei_field_sql,hiveConn)
#大圈的怀孕批次
dajuan_batch = houbei_field_df.loc[~(houbei_field_df.配种场区id == '繁殖场')]
dajuan_batch['大圈配种怀孕批次']='是'
dajuan_batch = dajuan_batch.loc[:, ['fbatchno', '大圈配种怀孕批次']].drop_duplicates()

# 待配和怀孕批次
daipei_batch_sql = ''' SELECT DISTINCT A.fbatchno, B.ffieldtypename AS 配种场区id FROM (
SELECT a.fbatchno, b.fbatchno AS source_fbatchno , b.fsegmentid, b.to_segmentid ,b.to_ffieldid AS 待配场区ID FROM (
SELECT * FROM myana.datamining_DWD_BI WHERE fstate = 1 AND fsegmentid in (13909,29535) ) a 
INNER JOIN myana.ybb_batch_retrospect_change_08 b ON a.fbatchno = b.to_fbatchno WHERE b.fsegmentid = 13908) A 
LEFT JOIN myana.datamining_DIM_AF B  ON A.待配场区ID = B.ffieldid '''
daipei_batch_df = pd.read_sql(daipei_batch_sql,hiveConn)

#大圈的怀孕批次
dajuan_batch2 = daipei_batch_df.loc[~(daipei_batch_df.配种场区id == '繁殖场')]
dajuan_batch2['大圈配种怀孕批次']='是'
dajuan_batch2 = dajuan_batch2.loc[:, ['fbatchno', '大圈配种怀孕批次']].drop_duplicates()

# 其他母猪批次，需要追述到第一次怀孕的地点（也就是13908后备 转到29535待配时，待配的场区类型）
other_mom_batch_sql = '''SELECT DISTINCT abc.fbatchno, f.ffieldtypename AS 配种场区id FROM (
SELECT DISTINCT t.fbatchno, t.source2_fieldid AS 待配场区id FROM (
SELECT aaa.*, bbb.fbatchno AS source3_fbatchno,bbb.fsegmentid AS source3_segmentid, bbb.ffieldid AS source3_fieldid FROM(
SELECT aa.*, bb.fbatchno AS source_source_fbatchno, bb.fsegmentid AS source2_segmentid, bb.ffieldid AS source2_fieldid FROM (
SELECT a.fbatchno, b.fbatchno AS source_fbatchno , b.fsegmentid AS source_segmentid , b.to_segmentid, b.ffieldid AS source_fieldid FROM (
SELECT * FROM myana.datamining_DWD_BI WHERE fstate = 1 AND fsegmentid in (13910,21254)) a 
INNER JOIN myana.ybb_batch_retrospect_change_08 b ON a.fbatchno = b.to_fbatchno ) aa 
LEFT JOIN  myana.ybb_batch_retrospect_change_08 bb ON aa.source_fbatchno = bb.to_fbatchno ) aaa 
LEFT JOIN myana.ybb_batch_retrospect_change_08 bbb ON aaa.source_source_fbatchno = bbb.to_fbatchno ) t 
WHERE t.source2_segmentid = 29535 AND t.source3_segmentid = 13908) abc 
LEFT JOIN  myana.datamining_DIM_AF f ON abc.待配场区id = f.ffieldid '''

other_mom_batch = pd.read_sql(other_mom_batch_sql,hiveConn)
dajuan_batch3 = other_mom_batch.loc[~(other_mom_batch.配种场区id == '繁殖场')]
dajuan_batch3['大圈配种怀孕批次']='是'
dajuan_batch3 = dajuan_batch3.loc[:, ['fbatchno', '大圈配种怀孕批次']].drop_duplicates()

#一胎怀孕母猪批次配种的场区类型
# 保育---该保育批次是否是大圈配种母猪分娩后代
baoyu_mom_batch_sql = '''SELECT DISTINCT t.fbatchno, t.配种场区id FROM (
SELECT DISTINCT  a.fbatchno, b.fsegmentid AS source_segmentid , b.to_segmentid, 
bb.fsegmentid AS source2_segmentid, bbb.fsegmentid AS source3_segmentid, f.ffieldtypename AS 配种场区id
FROM (
SELECT * FROM myana.datamining_DWD_BI WHERE fstate = 1 AND fsegmentid = 13906) a 
INNER JOIN myana.ybb_batch_retrospect_change_08 b ON a.fbatchno = b.to_fbatchno
LEFT JOIN myana.ybb_batch_retrospect_change_08 bb ON b.fbatchno = bb.to_fbatchno
INNER JOIN myana.ybb_batch_retrospect_change_08 bbb ON bb.fbatchno = bbb.to_fbatchno 
INNER JOIN  myana.datamining_DIM_AF f ON bbb.ffieldid = f.ffieldid ) t 
WHERE t.source_segmentid = 13905 AND t.to_segmentid = 13906 AND t.source2_segmentid = 13910 AND t.source3_segmentid = 13909 '''

baoyu_mom_batch = pd.read_sql(baoyu_mom_batch_sql,hiveConn)

# 添加字段：是否是大圈配种怀孕批次
dajuan_batch4 = baoyu_mom_batch.loc[~(baoyu_mom_batch.配种场区id == '繁殖场')]
dajuan_batch4['大圈配种怀孕批次']='是'
dajuan_batch4 = dajuan_batch4.loc[:, ['fbatchno', '大圈配种怀孕批次']].drop_duplicates()

yufei_mom_batch_sql = '''SELECT DISTINCT  t.fbatchno, t.配种场区id FROM (
SELECT DISTINCT  a.fbatchno, b.fsegmentid AS source_segmentid , b.to_segmentid, 
bb.fsegmentid AS source2_segmentid, bbb.fsegmentid AS source3_segmentid, bbbb.fsegmentid AS source4_segmentid, f.ffieldtypename AS 配种场区id
FROM (
SELECT * FROM myana.datamining_DWD_BI WHERE fstate = 1 AND fsegmentid = 13907) a 
INNER JOIN myana.ybb_batch_retrospect_change_08 b ON a.fbatchno = b.to_fbatchno
LEFT JOIN myana.ybb_batch_retrospect_change_08 bb ON b.fbatchno = bb.to_fbatchno
INNER JOIN myana.ybb_batch_retrospect_change_08 bbb ON bb.fbatchno = bbb.to_fbatchno 
INNER JOIN myana.ybb_batch_retrospect_change_08 bbbb ON bbb.fbatchno = bbbb.to_fbatchno 
INNER JOIN  myana.datamining_DIM_AF f ON bbbb.ffieldid = f.ffieldid ) t 
WHERE t.source_segmentid = 13906 AND t.to_segmentid = 13907 AND t.source2_segmentid = 13905 AND t.source3_segmentid = 13910 AND t.source4_segmentid = 13909 '''

yufei_mom_batch = pd.read_sql(yufei_mom_batch_sql,hiveConn)

# 添加字段：是否是大圈配种怀孕批次
dajuan_batch5 = yufei_mom_batch.loc[~(yufei_mom_batch.配种场区id == '繁殖场')]
dajuan_batch5['大圈配种怀孕批次']='是'
dajuan_batch5 = dajuan_batch5.loc[:, ['fbatchno', '大圈配种怀孕批次']].drop_duplicates()

#总和所有工段
dajuan_batch_total = dajuan_batch.append(dajuan_batch2, ignore_index = True).append(dajuan_batch3, ignore_index = True).append(dajuan_batch4, ignore_index = True).append(dajuan_batch5, ignore_index = True)


laner_huaiyun_batch = pcr_bi_rawdata.loc[(pcr_bi_rawdata.采样时间 >= '2021-12-15')&(pcr_bi_rawdata.采样时间 <= '2022-01-15')&(pcr_bi_rawdata.批次编号.notnull())&(pcr_bi_rawdata.工段 == '怀孕')&(pcr_bi_rawdata.批次编号 != '/')]
laner_huaiyun_batch['laner_huaiyun'] = 1

 #蓝耳抗原数量
laner_num_array = laner_huaiyun_batch.loc[:, ['批次编号', 'laner_huaiyun']].groupby(['批次编号']).sum().laner_huaiyun.values
# 创建新的dataframe 
base_df_feature4 = pd.DataFrame(columns = ['fbatchno', 'laner_huaiyun'])
base_df_feature4.fbatchno = laner_huaiyun_batch.批次编号.unique()
base_df_feature4.laner_huaiyun = laner_num_array
base_df_feature4 = base_df_feature4.loc[base_df_feature4.laner_huaiyun > 1]

#该场区后备母猪批次驯化隔离期是否<60
houbei_gap_sql = '''SELECT  t.*, datediff(t.last_changedate, t.fstarttime) AS day_gap FROM (
SELECT a.fbatchno, a.last_changedate, a.ffieldid, b.fstarttime FROM myana.ybb_batch_retrospect_change_08 a 
inner JOIN 
(SELECT * FROM myana.datamining_DWD_BI WHERE fstate = 1 AND fsegmentid = 13908) b 
ON   a.fbatchno = b.fbatchno AND a.ffieldid = b.ffieldid 
WHERE a.to_segmentid = 29535 ) t '''

houbei_gap_df = pd.read_sql(houbei_gap_sql,hiveConn)

# 加一步骤，是否大于60做对比，有些批次很多次转群，只要有一次小于60即为是
houbei_gap60 = houbei_gap_df.loc[houbei_gap_df.day_gap < 60, ['fbatchno','ffieldid']].drop_duplicates(subset = ['fbatchno','ffieldid'])
houbei_gap60['day_gap_60'] = 1

#该保育批次是否是大圈配种母猪分娩后代 = dajuan_batch4
baoyu_mom_dajuan = dajuan_batch4.rename(columns = {'大圈配种怀孕批次':'保育批次是否是大圈配种母猪分娩后代'})

#该场区产房批次是否检出蓝耳病原
laner_changfang_batch = pcr_bi_rawdata.loc[(pcr_bi_rawdata.采样时间 >= '2021-12-15')&(pcr_bi_rawdata.采样时间 <= '2022-01-15')&(pcr_bi_rawdata.批次编号.notnull())&((pcr_bi_rawdata.工段 == '泌乳母猪')|(pcr_bi_rawdata.工段 == '哺乳仔猪'))&(pcr_bi_rawdata.批次编号 != '/')]
laner_changfang_batch['laner_changfang'] = 1
laner_changfang_batch = laner_changfang_batch.loc[:, ['批次编号', 'laner_changfang']].groupby('批次编号').sum().reset_index()

#保育批次猪只产房来源个数
baoyu_chanfang_num_sql = '''SELECT t.fbatchno, count(t.source_batch) AS source_chanfang_num FROM (
SELECT a.fbatchno, b.fbatchno AS source_batch FROM (
SELECT fbatchno, fsegmentid FROM  myana.datamining_DWD_BI WHERE fstate = 1 AND fsegmentid = 13906) a
LEFT JOIN myana.ybb_batch_retrospect_change_08 b ON a.fbatchno = b.to_fbatchno 
WHERE b.fsegmentid = 13905) t GROUP BY t.fbatchno '''

baoyu_chanfang_num = pd.read_sql(baoyu_chanfang_num_sql,hiveConn)

#怀孕批次流产率
abortion_rate_sql = ''' SELECT B.fbatchno, A.abortion_rate FROM (
SELECT foutbatch,
SUM(CASE WHEN pregnresult in ('流产') THEN 1 ELSE 0 END)/SUM(CASE WHEN pregnresult in ('流产','返情','检空','空怀') THEN 1 ELSE 0 END) AS abortion_rate
FROM mydw.source_my_sc_pregnancyinfo WHERE pregndate BETWEEN '2021-12-15' AND '2022-01-15'
GROUP BY foutbatch) A 
INNER JOIN (SELECT fbatchno, ffieldid FROM myana.datamining_DWD_BI WHERE fstate = 1 AND fsegmentid = 13909) B ON A.foutbatch = B.fbatchno '''

abortion_rate_df = pd.read_sql(abortion_rate_sql,hiveConn)

other_batch_abortion_sql = '''SELECT T.fbatchno, avg(T.abortion_rate) AS abortion_rate FROM (
SELECT DISTINCT B.fbatchno, A.abortion_rate FROM (
SELECT foutbatch,
SUM(CASE WHEN pregnresult in ('流产') THEN 1 ELSE 0 END)/SUM(CASE WHEN pregnresult in ('流产','返情','检空','空怀') THEN 1 ELSE 0 END) AS abortion_rate
FROM mydw.source_my_sc_pregnancyinfo
GROUP BY foutbatch) A 
INNER JOIN 
(SELECT a.fbatchno, b.fbatchno AS source_fbatchno , b.fsegmentid AS source_segmentid , b.to_segmentid, b.ffieldid AS source_fieldid FROM (
SELECT * FROM myana.datamining_DWD_BI WHERE fstate = 1 AND fsegmentid in (13910,21254)) a 
INNER JOIN myana.ybb_batch_retrospect_change_08 b ON a.fbatchno = b.to_fbatchno 
WHERE b.fsegmentid = 13909) B
ON A.foutbatch = B.source_fbatchno) T GROUP BY T.fbatchno'''
other_batch_abortion = pd.read_sql(other_batch_abortion_sql,hiveConn)

# 保育和育肥段怀孕流产率
baoyu_mom_abortion_sql = '''SELECT T.fbatchno, AVG(T.abortion_rate) AS abortion_rate FROM (
SELECT DISTINCT B.fbatchno, A.abortion_rate FROM (
SELECT foutbatch,
SUM(CASE WHEN pregnresult in ('流产') THEN 1 ELSE 0 END)/SUM(CASE WHEN pregnresult in ('流产','返情','检空','空怀') THEN 1 ELSE 0 END) AS abortion_rate
FROM mydw.source_my_sc_pregnancyinfo
GROUP BY foutbatch) A 
INNER JOIN (
SELECT DISTINCT t.fbatchno, t.怀孕批次号 FROM (
SELECT a.fbatchno , b.fsegmentid AS source_segmentid , b.to_segmentid,
bb.fsegmentid AS source2_segmentid, bbb.fsegmentid AS source3_segmentid , bbb.fbatchno AS 怀孕批次号  FROM (
SELECT * FROM myana.datamining_DWD_BI WHERE fstate = 1 AND fsegmentid = 13906) a 
INNER JOIN myana.ybb_batch_retrospect_change_08 b ON a.fbatchno = b.to_fbatchno 
INNER JOIN myana.ybb_batch_retrospect_change_08 bb ON b.fbatchno = bb.to_fbatchno 
INNER JOIN myana.ybb_batch_retrospect_change_08 bbb ON bb.fbatchno = bbb.to_fbatchno) t 
WHERE t.source_segmentid = 13905 AND t.to_segmentid = 13906 AND t.source2_segmentid = 13910 AND t.source3_segmentid = 13909) B 
ON A.foutbatch = B.怀孕批次号) T 
GROUP BY T.fbatchno '''

baoyu_mom_abortion = pd.read_sql(baoyu_mom_abortion_sql,hiveConn)

# 育肥段
yufei_mom_abortion_sql = '''SELECT T.fbatchno, avg(T.abortion_rate) AS abortion_rate FROM (
SELECT DISTINCT B.fbatchno, A.abortion_rate FROM (
SELECT foutbatch,
SUM(CASE WHEN pregnresult in ('流产') THEN 1 ELSE 0 END)/SUM(CASE WHEN pregnresult in ('流产','返情','检空','空怀') THEN 1 ELSE 0 END) AS abortion_rate
FROM mydw.source_my_sc_pregnancyinfo
GROUP BY foutbatch) A 
INNER JOIN (
SELECT a.fbatchno , b.fsegmentid AS source_segmentid , b.to_segmentid,
bb.fsegmentid AS source2_segmentid, bbb.fsegmentid AS source3_segmentid, bbbb.fsegmentid AS source4_segmentid, bbbb.fbatchno AS 怀孕批次号 FROM (
SELECT * FROM myana.datamining_DWD_BI WHERE fstate = 1 AND fsegmentid = 13907) a 
INNER JOIN myana.ybb_batch_retrospect_change_08 b ON a.fbatchno = b.to_fbatchno 
INNER JOIN myana.ybb_batch_retrospect_change_08 bb ON b.fbatchno = bb.to_fbatchno 
INNER JOIN myana.ybb_batch_retrospect_change_08 bbb ON bb.fbatchno = bbb.to_fbatchno
INNER JOIN myana.ybb_batch_retrospect_change_08 bbbb ON bbb.fbatchno = bbbb.to_fbatchno ) B 
ON A.foutbatch = B.怀孕批次号) T 
GROUP BY T.fbatchno '''

yufei_mom_abortion = pd.read_sql(yufei_mom_abortion_sql,hiveConn)

#所有工段的怀孕流产率

abortion_df_total =abortion_rate_df.append(other_batch_abortion, ignore_index = True).append(baoyu_mom_abortion, ignore_index = True).append(yufei_mom_abortion, ignore_index = True)

# 首先填充某些污染源批次和其字段
batchbase_value_df = batchbase_level_df.copy()
#字段1 
batchbase_value_1 = batchbase_value_df.merge(mom_source_fieldnum_df.loc[:, ['to_fbatchno', 'source_fieldnum']], how = 'left', left_on = 'fbatchno', right_on = 'to_fbatchno')

#字段2 
batchbase_value_2 = batchbase_value_1.merge(laner_houbei_batch.loc[:, ['批次编号', 'laner_houbei']], how = 'left', left_on = 'fbatchno', right_on = '批次编号')

#字段3
batchbase_value_3 = batchbase_value_2.merge(dajuan_batch_total.loc[:, ['fbatchno', '大圈配种怀孕批次']], how = 'left', on = 'fbatchno')

#字段4
batchbase_value_4 = batchbase_value_3.merge(base_df_feature4, how = 'left', on = 'fbatchno')

#字段5
batchbase_value_5 = batchbase_value_4.merge(houbei_gap60.loc[:, ['fbatchno', 'day_gap_60']], how = 'left', on = 'fbatchno')

#字段6 
batchbase_value_6 = batchbase_value_5.merge(baoyu_mom_dajuan.loc[:, ['fbatchno', '保育批次是否是大圈配种母猪分娩后代']], how = 'left', on = 'fbatchno')

# 字段7
batchbase_value_7 = batchbase_value_6.merge(laner_changfang_batch.loc[:, ['批次编号', 'laner_changfang']], how = 'left', left_on = 'fbatchno', right_on = '批次编号')

#字段8 
batchbase_value_8 = batchbase_value_7.merge(baoyu_chanfang_num, how = 'left', on = 'fbatchno')

#字段9 
batchbase_value_9 = batchbase_value_8.merge(abortion_df_total.loc[:, ['fbatchno', 'abortion_rate']], how = 'left', on = 'fbatchno')

#添加字段=当前时间
batchbase_value_9['当前时间'] = currenttime=time.strftime('%Y-%m-%d',time.localtime(time.time())) #当前时间-年月日

batchbase_value = batchbase_value_9.loc[:, ['当前时间','ffieldid', 'ffield', 'fareaid', 'farea', 'fregionname', 'fbigregionname', 'fbatchno', 'fsegmentid', 'source_fieldnum', 'laner_houbei', '大圈配种怀孕批次', 'laner_huaiyun', 
                                           'day_gap_60', '保育批次是否是大圈配种母猪分娩后代', 'laner_changfang', 'source_chanfang_num', 'abortion_rate']]

#找到污染源
pollution_batch = batchbase_value.dropna(subset = ['source_fieldnum', 'laner_houbei', '大圈配种怀孕批次', 'laner_huaiyun', 
                                           'day_gap_60', '保育批次是否是大圈配种母猪分娩后代', 'laner_changfang', 'source_chanfang_num', 'abortion_rate'], how = 'all').index 
#对污染源添加字段 pollution_batch
batchbase_value.loc[pollution_batch , '是否污染源'] = '是'

#划分等级, 1,2,3等级分别对应 高风险，中风险，低风险
batchbase_value_copy = batchbase_value.copy()

#该场区后备母猪来源涉及场区个数>1， 所有场区高风险
highrisk_field_1 = list(batchbase_value.loc[batchbase_value.source_fieldnum > 1].ffieldid.unique()) #高风险场区列表
batchbase_value_copy.loc[batchbase_value.ffieldid.isin(highrisk_field_1), 'risk_level1'] = 1
batchbase_value_copy.loc[batchbase_value.ffieldid.isin(highrisk_field_1),  'reason1']= '该场区后备母猪来源涉及场区个数>1'

#该场区后备猪进群前检出蓝耳抗原， 所有场区高风险
highrisk_field_2 = list(batchbase_value.loc[batchbase_value.laner_houbei.notnull()].ffieldid.unique()) 
batchbase_value_copy.loc[batchbase_value.ffieldid.isin(highrisk_field_2), 'risk_level2'] = 1
batchbase_value_copy.loc[batchbase_value.ffieldid.isin(highrisk_field_2), 'reason2'] = '该场区后备猪进群前检出蓝耳抗原'

#一胎怀孕母猪是大圈配种的批次, 仅指这个批次
batchbase_value_copy.loc[batchbase_value.大圈配种怀孕批次.notnull(), 'risk_level3'] = 1
batchbase_value_copy.loc[batchbase_value.大圈配种怀孕批次.notnull(),  'reason3'] = '该批次是一胎怀孕母猪是大圈配种的批次'

#该场区怀孕批次检出蓝耳抗原个数>1， 所有场区中风险
highrisk_field_4 = list(batchbase_value.loc[batchbase_value.laner_huaiyun.notnull()].ffieldid.unique())
batchbase_value_copy.loc[batchbase_value.ffieldid.isin(highrisk_field_4), 'risk_level4'] = 2
batchbase_value_copy.loc[batchbase_value.ffieldid.isin(highrisk_field_4), 'reason4'] = '该场区怀孕批次检出蓝耳抗原个数>1'

#该场区后备母猪隔离期<60， 所有场区中风险
highrisk_field_5 = list(batchbase_value.loc[batchbase_value.day_gap_60.notnull()].ffieldid.unique())
batchbase_value_copy.loc[batchbase_value.ffieldid.isin(highrisk_field_5), 'risk_level5'] = 2
batchbase_value_copy.loc[batchbase_value.ffieldid.isin(highrisk_field_5), 'reason5'] = '该场区后备母猪隔离期<60'

#该保育批次是大圈配种母猪分娩后代，仅指这个批次
batchbase_value_copy.loc[batchbase_value.保育批次是否是大圈配种母猪分娩后代.notnull(), 'risk_level6'] = 1
batchbase_value_copy.loc[batchbase_value.保育批次是否是大圈配种母猪分娩后代.notnull(),  'reason6'] = '该保育批次是大圈配种母猪分娩后代'

#该产房批次病原检出批次个数>1
batchbase_value_copy.loc[batchbase_value.laner_changfang > 1, 'risk_level7'] = 2
batchbase_value_copy.loc[batchbase_value.laner_changfang > 1, 'reason7'] = '该产房批次病原检出批次个数>1'

#保育批次接猪混群单元>=2(产房单元>=2)，仅指这个批次
batchbase_value_copy.loc[batchbase_value.source_chanfang_num >= 2, 'risk_level8'] = 2
batchbase_value_copy.loc[batchbase_value.source_chanfang_num >= 2,  'reason8'] = '保育批次接猪混群单元>=2(产房单元>=2)'

#怀孕批次流产率>3% = 高风险, 1.5%<怀孕批次流产率≤3% = 中风险, 怀孕批次流产率≤1.5% = 低风险
batchbase_value_copy.loc[batchbase_value.abortion_rate > 0.03, 'risk_level9'] = 1
batchbase_value_copy.loc[batchbase_value.abortion_rate > 0.03,  'reason9'] = '怀孕批次流产率>3%'

batchbase_value_copy.loc[(batchbase_value.abortion_rate <= 0.03)&(batchbase_value.abortion_rate > 0.015), 'risk_level7'] = 2
batchbase_value_copy.loc[(batchbase_value.abortion_rate <= 0.03)&(batchbase_value.abortion_rate > 0.015), 'reason7'] = '1.5%<怀孕批次流产率≤3%'

batchbase_value_copy.loc[batchbase_value.abortion_rate <= 0.015, 'risk_level9'] = 3
batchbase_value_copy.loc[batchbase_value.abortion_rate <= 0.015, 'reason9'] = '怀孕批次流产率≤1.5%'

#该场区后备母猪来源涉及场区个数 = 1 低风险
highrisk_field_6 = list(batchbase_value.loc[batchbase_value.source_fieldnum == 1].ffieldid.unique()) #风险场区列表
batchbase_value_copy.loc[batchbase_value.ffieldid.isin(highrisk_field_6), 'risk_level10'] = 3
batchbase_value_copy.loc[batchbase_value.ffieldid.isin(highrisk_field_6),  'reason10'] = '该场区后备母猪来源涉及场区个数=1'

#risk_level所有数值权为空则为 ’无法判断“等级
batchbase_value_nonnull = batchbase_value_copy.dropna(subset = ['risk_level1','risk_level2', 'risk_level3', 'risk_level4', 'risk_level5', 'risk_level6', 'risk_level7', 'risk_level8', 'risk_level9', 'risk_level10'], how = 'all')

# 每一行第一个非空数字即为风险等级
risk_values = batchbase_value_nonnull.loc[:, ['risk_level1','risk_level2', 'risk_level3', 'risk_level4', 'risk_level5', 'risk_level6', 'risk_level7', 'risk_level8','risk_level9', 'risk_level10']].values 
batchbase_value_nonnull["risk_level_no"] = [risk_values[x][np.where(~np.isnan(risk_values[x]))][0] for x in range(len(batchbase_value_nonnull))]
risk_dict = {1:'高风险', 2:'中风险', 3:'低风险'} #风险等级字典
batchbase_value_nonnull['风险等级'] = batchbase_value_nonnull.risk_level_no.replace(risk_dict)
#填充到原来的基础表，index不变
batchbase_value_copy['风险等级'] = batchbase_value_nonnull['风险等级']
#其他没有划分等级的批次, 填充为”无法判断“
batchbase_value_copy.风险等级 = batchbase_value_copy.风险等级.fillna('无法判断')

#等级划分原因罗列
reason_list = ['reason1', 'reason2', 'reason3', 'reason4', 'reason5', 'reason6', 'reason7', 'reason8', 'reason8', 'reason9', 'reason10']
batchbase_value_copy['划分原因'] = batchbase_value_copy.loc[:,reason_list].apply(lambda x: ','.join(x.dropna()), axis = 1)
batchbase_value.loc[:, '风险等级'] = batchbase_value_copy.风险等级
batchbase_value.loc[:, '划分原因'] = batchbase_value_copy.划分原因

# 工段
segment_dict = {13910:'泌乳母猪', 13909:'怀孕', 13905:'哺乳仔猪',13906:'保育', 13907:'育肥', 21254: '空怀', 29535:'待配', 13908:'后备'}

batchbase_value.fsegmentid = batchbase_value.fsegmentid.replace(segment_dict)
batchbase_value.是否污染源 = batchbase_value.是否污染源.fillna('否')

#改成保存格式
save_batchvalue_df = batchbase_value.rename(columns = {'ffieldid':'所在场区ID', 'ffield':'所在场区名', 'fareaid':'所在子公司id','farea':'所在子公司名', 'fregionname': '区域', 'fbigregionname': '大区',
                                                       'fbatchno':'批次号', 'fsegmentid':'工段','source_fieldnum':'该场区母猪来源涉及场区个数', 'laner_houbei':'该场区后备猪批次进群前是否检出蓝耳抗原', 
                                                        'laner_huaiyun':'该场区怀孕批次检出蓝耳抗原个数', 
                                                       'day_gap_60':'该场区后备母猪批次驯化隔离期是否<60', 'laner_changfang':'该场区产房批次是否检出蓝耳病原','source_chanfang_num':'保育批次猪只产房来源个数', 'abortion_rate':'怀孕批次流产率'})
