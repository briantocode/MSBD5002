import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer
import pickle
import matplotlib.pyplot as plt
import lightgbm as lgb
def restructure(df):
    df = df.sort_values(by =  ['station_id','time'])
#     df.sort_values(by =  ['time'])
    df = df.set_index(['station_id','time'])
    return df
stationId = ['fangshan_aq',
             'daxing_aq',
             'yizhuang_aq',
             'tongzhou_aq',
             'shunyi_aq',
             'pingchang_aq',
             'mentougou_aq',
             'pinggu_aq',
             'huairou_aq',
             'miyun_aq',
             'yanqin_aq',
             'dingling_aq',
             'badaling_aq',
             'miyunshuiku_aq',
             'donggaocun_aq',
             'yongledian_aq',
             'yufa_aq',
             'liulihe_aq',
             'qianmen_aq',
             'yongdingmennei_aq',
             'xizhimenbei_aq',
             'nansanhuan_aq',
             'dongsihuan_aq',
             'dongsi_aq',
             'tiantan_aq',
             'guanyuan_aq',
             'wanshouxigong_aq',
             'aotizhongxin_aq',
             'nongzhanguan_aq',
             'wanliu_aq',
             'beibuxinqu_aq',
             'zhiwuyuan_aq',
             'fengtaihuayuan_aq',
             'yungang_aq',
             'gucheng_aq']
stations = []
a = test_x.copy()
a = restructure(a)
for i in range(len(a.index)):
    temp = a.index[i]
    temp = temp[0]+'#'+str(temp[1])
    stations.append(temp)
filename_1 = '../data/model/LGBM_model_PM25_1.sav'
filename_2 = '../data/model/LGBM_model_PM10_1.sav'
filename_3 = '../data/model/LGBM_model_O3_1.sav'
LGBM_model_PM25_1 = pickle.load(open(filename_1, 'rb'))
LGBM_model_PM10_1 = pickle.load(open(filename_2, 'rb'))
LGBM_model_O3_1 = pickle.load(open(filename_3, 'rb'))
features_PM25 = ['ob_w', 'h', 'p', 't', 'ws','time','mean_PM2.5']
features_PM10 = ['ob_w', 'h', 'p', 't', 'ws','time','mean_PM10']
features_O3 = ['ob_w', 'h', 'p', 't', 'ws','time','mean_O3']


test_x = pd.read_csv('../data/test_by_station/all_stations.csv')
test_x = test_x.drop('time',axis = 1)
test_x = test_x.rename({'day_time':'time'})
stations_id_hour = []
a = test_x.copy()
a = restructure(a)
for i in range(len(a.index)):
    temp = a.index[i]
    temp = temp[0]+'#'+str(temp[1])
    stations_id_hour.append(temp)
 # use the ave of last 48 hours in train set to estimate the two days in test set
#load the train set
file_path = '../data/train_by_station/allstation.csv'
train_df = pd.read_csv(file_path)
#restructure it 
train_df = restructure(train_df)
LGBM_test = test_x.copy()
LGBM_test = restructure(LGBM_test)
#  add new feature
for Id in stationId:
    temp = train_df.loc[Id,'PM2.5'][0:-48]
    temp = temp.mean()
#     print(temp)
    LGBM_test.loc[Id,'mean_PM2.5'] =temp
for Id in stationId:
    temp = train_df.loc[Id,'PM10'][0:-48]
    temp = temp.mean()
#     print(temp)
    LGBM_test.loc[Id,'mean_PM10'] =temp
for Id in stationId:
    temp = train_df.loc[Id,'O3'][0:-48]
    temp = temp.mean()
#     print(temp)
    LGBM_test.loc[Id,'mean_O3'] =temp
LGBM_pre_PM25 = LGBM_model_PM25_1.predict(LGBM_test[features_PM25])
LGBM_pre_PM10 = LGBM_model_PM10_1.predict(LGBM_test[features_PM10])
LGBM_pre_O3 = LGBM_model_O3_1.predict(LGBM_test[features_O3])
stations_id_hour= pd.Series(stations_id_hour)
LGBM_pre_PM25 = pd.Series(LGBM_pre_PM25)
LGBM_pre_PM10 = pd.Series(LGBM_pre_PM10)
LGBM_pre_O3 = pd.Series(LGBM_pre_PM25)
LGBM_pre_df =  pd.concat([stations_id_hour, LGBM_pre_PM25,LGBM_pre_PM10,LGBM_pre_O3 ], axis=1)
LGBM_pre_df = LGBM_pre_df.rename(index = str,columns = {0:'station_id',
                                                        1:'PM2.5',2:'PM10',3:'O3'})
LGBM_pre_df.to_csv('../data/prediction/group53_submission.csv')
# still need to use convert.ipynb but until now we got the final answer