import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer
import pickle
import matplotlib.pyplot as plt
import lightgbm as lgb
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
features = ['ob_w', 'h', 'p', 't', 'ws','time']
label = ['PM2.5', 'PM10', 'O3']
features_PM25 = ['ob_w', 'h', 'p', 't', 'ws','time','mean_PM2.5']
features_PM10 = ['ob_w', 'h', 'p', 't', 'ws','time','mean_PM10']
features_O3 = ['ob_w', 'h', 'p', 't', 'ws','time','mean_O3']

file_path = '../data/train_by_station/allstation.csv'

#some preprocessing of X, all traing. X is from. X_df
def preprocessing_all(file_path):
      X_df = pd.read_csv(file_path)
      X_df.dropna(inplace=True)
      X_df['time'] = X_df['time'].apply(lambda x:x%24)
      return X_df
#X_GBM is a training set includes rolling mean, choose different mean to train different model
def rolling_mean(X_df):
      station_list_1 = []
      X_df_raw  =  X_df.copy()
      for Id in stationId:
          file_path_1 = '../data/train_by_station/after_fill_missing_value/'+Id+'.csv'

          df = pd.read_csv(file_path_1)
          df['mean_PM2.5'] = df['PM2.5'].rolling(48).mean()
          df['mean_PM10'] = df['PM10'].rolling(48).mean()
          df['mean_O3'] = df['O3'].rolling(48).mean()
          df.dropna(inplace=True)
          # df['time'] = X_df_raw['time'].apply(lambda x:x%24)
          df['station_id'] = Id
          station_list_1.append(df)
      X_df_GBM=  pd.concat(station_list_1)
      return X_df_GBM
# Three model, the parameters is from grid search
gmb_PM25=lgb.LGBMRegressor(learning_rate = 0.01,
                         max_depth = 10 ,
                         num_leaves= 400, 
                         subsamples  = 0.8, 
                         seed = 3,
                         num_iterations = 1000
                            )
gmb_PM10=lgb.LGBMRegressor(learning_rate = 0.01,
                         max_depth = 10 ,
                         num_leaves= 400, 
                         subsamples  = 0.8, 
                         seed = 3,
                         num_iterations = 900
                            )
gmb_O3=lgb.LGBMRegressor(learning_rate = 0.01,
                         max_depth = 10 ,
                         num_leaves= 400, 
                         subsamples  = 0.8, 
                         seed = 3,
                         num_iterations = 1200
                            )
if __name__=="__main__":  
      X_df=preprocessing_all(file_path)
      X_df_GBM = rolling_mean(X_df)

      X_PM25 = X_df_GBM[features_PM25]
      X_PM10 = X_df_GBM[features_PM10]
      X_PMO3 = X_df_GBM[features_O3]

      y_PM25 = X_df_GBM[label[0]]
      y_PM10 = X_df_GBM[label[1]]
      y_O3= X_df_GBM[label[2]]

      gmb_PM25.fit(X_PM25,y_PM25)
      filename = 'LGBM_model_PM25_1.sav'
      pickle.dump(gmb_O3, open(filename, 'wb'))

      gmb_PM10.fit(X_PM25,y_PM25)
      filename = 'LGBM_model_PM10_1.sav'
      pickle.dump(gmb_O3, open(filename, 'wb'))

      gmb_O3.fit(X_PM25,y_PM25)
      filename = 'LGBM_model_O3_1.sav'
      pickle.dump(gmb_O3, open(filename, 'wb'))






