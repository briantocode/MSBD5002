from geopy import distance
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from collections import Counter
from tqdm import tqdm


AQ_Train = pd.read_csv('_AQ_n_AQI.csv', header=[0,1], parse_dates = True, index_col = 0)
GW_Train = pd.read_csv('_GW_Train.csv', header=[0,1], parse_dates = True, index_col = 0)
OW_Train = pd.read_csv('_OW_Train.csv', header=[0,1], parse_dates = True, index_col = 0)
OW_Train.index = pd.to_datetime(OW_Train.index)

AQ2OW = pd.read_csv('AQ2OW.csv', index_col = 0)
AQ2GW = pd.read_csv('AQ2GW.csv', index_col = 0)
AQ2GW.set_index(['AQ_sid'], inplace=True)

def getNewObNumFeature(weather_data, dist_table, ori_loc, feature):
    df_idx = weather_data[dist_table.loc[ori_loc][0]].index
    df_val = []
    L1_wt = dist_table.loc[ori_loc][1]
    L2_wt = dist_table.loc[ori_loc][3]
    for i in range(len(weather_data)):
        L1_val = weather_data[dist_table.loc[ori_loc][0]][feature][i]
        L2_val = weather_data[dist_table.loc[ori_loc][2]][feature][i]
        if np.isnan(L1_val) & np.isnan(L2_val):
            df_val.append(np.nan)
        elif np.isnan(L2_val):
            df_val.append(round(L1_val, 3))
        elif np.isnan(L1_val):
            df_val.append(round(L2_val, 3))
        else:
            df_val.append(round((L1_val*L1_wt + L2_val*L2_wt), 3))
    df = pd.DataFrame({'time': df_idx, 'new_val': df_val})
    df.set_index(['time'], inplace=True)
    return df

def getNewObCatFeature(weather_data, dist_table, ori_loc, feature):
    df_idx = weather_data[dist_table.loc[ori_loc][0]].index
    df_val = []
    L1_wt = dist_table.loc[ori_loc][1]
    L2_wt = dist_table.loc[ori_loc][3]
    for i in range(len(weather_data)):
        L1_val = weather_data[dist_table.loc[ori_loc][0]][feature][i]
        L2_val = weather_data[dist_table.loc[ori_loc][2]][feature][i]
        if np.isnan(L1_val) & np.isnan(L2_val):
            df_val.append(np.nan)
        elif np.isnan(L1_val):
            df_val.append(L2_val)
        else:
            df_val.append(L1_val)
    df = pd.DataFrame({'time': df_idx, 'new_val': df_val})
    df.set_index(['time'], inplace=True)
    return df

def getNewGridNumFeature(weather_data, dist_table, ori_loc, feature):
    df_idx = weather_data[dist_table.loc[ori_loc][0]].index
    df_val = []
    L1_wt = dist_table.loc[ori_loc][1]
    L2_wt = dist_table.loc[ori_loc][3]
    L3_wt = dist_table.loc[ori_loc][5]
    L4_wt = dist_table.loc[ori_loc][7]
    for i in range(len(weather_data)):
        L1_val = weather_data[dist_table.loc[ori_loc][0]][feature][i]
        L2_val = weather_data[dist_table.loc[ori_loc][2]][feature][i]
        L3_val = weather_data[dist_table.loc[ori_loc][4]][feature][i]
        L4_val = weather_data[dist_table.loc[ori_loc][6]][feature][i]
        if np.isnan(L1_val) or np.isnan(L2_val) or np.isnan(L3_val) or np.isnan(L4_val):
            df_val.append(np.nan)
        else:
            df_val.append(round((L1_val*L1_wt + L2_val*L2_wt + L3_val*L3_wt + L4_val*L4_wt), 3))
    df = pd.DataFrame({'time': df_idx, 'new_val': df_val})
    df.set_index(['time'], inplace=True)
    return df

def getNewGridCatFeature(weather_data, dist_table, ori_loc, feature):
    df_idx = weather_data[dist_table.loc[ori_loc][0]].index
    df_val = []
    L1_wt = dist_table.loc[ori_loc][1]
    L2_wt = dist_table.loc[ori_loc][3]
    L3_wt = dist_table.loc[ori_loc][5]
    L4_wt = dist_table.loc[ori_loc][7]
    for i in range(len(weather_data)):
        L1_val = weather_data[dist_table.loc[ori_loc][0]][feature][i]
        L2_val = weather_data[dist_table.loc[ori_loc][2]][feature][i]
        L3_val = weather_data[dist_table.loc[ori_loc][4]][feature][i]
        L4_val = weather_data[dist_table.loc[ori_loc][6]][feature][i]
        if L1_wt == max(L1_wt, L2_wt, L3_wt, L4_wt):
            df_val.append(L1_val)
        elif L2_wt == max(L1_wt, L2_wt, L3_wt, L4_wt):
            df_val.append(L2_val)
        elif L3_wt == max(L1_wt, L2_wt, L3_wt, L4_wt):
            df_val.append(L3_val)
        elif L4_wt == max(L1_wt, L2_wt, L3_wt, L4_wt):
            df_val.append(L4_val)
        else:
            df_val.append(np.nan)
    df = pd.DataFrame({'time': df_idx, 'new_val': df_val})
    df.set_index(['time'], inplace=True)
    return df

dict_temp = {}
dict_pres = {}

ObFeature = sorted(list(AQ2OW.index))
for i in tqdm(ObFeature):
	new_wr = getNewObCatFeature(OW_Train, AQ2OW, i, 'weather')
    new_t = getNewObNumFeature(OW_Train, AQ2OW, i, 'temperature')
    new_p = getNewObNumFeature(OW_Train, AQ2OW, i, 'pressure')
	new_h = getNewObNumFeature(OW_Train, AQ2OW, i, 'humidity')
	new_wd = getNewObCatFeature(OW_Train, AQ2OW, i, 'wind_direction')
	new_ws = getNewObNumFeature(OW_Train, AQ2OW, i, 'wind_speed')
	new_df = pd.concat([new_wr, new_t, new_p, new_h, new_wd, new_w], axis = 1)
	new_df.columns = ['ob_wr', 'ob_t', 'ob_p', 'ob_h', 'ob_wd', 'ob_ws']
	new_df.columns = pd.MultiIndex.from_product([[i], new_df.columns])
    AQ_Train = pd.concat([AQ_Train, new_t], axis = 1)

GrFeature = sorted(list(AQ2GW.index))
for i in tqdm(GrFeature):
	new_wr = getNewGridCatFeature(GW_Train, AQ2GW, i, 'weather')
	new_t = getNewGridNumFeature(GW_Train, AQ2GW, i, 'temperature')
	new_p = getNewGridNumFeature(GW_Train, AQ2GW, i, 'pressure')
	new_h = getNewGridNumFeature(GW_Train, AQ2GW, i, 'humidity')
	new_wd = getNewGridCatFeature(GW_Train, AQ2GW, i, 'wind_direction')
	new_ws = getNewGridNumFeature(GW_Train, AQ2GW, i, 'wind_speed')
	new_df = pd.concat([new_wr, new_t, new_p, new_h, new_wd, new_w], axis = 1)
	new_df.columns = ['gr_wr', 'gr_t', 'gr_p', 'gr_h', 'gr_wd', 'gr_ws']
	new_df.columns = pd.MultiIndex.from_product([[i], new_df.columns])
	AQ_Train = pd.concat([AQ_Train, new_df], axis = 1)

AQ_Train.sort_index(axis=1, inplace = True)
AQ_Train.to_csv('addfeature_test.csv')


print(AQ_Train)
