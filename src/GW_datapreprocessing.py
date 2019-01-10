import numpy as np
import pandas as pd
from datetime import datetime, timedelta
#from tqdm import tqdm


def getGWS(data):
	'''
	input a dataframe and return the longitude and latitude location of each observe station
	'''
	data.rename(columns = {'stationName': 'GW_sid', 'utc_time': 'time', 'wind_speed/kph': 'wind_speed'}, inplace = True)
	GWS = data[['GW_sid', 'longitude', 'latitude']]
	GWS.set_index(['GW_sid'], inplace=True)
	GWS = GWS.drop_duplicates()
	return GWS

def getlocaltime(data):
	'''
	convert utc time to location time
	'''
	data.utc_time = pd.to_datetime(data.utc_time, format='%Y%m%d %H:%M:%S')
	data.utc_time = data.utc_time + timedelta(hours=8)
	return data

def convertWeather(data):
	column_name = 'weather'
	mask1 = data.weather == 'CLEAR_DAY'
	mask2 = data.weather == 'CLEAR_NIGHT'
	mask3 = data.weather == 'CLOUDY'
	mask4 = data.weather == 'HAZE'
	mask5 = data.weather == 'PARTLY_CLOUDY_DAY'
	mask6 = data.weather == 'PARTLY_CLOUDY_NIGHT'
	mask7 = data.weather == 'RAIN'
	mask8 = data.weather == 'SNOW'
	mask9 = data.weather == 'WIND'

	data.loc[mask1, column_name] = 0
	data.loc[mask2, column_name] = 0
	data.loc[mask3, column_name] = 1
	data.loc[mask4, column_name] = 4
	data.loc[mask5, column_name] = 1
	data.loc[mask6, column_name] = 1
	data.loc[mask7, column_name] = 3
	data.loc[mask8, column_name] = 5
	data.loc[mask9, column_name] = 2
	return data

def convertWD(data):
	column_name = 'wind_direction'
	mask1 = data.wind_direction <= 30
	mask2 = (30 < data.wind_direction) & (data.wind_direction <= 60)
	mask3 = (60 < data.wind_direction) & (data.wind_direction <= 90)
	mask4 = (90 < data.wind_direction) & (data.wind_direction <= 120)
	mask5 = (120 < data.wind_direction) & (data.wind_direction <= 150)
	mask6 = (150 < data.wind_direction) & (data.wind_direction <= 180)
	mask7 = (180 < data.wind_direction) & (data.wind_direction <= 210)
	mask8 = (210 < data.wind_direction) & (data.wind_direction <= 240)
	mask9 = (240 < data.wind_direction) & (data.wind_direction <= 270)
	mask10 = (270 < data.wind_direction) & (data.wind_direction <= 300)
	mask11 = (300 < data.wind_direction) & (data.wind_direction <= 330)
	mask12 = (330 < data.wind_direction) & (data.wind_direction <= 360)  

	data.loc[mask1, column_name] = 1
	data.loc[mask2, column_name] = 2
	data.loc[mask3, column_name] = 3
	data.loc[mask4, column_name] = 4
	data.loc[mask5, column_name] = 5
	data.loc[mask6, column_name] = 6
	data.loc[mask7, column_name] = 7
	data.loc[mask8, column_name] = 8
	data.loc[mask9, column_name] = 9
	data.loc[mask10, column_name] = 10
	data.loc[mask11, column_name] = 11
	data.loc[mask12, column_name] = 12
	return data

def createDF(data, GWS_name, start_date, end_date):
	dict_keys = list(map(lambda x:'BJ%03d' % x,range(651)))
	inc_value = getInc(data, GWS_name)
	df_dummie = getDummie(start_date, end_date)
	dict_value = getDictValue(data, inc_value, df_dummie)
	GW_dict = dict(zip(dict_keys, dict_value))
	return pd.concat(GW_dict.values(), axis=1, keys=GW_dict.keys())

def daterange(start_date, end_date):
    delta = timedelta(hours=1)
    while start_date < end_date:
        yield start_date
        start_date += delta

def getDummie(start_date, end_date):
	'''
	generate a dummie variable to fill the missing time gap, parameters are the starting time and ending time
	'''
	idx = []
	for single_date in daterange(start_date, end_date):
		idx.append(single_date.strftime("%Y-%m-%d %H:%M:%S"))
	val = np.zeros(len(idx))
	df_dummie = pd.DataFrame({'time': idx, 'dummie': val})
	df_dummie.set_index(['time'], inplace=True)
	return df_dummie

def getDictValue(data, inc_value, df_dummie):
	'''
	input are the timestamp, value and the dummie dataframe
	output are the list
	'''
	dict_value = []
	begin = 0
	end = 0
	for i in inc_value:
	    end += i
	    temp = data[begin:end]
	    temp.set_index(['time'], inplace=True)
	    temp.drop_duplicates(keep = False, inplace=True)
	    temp = temp[~temp.index.duplicated(keep='first')]
	    temp = temp.join(df_dummie, how = 'outer')
	    temp.drop(['GW_sid','dummie'], axis =1, inplace = True)
	    dict_value.append(temp)
	    begin = end
	return dict_value

def getInc(data, GWS_name):
	#function like a counter
	inc_value = []
	for i in range(len(GWS_name)):
		x = data[data.GW_sid == GWS_name[i]].shape[0]
		inc_value.append(x)
	return inc_value

def dataCleansing(data):
	data = convertWeather(data)
	data = convertWD(data)
	return data

def transformGW1(df_GW_1, GWS_name):
	'''
	Transform the the dataframe of AQ1 into index of time stamp and 2 level columns, level 0 is the 6 pollutants, and level 1 is station name
	'''
	df_GW_1.rename(columns = {'stationName': 'GW_sid', 'time': 'utc_time', 'wind_speed/kph': 'wind_speed'}, inplace = True)
	df_GW_1 = getlocaltime(df_GW_1)
	df_GW_1.rename(columns = {'utc_time': 'time'}, inplace = True)
	df_GW_1.wind_speed = round(df_GW_1.wind_speed*5/18, 2)
	df_GW_1['weather'] = np.nan
	df_GW_1 = convertWD(df_GW_1)
	#df_GW_1 = dataCleansing(df_GW_1)
	
	df_GW_1 = df_GW_1[['time','GW_sid','weather','temperature','pressure','humidity','wind_direction','wind_speed']]
	df_GW_1.sort_values(by=['GW_sid', 'time'], inplace = True)
	df_GW_1 = df_GW_1.reset_index(drop=True)

	start_date = datetime(2017, 1, 1, 22, 0)
	end_date = datetime(2018, 2, 1, 0, 0)
	return createDF(df_GW_1, GWS_name, start_date, end_date)

def transformGW2(df_GW_2, GWS_name):
	df_GW_2.rename(columns = {'time': 'utc_time'}, inplace = True)
	df_GW_2.drop(['id'], axis= 1, inplace = True)
	df_GW_2 = dataCleansing(df_GW_2)
	df_GW_2 = getlocaltime(df_GW_2)

	df_GW_2.rename(columns = {'station_id': 'GW_sid', 'utc_time': 'time'}, inplace = True)
	df_GW_2.wind_speed = round(df_GW_2.wind_speed*5/18, 2)
	df_GW_2 = df_GW_2[['time','GW_sid','weather','temperature','pressure','humidity','wind_direction','wind_speed']]
	df_GW_2.sort_values(by=['GW_sid', 'time'], inplace = True)
	df_GW_2 = df_GW_2.reset_index(drop=True)

	start_date = datetime(2018, 2, 1, 0, 0)
	end_date = datetime(2018, 4, 1, 9, 0)
	return createDF(df_GW_2, GWS_name, start_date, end_date)

def transformGWT(df_GW_T, GWS_name):
	df_GW_T.drop(['id'], axis= 1, inplace = True)
	df_GW_T = dataCleansing(df_GW_T)
	df_GW_T.rename(columns = {'time': 'utc_time'}, inplace = True)
	df_GW_T = getlocaltime(df_GW_T)
	df_GW_T.rename(columns = {'utc_time': 'time', 'station_id': 'GW_sid'}, inplace = True)
	df_GW_T = df_GW_T[['time','GW_sid','weather','temperature','pressure','humidity','wind_direction','wind_speed']]
	#convert wind speed form km/h to m/s
	df_GW_T.wind_speed = round(df_GW_T.wind_speed*5/18, 2)

	df_GW_T.sort_values(by=['GW_sid', 'time'], inplace = True)
	df_GW_T = df_GW_T.reset_index(drop=True)

	start_date = datetime(2018, 5, 1, 8, 0)
	end_date = datetime(2018, 5, 3, 8, 0)
	return createDF(df_GW_T, GWS_name, start_date, end_date)

def main():
	df_GW_1 = pd.read_csv('gridWeather_201701-201803.csv')
	df_GW_2 = pd.read_csv('gridWeather_201804.csv')
	df_GW_test = pd.read_csv('gridWeather_20180501-20180502.csv')

	# df_BGS = pd.read_csv('Beijing_grid_weather_station.csv', header = None, names = ['station_id', 'latitude', 'longitude'])
	# df_BGS = df_BGS[['station_id', 'latitude', 'longitude']]

	df_GWS = getGWS(df_GW_1)
	GWS_name = df_GWS.index.tolist()
	GW1_T = transformGW1(df_GW_1, GWS_name)
	GW2_T = transformGW2(df_GW_2, GWS_name)
	GW_test = transformGWT(df_GW_test, GWS_name)
	
	GW_T = pd.concat([GW1_T, GW2_T, GW_test])

	GW_T = GW_T[:-8]
	GW_Train = GW_T[:-48]
	GW_Test = GW_T[-48:]

	print(GW_Train)
	print(GW_Test)


	# df_GWS.to_csv('GridLoc.csv')
	# GW_Train.to_csv('_GW_Train.csv')
	# GW_Test.to_csv('_GW_Test.csv')

if __name__== "__main__":
  main()



