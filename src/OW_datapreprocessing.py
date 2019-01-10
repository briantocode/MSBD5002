import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def getOWS(data):
	'''
	input a dataframe and return the longitude and latitude location of each observe station
	'''
	data = data[['station_id', 'longitude', 'latitude']]
	data.rename(columns = {'station_id': 'OW_sid'}, inplace = True)
	data.set_index(['OW_sid'], inplace=True)
	data = data.drop_duplicates()
	return data

def getlocaltime(data):
	'''
	convert utc time to location time
	'''
	data.utc_time = pd.to_datetime(data.utc_time, format='%Y%m%d %H:%M:%S')
	data.utc_time = data.utc_time + timedelta(hours=8)
	return data

def convertWeather(data):
	column_name = 'weather'
	mask1 = data.weather == 'Cloudy'	#1
	mask2 = data.weather == 'Dust'		#6
	mask3 = data.weather == 'Fog'		#8
	mask4 = data.weather == 'Hail'		#7
	mask5 = data.weather == 'Haze'		#4
	mask6 = data.weather == 'Light Rain'#3
	mask7 = data.weather == 'Overcast'	#1
	mask8 = data.weather == 'Rain'		#3
	mask9 = data.weather == 'Rain with Hail'		#7
	mask10 = data.weather == 'Rain/Snow with Hail'	#7
	mask11 = data.weather == 'Sand'		#9
	mask12 = data.weather == 'Sleet'	#3
	mask13 = data.weather == 'Snow'		#5
	mask14 = data.weather == 'Sunny/clear'			#0
	mask15 = data.weather == 'Thundershower'		#3
	mask16 = (data.weather == 999999)

	data.loc[mask1, column_name] = 1
	data.loc[mask2, column_name] = 6
	data.loc[mask3, column_name] = 8
	data.loc[mask4, column_name] = 7
	data.loc[mask5, column_name] = 4
	data.loc[mask6, column_name] = 3
	data.loc[mask7, column_name] = 1
	data.loc[mask8, column_name] = 3
	data.loc[mask9, column_name] = 7
	data.loc[mask10, column_name] = 7
	data.loc[mask11, column_name] = 9
	data.loc[mask12, column_name] = 3
	data.loc[mask13, column_name] = 5
	data.loc[mask14, column_name] = 0
	data.loc[mask15, column_name] = 3
	data.loc[mask16, column_name] = np.nan
	return data

def convertWD(data):
	column_name = 'wind_direction'

	mask1 = data.wind_direction <= 30
	mask2 = (30 < data.wind_direction) & (data.wind_direction<= 60)
	mask3 = (60 < data.wind_direction) & (data.wind_direction<= 90)
	mask4 = (90 < data.wind_direction) & (data.wind_direction<= 120)
	mask5 = (120 < data.wind_direction) & (data.wind_direction<= 150)
	mask6 = (150 < data.wind_direction) & (data.wind_direction<= 180)
	mask7 = (180 < data.wind_direction) & (data.wind_direction<= 210)
	mask8 = (210 < data.wind_direction) & (data.wind_direction<= 240)
	mask9 = (240 < data.wind_direction) & (data.wind_direction<= 270)
	mask10 = (270 < data.wind_direction) & (data.wind_direction<= 300)
	mask11 = (300 < data.wind_direction) & (data.wind_direction<= 330)
	mask12 = (330 < data.wind_direction) & (data.wind_direction<= 360)
	mask13 = (data.wind_direction == 999017)
	mask14 = (data.wind_direction == 999999)      

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
	data.loc[mask13, column_name] = 0
	data.loc[mask14, column_name] = np.nan

	return data

def changeSP(data):
	col_1 = 'temperature'
	col_2 = 'pressure'
	col_3 = 'humidity'
	col_4 = 'wind_speed'

	mask1 = (data.temperature == 999999)
	mask2 = (data.pressure == 999999)
	mask3 = (data.humidity == 999999)
	mask4 = (data.wind_speed == 999999)

	data.loc[mask1, col_1] = np.nan
	data.loc[mask2, col_2] = np.nan
	data.loc[mask3, col_3] = np.nan
	data.loc[mask4, col_4] = np.nan
	return data

def createDF(data, OWS_name, start_date, end_date):
	inc_value = getInc(data, OWS_name)
	df_dummie = getDummie(start_date, end_date)
	dict_value = getDictValue(data, inc_value, df_dummie)
	OW_dict = dict(zip(OWS_name, dict_value))
	return pd.concat(OW_dict.values(), axis=1, keys=OW_dict.keys())

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
	    temp.drop(['OW_sid','dummie'], axis =1, inplace = True)
	    dict_value.append(temp)
	    begin = end
	return dict_value

def getInc(data, OWS_name):
	#function like a counter
	inc_value = []
	for i in range(len(OWS_name)):
		x = data[data.OW_sid == OWS_name[i]].shape[0]
		inc_value.append(x)
	return inc_value

def dataCleansing(data):
	data = convertWeather(data)
	data = convertWD(data)
	data = changeSP(data)
	return data

def transformOW1(df_OW_1, OWS_name):
	'''
	Transform the the dataframe of AQ1 into index of time stamp and 2 level columns, level 0 is the 6 pollutants, and level 1 is station name
	'''
	df_OW_1 = dataCleansing(df_OW_1)
	df_OW_1 = getlocaltime(df_OW_1)

	df_OW_1.rename(columns = {'station_id': 'OW_sid', 'utc_time': 'time'}, inplace = True)
	df_OW_1 = df_OW_1[['time','OW_sid','weather','temperature','pressure','humidity','wind_direction','wind_speed']]
	df_OW_1.sort_values(by=['OW_sid', 'time'], inplace = True)
	df_OW_1 = df_OW_1.reset_index(drop=True)

	start_date = datetime(2017, 1, 1, 22, 0)
	end_date = datetime(2018, 2, 1, 0, 0)
	return createDF(df_OW_1, OWS_name, start_date, end_date)

def transformOW2(df_OW_2, OWS_name):
	df_OW_2 = dataCleansing(df_OW_2)
	df_OW_2 = getlocaltime(df_OW_2)

	df_OW_2.rename(columns = {'station_id': 'OW_sid', 'utc_time': 'time'}, inplace = True)
	df_OW_2 = df_OW_2[['time','OW_sid','weather','temperature','pressure','humidity','wind_direction','wind_speed']]
	df_OW_2.sort_values(by=['OW_sid', 'time'], inplace = True)
	df_OW_2 = df_OW_2.reset_index(drop=True)

	start_date = datetime(2018, 2, 1, 0, 0)
	end_date = datetime(2018, 4, 1, 9, 0)
	return createDF(df_OW_2, OWS_name, start_date, end_date)

def transformOW3(df_OW_3, OWS_name):
	df_OW_3 = dataCleansing(df_OW_3)
	df_OW_3.rename(columns = {'time': 'utc_time'}, inplace = True)
	df_OW_3 = getlocaltime(df_OW_3)
	df_OW_3.rename(columns = {'utc_time': 'time'}, inplace = True)

	df_OW_3 = df_OW_3[['time','station_id','weather','temperature','pressure','humidity','wind_direction','wind_speed']]
	df_OW_3.sort_values(by=['station_id', 'time'], inplace = True)
	df_OW_3.rename(columns = {'station_id': 'OW_sid'}, inplace = True)
	df_OW_3 = df_OW_3.reset_index(drop=True)

	start_date = datetime(2018, 4, 1, 9, 0)
	end_date = datetime(2018, 5, 1, 8, 0)
	return createDF(df_OW_3, OWS_name, start_date, end_date)

def transformOWT(df_OW_T, OWS_name):
	df_OW_T = dataCleansing(df_OW_T)
	df_OW_T.rename(columns = {'time': 'utc_time'}, inplace = True)
	df_OW_T = getlocaltime(df_OW_T)
	df_OW_T.rename(columns = {'utc_time': 'time'}, inplace = True)

	df_OW_T = df_OW_T[['time','station_id','weather','temperature','pressure','humidity','wind_direction','wind_speed']]
	df_OW_T.sort_values(by=['station_id', 'time'], inplace = True)
	df_OW_T.rename(columns = {'station_id': 'OW_sid'}, inplace = True)
	df_OW_T = df_OW_T.reset_index(drop=True)

	start_date = datetime(2018, 5, 1, 8, 0)
	end_date = datetime(2018, 5, 3, 8, 0)
	return createDF(df_OW_T, OWS_name, start_date, end_date)

def main():
	df_OW_1 = pd.read_csv('observedWeather_201701-201801.csv')
	df_OW_2 = pd.read_csv('observedWeather_201802-201803.csv')
	df_OW_3 = pd.read_csv('observedWeather_201804.csv')
	df_OW_test = pd.read_csv('observedWeather_20180501-20180502.csv')

	df_OWS = getOWS(df_OW_1)
	OWS_name = list(sorted(df_OW_1.station_id.drop_duplicates()))

	#OWS_name = df_OWS.index.tolist()


	OW1_T = transformOW1(df_OW_1, OWS_name)
	OW2_T = transformOW2(df_OW_2, OWS_name)
	OW3_T = transformOW3(df_OW_3, OWS_name)
	OW_test = transformOWT(df_OW_test, OWS_name)

	OW_T = pd.concat([OW1_T, OW2_T, OW3_T, OW_test])

	OW_T = OW_T[:-8]
	OW_Train = OW_T[:-48]
	OW_Test = OW_T[-48:]

	df_OWS.to_csv('OWS_loc.csv')
	OW_Train.to_csv('_OW_Train.csv')
	OW_Test.to_csv('_OW_Test.csv')

if __name__== "__main__":
  main()



