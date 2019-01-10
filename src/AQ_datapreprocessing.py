import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def getAQS(data):
	'''
	input a dataframe and return the longitude and latitude location of each air quality station
	'''
	data.dropna(inplace = True)
	data = data[1:]
	data.columns = ['AQ_sid', 'longitude', 'latitude']
	data.set_index('AQ_sid', inplace = True)
	return data

def getlocaltime(data):
	'''
	convert utc time to location time
	'''
	data.utc_time = pd.to_datetime(data.utc_time, format='%Y%m%d %H:%M:%S')
	data.utc_time = data.utc_time + timedelta(hours=8)
	return data

def getInc(data, AQS_name):
	#function like a counter
	inc_value = []
	for i in range(len(AQS_name)):
		x = data[data.AQ_sid == AQS_name[i]].shape[0]
		inc_value.append(x)
	return inc_value

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
	    temp.drop(['AQ_sid','dummie'], axis =1, inplace = True)
	    dict_value.append(temp)
	    begin = end
	return dict_value

def createDF(data, AQS_name, start_date, end_date):
	inc_value = getInc(data, AQS_name)
	df_dummie = getDummie(start_date, end_date)
	dict_value = getDictValue(data, inc_value, df_dummie)
	AQ_dict = dict(zip(AQS_name, dict_value))
	return pd.concat(AQ_dict.values(), axis=1, keys=AQ_dict.keys())

def transformAQ1(df_AQ_1, AQS_name):
	'''
	Transform the the dataframe of AQ1 into index of time stamp and 2 level columns, level 0 is the 6 pollutants, and level 1 is station name
	'''
	df_AQ_1 = getlocaltime(df_AQ_1)
	df_AQ_1.rename(columns = {'utc_time': 'time', 'stationId': 'AQ_sid'}, inplace = True)
	df_AQ_1.sort_values(by=['AQ_sid', 'time'], inplace = True)	#sort 
	df_AQ_1 = df_AQ_1.reset_index(drop=True)					#set index to timestamp

	start_date = datetime(2017, 1, 1, 22, 0)
	end_date = datetime(2018, 2, 1, 0, 0)
	return createDF(df_AQ_1, AQS_name, start_date, end_date)

def transformAQ2(df_AQ_2, AQS_name):
	'''
	Transform the the dataframe of AQ2 into index of time stamp and 2 level columns, level 0 is the 6 pollutants, and level 1 is station name
	'''
	df_AQ_2 = getlocaltime(df_AQ_2)
	df_AQ_2.rename(columns = {'utc_time': 'time', 'stationId': 'AQ_sid'}, inplace = True)
	df_AQ_2.sort_values(by=['AQ_sid', 'time'], inplace = True)
	df_AQ_2 = df_AQ_2.reset_index(drop=True)

	start_date = datetime(2018, 2, 1, 0, 0)
	end_date = datetime(2018, 4, 1, 0, 0)
	return createDF(df_AQ_2, AQS_name, start_date, end_date)

def transformAQ3(df_AQ_3, AQS_name):
	'''
	Transform the the dataframe of AQ3 into index of time stamp and 2 level columns, level 0 is the 6 pollutants, and level 1 is station name
	'''
	df_AQ_3.drop(['id'], axis = 1, inplace = True)
	df_AQ_3.columns = ['AQ_sid', 'time', 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
	df_AQ_3.sort_values(by=['AQ_sid', 'time'], inplace = True)
	df_AQ_3 = df_AQ_3.reset_index(drop=True)

	start_date = datetime(2018, 4, 1, 0, 0)
	end_date = datetime(2018, 5, 1, 0, 0)
	return createDF(df_AQ_3, AQS_name, start_date, end_date)

def main():
	df_AQS = pd.read_excel('Beijing_AirQuality_Stations_en.xlsx')
	df_AQ_1 = pd.read_csv('airQuality_201701-201801.csv')
	df_AQ_2 = pd.read_csv('airQuality_201802-201803.csv')
	df_AQ_3 = pd.read_csv('aiqQuality_201804.csv')

	df_AQS = getAQS(df_AQS)
	AQS_name = list(df_AQ_1.stationId.drop_duplicates())
	AQ1_T = transformAQ1(df_AQ_1, AQS_name)
	AQ2_T = transformAQ2(df_AQ_2, AQS_name)
	AQ3_T = transformAQ3(df_AQ_3, AQS_name)
	AQ_T = pd.concat([AQ1_T, AQ2_T, AQ3_T])


	#AQ_test = transformTest(df_AQ_3, AQS_name)

	AQ_T.to_csv('_AQ_Train.csv')
	df_AQS.to_csv('AQS.csv')

if __name__== "__main__":
  main()



