# MSBD5002

Language: Python 3.x

Files Description

1. AQ_datapreprocessing.py	This file will generate a table for the Air Quality Station where all the missing day are filled with null. 
2. OW_datapreprocessing.py.     This file will generate a table for the Observatory Station where all the missing day are filled with null after some simple processing
3. GW_datapreprocessing.py.     This file will generate a table for the Grid intersection where all the missing day / error after some simple processing
4. AddNewFeatures.py.    This file will do some datacleansing, add the weather feathures to the obdservatory station.
5. fill_missing_value_final.ipynb This file generate the weather feature for the 35 air pollution stations from the grid weather feature and observation weather feature. Then fill the missing value in air pollution and weather feature as much as possible. 
6. LGBM_grid_search.py This file use grid research to find the best parameters.
7. train_LGBM_model.py This file train 3 LightGBM models for three air pollutions
8. prediction_final_version.ipynb This file make the prediction of the air pollutions in May 1st and 2nd in 2018.
9.	convert_submission_file.ipynb convert  file to submission sample format


Package Requirements

1. Geopy
2. Numpy
3. Pandas
4. Math
5. DateTime
6. tqdm
7. xgboost
8. lightgbm
9. pickle
10. sklearn
11. matplotlib

