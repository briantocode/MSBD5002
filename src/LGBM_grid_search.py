import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer
file_path = '../data/train_by_station/allstation.csv'
features = ['ob_w', 'h', 'p', 't', 'ws','time']
label = ['PM2.5', 'PM10', 'O3']


X_df = pd.read_csv(file_path)
X_df.dropna(inplace=True)
X_df['time'] = X_df['time'].apply(lambda x:x%24)

X = X_df[features]
y = X_df[label[2]]

X_p, X_r,y_p,y_r = train_test_split(X, y, test_size=0.6, random_state=1,shuffle = True)
def scoring(reg, x, y):
    pred = reg.predict(x)
    return -mape_error(pred, y)
def mape_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

params = {

    # 'boosting_type': ['gbdt'],
    # 'objective':['regression_l2'],
    # 'metric': ['l2', 'l1'],
    'max_depth': [10,15,20],
    'num_leaves': [200,400,500],
    'min_data_in_leaf': [20],
    'learning_rate': [0.05,0.1],
    'feature_fraction': [1],
    'bagging_fraction': [0.8,0.9],
    'bagging_freq': [1],
    'bagging_seed': [3],
    'verbose': [0],
'num_iterations':[1000,1500,2000]
}
cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=2)
estimator=lgb.LGBMRegressor()

gbm = GridSearchCV(estimator, param_grid=params, scoring=scoring, n_jobs=-1, cv=cv, verbose=6)


def params(train_X, train_Y):
#     train_X, test_X, train_Y, test_Y = load_train_test(city=city, attr=attr)
    # RF.fit(train_X, train_Y)
    gbm.fit(train_X, train_Y)
    print ('best params:\n', gbm.best_params_)
    mean_scores = np.array(gbm.cv_results_['mean_test_score'])
    print ('mean score', mean_scores)
    print ('best score', gbm.best_score_)
    return gbm.best_params_

best_params = params(X_p,y_p)
