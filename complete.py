import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CustomImputer import DataFrameImputer
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def calc_smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()
    
    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean']).reset_index()
    
    agg['smooth'] = (agg['count'] * agg['mean'] + m * mean) / (agg['count'] + m)
    d = pd.Series(agg['smooth'].values, index=agg[by]).to_dict()

    # Replace each value by the according smoothed mean
    return d, mean

##### min_price
train_min = pd.read_csv("train.csv")
test_min = pd.read_csv("test.csv")

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price']
num_vars = [col for col in train_min.columns if col not in cat_vars + dummy_vars + target_vars]

imputer = DataFrameImputer()
imputer.fit(train_min)
train_min = imputer.transform(train_min)
test_min = imputer.transform(test_min)

for var in cat_vars:
    replace_dict, mean = calc_smooth_mean(train_min, by=var, on="min_price", m=10)
    train_min.replace(replace_dict, inplace=True)
    test_min[var] = np.where(test_min[var].isin(replace_dict.keys()), test_min[var].map(replace_dict), mean).astype(float)

y_train_min = train_min['min_price'].values
X_train_min = train_min.drop(columns = ["max_price", 'min_price', 'id', 'name', 'base_name', 'pixels_y']).values
X_test_min = test_min.drop(columns=['id', 'name', 'base_name', 'pixels_y']).values

xgb_reg = xgb.XGBRegressor(n_estimators=20, max_depth=3)

estimator = xgb_reg
estimator.fit(X_train_min, y_train_min)
predictions = estimator.predict(X_test_min)
test_min['MIN'] = predictions
df_min = test_min[['id', 'MIN']]


##### max_price
train_max = pd.read_csv("train.csv")
test_max = pd.read_csv("test.csv")

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price']
num_vars = [col for col in train_max.columns if col not in cat_vars + dummy_vars + target_vars]

imputer = DataFrameImputer()
imputer.fit(train_max)
train_max = imputer.transform(train_max)
test_max = imputer.transform(test_max)

for var in cat_vars:
    replace_dict, mean = calc_smooth_mean(train_max, by=var, on="max_price", m=10)
    train_max.replace(replace_dict, inplace=True)
    test_max[var] = np.where(test_max[var].isin(replace_dict.keys()), test_max[var].map(replace_dict), mean).astype(float)

y_train_max = train_max['max_price'].values
X_train_max = train_max.drop(columns = ['max_price', 'min_price', 'id', 'name', 'base_name', 'pixels_y']).values
X_test_max = test_max.drop(columns=['id', 'name', 'base_name', 'pixels_y']).values

xgb_reg = xgb.XGBRegressor(n_estimators=20, max_depth=3)

estimator = xgb_reg
estimator.fit(X_train_max, y_train_max)
predictions = estimator.predict(X_test_max)
test_max['MAX'] = predictions
df_max = test_max[['id', 'MAX']]


##### difference
train_dif = pd.read_csv("train.csv")
test_dif = pd.read_csv("test.csv")

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price']
num_vars = [col for col in train_dif.columns if col not in cat_vars + dummy_vars + target_vars]

imputer = DataFrameImputer()
imputer.fit(train_dif)
train_dif = imputer.transform(train_dif)
test_dif = imputer.transform(test_dif)

for var in cat_vars:
    replace_dict, mean = calc_smooth_mean(train_dif, by=var, on="max_price", m=10)
    train_dif.replace(replace_dict, inplace=True)
    test_dif[var] = np.where(test_dif[var].isin(replace_dict.keys()), test_dif[var].map(replace_dict), mean).astype(float)

y_train_dif = train_dif['max_price'].values - train_dif['min_price'].values
X_train_dif = train_dif.drop(columns = ['max_price', 'min_price', 'id', 'name', 'base_name', 'pixels_y']).values
X_test_dif = test_dif.drop(columns=['id', 'name', 'base_name', 'pixels_y']).values

xgb_reg = xgb.XGBRegressor(n_estimators=20, max_depth=3)

estimator = xgb_reg
estimator.fit(X_train_dif, y_train_dif)
predictions = estimator.predict(X_test_dif)
test_dif['DIF'] = predictions
df_dif = test_dif[['id', 'DIF']]


# put together predictions
df_out = df_min.merge(df_max, on="id").set_index("id")
df_out.loc[df_out['MAX'] < df_out['MIN'], ['MIN', 'MAX']] = df_out.loc[df_out['MAX'] < df_out['MIN'], ['MIN', 'MAX']].mean(axis=1)

df_out = df_max.merge(df_dif, on="id").set_index("id")
df_out['MIN'] = df_out['MAX'] - df_out['DIF']
df_out = df_out.loc[:, ['MIN', 'MAX']]


df_out.to_csv("result.csv")
