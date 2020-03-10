import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CustomImputer import DataFrameImputer
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def split_train_test(df, test_size=0.2, random_state=42):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    df['train'] = 0
    df.loc[train.index, 'train'] = 1 

df = pd.read_csv("train.csv")
split_train_test(df)

##### min_price


cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price']
num_vars = [col for col in train.columns if col not in cat_vars + dummy_vars + target_vars]

imputer = DataFrameImputer()
imputer.fit(df[df['train'] == 1])
df = imputer.transform(df)
df = df.drop(columns=['base_name', 'name'])
cat_vars.remove('base_name')
cat_vars.remove('name')

for var in cat_vars:
    replace_dict, mean = calc_smooth_mean(df[df['train'] == 1], by=var, on="min_price", m=10)
    df.loc[df['train'] == 1, :].replace(replace_dict, inplace=True)
    df.loc[df['train'] == 0, var] = np.where(df.loc[df['train'] == 0, var] .isin(replace_dict.keys()), df.loc[df['train'] == 0, var] .map(replace_dict), mean).astype(float)

for var in cat_vars:
    replace_dict = reduce_categories(df[df['train'] == 1], var, "min_price")
    df.replace(replace_dict, inplace=True)

df = to_categorical(df, cat_vars)
df = generate_dummies(df, cat_vars)

y_train =  df.loc[df['train'] == 1, 'min_price'].values
y_test = df.loc[df['train'] == 0, 'min_price'].values
X_train = df[df['train'] == 1].drop(columns=["max_price", 'min_price', 'id', 'pixels_y']).values
X_test = df[df['train'] == 0].drop(columns=["max_price", 'min_price', 'id', 'pixels_y']).values

lasso = Lasso(alpha=1)
xgb_reg = xgb.XGBRegressor(n_estimators=200, max_depth=3)
lin_reg = LinearRegression()
rf = RandomForestRegressor(n_estimators=100)

estimator = xgb_reg
estimator.fit(X_train, y_train)
predictions_min = estimator.predict(X_test)
mae_min = mean_absolute_error(y_test, predictions_min)
mae_min

test['MIN'] = predictions_min
test['TRUE_MIN'] = y_test
df_min = test[['id', 'MIN', 'TRUE_MIN']]

##### max_price

df = pd.read_csv("train.csv")
train, test = train_test_split(df, test_size=0.2, random_state=42)

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price']
num_vars = [col for col in train.columns if col not in cat_vars + dummy_vars + target_vars]

imputer = DataFrameImputer()
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)

for var in cat_vars:
    replace_dict, mean = calc_smooth_mean(train, by=var, on="max_price", m=10)
    train.replace(replace_dict, inplace=True)
    test[var] = np.where(test[var].isin(replace_dict.keys()), test[var].map(replace_dict), mean).astype(float)

test['min_price'] = df_min['MIN']
y_train = train['max_price'].values
y_test = test['max_price'].values
X_train = train.drop(columns=['max_price', 'id', 'name', 'base_name', 'pixels_y']).values
X_test = test.drop(columns=['max_price', 'id', 'name', 'base_name', 'pixels_y']).values

xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=3)
lin_reg = LinearRegression()
rf = RandomForestRegressor(n_estimators=100)

estimator = xgb_reg
estimator.fit(X_train, y_train)
predictions_max = estimator.predict(X_test)
mae_max = mean_absolute_error(y_test, predictions_max)
mae_max

test['MAX'] = predictions_max
test['TRUE_MAX'] = y_test
df_max = test[['id', 'MAX', 'TRUE_MAX']]

# difference

df = pd.read_csv("train.csv")
train, test = train_test_split(df, test_size=0.2, random_state=42)

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price']
num_vars = [col for col in train.columns if col not in cat_vars + dummy_vars + target_vars]

imputer = DataFrameImputer()
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)
train['dif'] = train['max_price'] - train['min_price']

for var in cat_vars:
    replace_dict, mean = calc_smooth_mean(train, by=var, on="dif", m=10)
    train.replace(replace_dict, inplace=True)
    test[var] = np.where(test[var].isin(replace_dict.keys()), test[var].map(replace_dict), mean).astype(float)

test['min_price'] = df_min['MIN']
y_train = train['max_price'].values - train['min_price'].values
y_test = test['max_price'].values - test['min_price'].values
X_train = train.drop(columns=['max_price', 'dif', 'id', 'name', 'base_name', 'pixels_y']).values
X_test = test.drop(columns=['max_price', 'id', 'name', 'base_name', 'pixels_y']).values

xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=3)
lin_reg = LinearRegression()
rf = RandomForestRegressor(n_estimators=100)

estimator = xgb_reg
estimator.fit(X_train, y_train)
predictions_dif = estimator.predict(X_test)
mae_dif = mean_absolute_error(y_test, predictions_dif)
mae_dif

test['DIF'] = predictions_dif
test['TRUE_DIF'] = y_test
df_dif = test[['id', 'DIF', 'TRUE_DIF']]

# put together
df_fin = df_min.merge(df_max, on="id")
tot_mae1 = mean_absolute_error(df_fin['TRUE_MAX'], df_fin['MAX']) + mean_absolute_error(df_fin['TRUE_MIN'], df_fin['MIN'])

df_fin2 = df_min.merge(df_dif, on="id").merge(df_max, on="id")
df_fin2['MAX'] = df_fin2['MIN'] + df_fin2['DIF']
tot_mae2 = mean_absolute_error(df_fin2['TRUE_MAX'], df_fin2['MAX']) + mean_absolute_error(df_fin2['TRUE_MIN'], df_fin2['MIN'])

df_fin3 = df_max.merge(df_dif, on="id").merge(df_min, on="id")
df_fin3['MIN'] = df_fin3['MAX'] - df_fin3['DIF']
tot_mae3 = mean_absolute_error(df_fin3['TRUE_MAX'], df_fin3['MAX']) + mean_absolute_error(df_fin3['TRUE_MIN'], df_fin3['MIN'])
