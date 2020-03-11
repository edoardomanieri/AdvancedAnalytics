import numpy as np
import pandas as pd
from CustomImputer import DataFrameImputer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import utils

# CV
df = pd.read_csv("train.csv")
cat_vars = ['brand', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
xgb_reg = xgb.XGBRegressor(n_estimators=200, max_depth=3)
utils.CV_pipeline(df, xgb_reg, cat_vars, utils.smooth_handling, 'min_price')


##### min_price
df = pd.read_csv("train.csv")
utils.train_test_index(df)

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price']
target = 'min_price'
num_vars = [col for col in df.columns if col not in cat_vars + dummy_vars + target_vars]
variable_lists = [cat_vars, dummy_vars, target_vars, num_vars]

df = utils.imputation(df)
utils.drop_columns(df, ['name', 'base_name', 'pixels_y', 'max_price'], variable_lists)
# utils.decrease_cat_size_handling(df, cat_vars, target)
# df = utils.one_hot_encoding(df, cat_vars)
utils.smooth_handling(df, cat_vars, target)

xgb_reg = xgb.XGBRegressor(n_estimators=200, max_depth=3)
estimator = xgb_reg

df_min, mae_min = utils.fit_mae(df, estimator, target, 'id', 'MIN')
df_complete_predictions = utils.get_predictions(df, estimator, target, 'id', 'min_price_pred')

##### max_price
df = pd.read_csv("train.csv")
utils.train_test_index(df)

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price']
target = 'max_price'
num_vars = [col for col in df.columns if col not in cat_vars + dummy_vars + target_vars]
variable_lists = [cat_vars, dummy_vars, target_vars, num_vars]

df = utils.imputation(df)
utils.drop_columns(df, ['name', 'base_name', 'pixels_y', 'min_price'], variable_lists)
df = df.merge(df_complete_predictions, on='id')

# utils.decrease_cat_size_handling(df, cat_vars, target)
# df = utils.one_hot_encoding(df, cat_vars)
utils.smooth_handling(df, cat_vars, target)

xgb_reg = xgb.XGBRegressor(n_estimators=200, max_depth=3)
estimator = xgb_reg

df_max, mae_max = utils.fit_mae(df, estimator, target, 'id', 'MAX')

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
