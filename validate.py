import numpy as np
import pandas as pd
from CustomImputer import DataFrameImputer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import utils

# CV
params = {
        'n_estimators': [50, 100, 200, 300],
        'min_child_weight': [5, 10, 15],
        'gamma': [0.3, 0.5, 1, 1.5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [2, 3, 4]
        }
df = pd.read_csv("train.csv")
cat_vars = ['brand', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
xgb_reg = xgb.XGBRegressor(n_estimators=50, max_depth=3)
lin_reg = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, criterion="mae")
utils.randomizedsearch_CV(df, xgb_reg, cat_vars, utils.smooth_handling, params, weights=[0.1, 0.9], trials=20)
#utils.gridsearch_CV(df, xgb_reg, cat_vars, utils.smooth_handling, params)
#utils.full_CV_pipeline(df, xgb_reg, cat_vars, utils.smooth_handling)


df = pd.read_csv("train.csv")
utils.train_test_index(df)
##### min_price

df_min_in = df.copy()

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price']
target = 'min_price'
num_vars = [col for col in df.columns if col not in cat_vars + dummy_vars + target_vars]
variable_lists = [cat_vars, dummy_vars, target_vars, num_vars]

df_min_in = utils.imputation(df_min_in)
utils.drop_columns(df_min_in, ['name', 'base_name', 'pixels_y', 'max_price'], variable_lists)
# utils.decrease_cat_size_handling(df_min_in, cat_vars, target)
# df_min_in = utils.one_hot_encoding(df_min_in, cat_vars)
utils.smooth_handling(df_min_in, cat_vars, target)

xgb_reg = xgb.XGBRegressor(n_estimators=200, max_depth=4, gamma=0.3, colsample_bytree=0.6, subsample=1, min_child_weight=15)
estimator = xgb_reg

df_min, mae_min = utils.fit_mae(df_min_in, estimator, target, 'id', 'MIN')
df_comp_min = utils.get_predictions(df_min_in, estimator, target, 'id', 'min_price_pred')

##### max_price

df_max_in = df.copy()

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price']
target = 'max_price'
num_vars = [col for col in df.columns if col not in cat_vars + dummy_vars + target_vars]
variable_lists = [cat_vars, dummy_vars, target_vars, num_vars]

df_max_in = utils.imputation(df_max_in)
utils.drop_columns(df_max_in, ['name', 'base_name', 'pixels_y', 'min_price'], variable_lists)
df_max_in = df_max_in.merge(df_comp_min, on='id')

# utils.decrease_cat_size_handling(df, cat_vars, target)
# df = utils.one_hot_encoding(df, cat_vars)
utils.smooth_handling(df_max_in, cat_vars, target)

estimator = xgb.XGBRegressor(n_estimators=200, max_depth=4, gamma=0.3, colsample_bytree=0.6, subsample=1, min_child_weight=15)

df_max, mae_max = utils.fit_mae(df_max_in, estimator, target, 'id', 'MAX')
df_comp_max = utils.get_predictions(df_max_in, estimator, target, 'id', 'max_price_pred')

# difference
df['dif'] = df['max_price'] - df['min_price']

df_dif_in = df.copy()

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price', 'dif']
target = 'dif'
num_vars = [col for col in df.columns if col not in cat_vars + dummy_vars + target_vars]
variable_lists = [cat_vars, dummy_vars, target_vars, num_vars]

df_dif_in = utils.imputation(df_dif_in)
utils.drop_columns(df_dif_in, ['name', 'base_name', 'pixels_y'], variable_lists)
df_dif_in = df_dif_in.merge(df_comp_min, on='id')
df_dif_in = df_dif_in.merge(df_comp_max, on='id')
utils.smooth_handling(df_dif_in, cat_vars, target)

estimator = xgb.XGBRegressor(n_estimators=50, max_depth=4, gamma=0.5, colsample_bytree=0.8, subsample=1, min_child_weight=10)
df_dif, mae_dif = utils.fit_mae(df_dif_in, estimator, target, 'id', 'DIF')


# put together
df_fin = df_min.merge(df_max, on="id")
tot_mae1 = mean_absolute_error(df_fin['TRUE_MAX'], df_fin['MAX']) + mean_absolute_error(df_fin['TRUE_MIN'], df_fin['MIN'])

df_fin2 = df_min.merge(df_dif, on="id").merge(df_max, on="id")
df_fin2['MAX'] = df_fin2['MIN'] + df_fin2['DIF']
tot_mae2 = mean_absolute_error(df_fin2['TRUE_MAX'], df_fin2['MAX']) + mean_absolute_error(df_fin2['TRUE_MIN'], df_fin2['MIN'])

df_fin3 = df_max.merge(df_dif, on="id").merge(df_min, on="id")
df_fin3['MIN'] = df_fin3['MAX'] - df_fin3['DIF']
tot_mae3 = mean_absolute_error(df_fin3['TRUE_MAX'], df_fin3['MAX']) + mean_absolute_error(df_fin3['TRUE_MIN'], df_fin3['MIN'])

df_out = df_min.merge(df_max, on="id").set_index("id")
df_out = df_out.merge(df_dif, on="id").set_index("id")
df_out['MAX'] = df_out['MAX']*0.1 + (df_out['MIN'] + abs(df_out['DIF']))*0.9
df_out['MIN'] = df_out['MIN']*0.1 + (df_out['MAX'] - abs(df_out['DIF']))*0.9
df_out.loc[df_out['MAX'] < df_out['MIN'], ['MIN', 'MAX']] = df_out.loc[df_out['MAX'] < df_out['MIN'], ['MIN', 'MAX']].mean(axis=1)
tot_mae = mean_absolute_error(df_out['TRUE_MAX'], df_out['MAX']) + mean_absolute_error(df_out['TRUE_MIN'], df_out['MIN'])