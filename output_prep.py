import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CustomImputer import DataFrameImputer
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import utils
from pathlib import Path
import datetime


directory = "./" + datetime.now().strftime("%m-%d-%Y,%H:%M:%S")
Path(directory).mkdir(parents=True, exist_ok=True)
report_file = open(directory + "/report", "w+")

##### min_price
train_min = pd.read_csv("train.csv")
train_min.drop(columns=['max_price'], inplace=True)
test_min = pd.read_csv("test.csv")
df = utils.merge_train_test(train_min, test_min, 'min_price')

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price']
target = 'min_price'
num_vars = [col for col in df.columns if col not in cat_vars + dummy_vars + target_vars]
variable_lists = [cat_vars, dummy_vars, target_vars, num_vars]

df = utils.imputation(df, report_file)
utils.drop_columns(df, ['name', 'base_name', 'pixels_y'], variable_lists, report_file)
# utils.decrease_cat_size_handling(df, cat_vars, target)
# df = utils.one_hot_encoding(df, cat_vars)
utils.smooth_handling(df, cat_vars, target, report_file)

xgb_reg = xgb.XGBRegressor(n_estimators=200, max_depth=4, gamma=0.3, colsample_bytree=0.6, subsample=1, min_child_weight=15)
estimator = xgb_reg

df_min = utils.fit_predict(df, estimator, target, 'id', 'MIN', report_file)
df_complete_predictions = utils.get_predictions(df, estimator, target, 'id', 'min_price_pred')


##### max_price
train_min = pd.read_csv("train.csv")
train_min.drop(columns=['min_price'], inplace=True)
test_min = pd.read_csv("test.csv")
df = utils.merge_train_test(train_min, test_min, 'min_price')

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price']
target = 'max_price'
num_vars = [col for col in df.columns if col not in cat_vars + dummy_vars + target_vars]
variable_lists = [cat_vars, dummy_vars, target_vars, num_vars]

df = utils.imputation(df)
utils.drop_columns(df, ['name', 'base_name', 'pixels_y'], variable_lists)
df = df.merge(df_complete_predictions, on='id')
report_file.write("use min predictions to predict max \n")

# utils.decrease_cat_size_handling(df, cat_vars, target)
# df = utils.one_hot_encoding(df, cat_vars)
utils.smooth_handling(df, cat_vars, target)


xgb_reg = xgb.XGBRegressor(n_estimators=200, max_depth=4, gamma=0.3, colsample_bytree=0.6, subsample=1, min_child_weight=15)
estimator = xgb_reg

df_max = utils.fit_predict(df, estimator, target, 'id', 'MAX')

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
report_file.write("replace values where max smaller than min with mean of two values \n")

df_out = df_max.merge(df_dif, on="id").set_index("id")
df_out['MIN'] = df_out['MAX'] - df_out['DIF']
df_out = df_out.loc[:, ['MIN', 'MAX']]

df_out.to_csv(directory + "/result.csv")

