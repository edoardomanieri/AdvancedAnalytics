import pandas as pd
import xgboost as xgb
import utils
from pathlib import Path
from datetime import datetime


directory = "./results/" + datetime.now().strftime("%m-%d-%Y,%H:%M:%S")
Path(directory).mkdir(parents=True, exist_ok=True)
report_file = open(directory + "/report", "w+")

# pre-CV-validate
df = pd.read_csv("train.csv")
cat_vars = ['brand', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
estimator_1 = xgb.XGBRegressor(n_estimators=300, max_depth=3, gamma=0.3, colsample_bytree=0.6, subsample=0.8, min_child_weight=15)
estimator_2 = xgb.XGBRegressor(n_estimators=300, max_depth=3, gamma=1.5, colsample_bytree=1, subsample=1, min_child_weight=10)
estimator_3 = xgb.XGBRegressor(n_estimators=200, max_depth=3, gamma=0.5, colsample_bytree=0.6, subsample=0.6, min_child_weight=10)
estimators = [estimator_1, estimator_2, estimator_3]
_, mae = utils.full_CV_pipeline_m(df, estimators, cat_vars, utils.smooth_handling, weights=[0.4, 0.6])
report_file.write(f"CV Score: {mae} \n\n\n")


##### min_price
report_file.write("MIN PRICE \n")
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

df = utils.imputation(df, report_file=report_file)
utils.drop_columns(df, ['name', 'base_name', 'pixels_y'], variable_lists, report_file=report_file)
utils.smooth_handling(df, cat_vars, target, report_file=report_file)

estimator = xgb.XGBRegressor(n_estimators=300, max_depth=3, gamma=0.3, colsample_bytree=0.6, subsample=0.8, min_child_weight=15)
df_min = utils.fit_predict(df, estimator, target, 'id', 'MIN', report_file=report_file)
df_comp_min = utils.get_predictions(df, estimator, target, 'id', 'min_price_pred')
report_file.write("\n\n\n")

##### max_price
report_file.write("MAX PRICE \n")
train_max = pd.read_csv("train.csv")
train_max.drop(columns=['min_price'], inplace=True)
test_max = pd.read_csv("test.csv")
df = utils.merge_train_test(train_max, test_max, 'max_price')

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price']
target = 'max_price'
num_vars = [col for col in df.columns if col not in cat_vars + dummy_vars + target_vars]
variable_lists = [cat_vars, dummy_vars, target_vars, num_vars]

df = utils.imputation(df, report_file=report_file)
utils.drop_columns(df, ['name', 'base_name', 'pixels_y'], variable_lists, report_file=report_file)
df = df.merge(df_comp_min, on='id')
utils.smooth_handling(df, cat_vars, target, report_file=report_file)

estimator = xgb.XGBRegressor(n_estimators=300, max_depth=3, gamma=1.5, colsample_bytree=1, subsample=1, min_child_weight=10)
df_max = utils.fit_predict(df, estimator, target, 'id', 'MAX', report_file=report_file)
df_comp_max = utils.get_predictions(df, estimator, target, 'id', 'max_price_pred')
report_file.write("\n\n\n")

##### difference
report_file.write("DIFFERENCE \n")
train_dif = pd.read_csv("train.csv")
train_dif['dif'] = train_dif['max_price'] - train_dif['min_price']
train_dif.drop(columns=['min_price', 'max_price'], inplace=True)
test_dif = pd.read_csv("test.csv")
df = utils.merge_train_test(train_dif, test_dif, 'dif')

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price', 'dif']
target = 'dif'
num_vars = [col for col in df.columns if col not in cat_vars + dummy_vars + target_vars]
variable_lists = [cat_vars, dummy_vars, target_vars, num_vars]

df = utils.imputation(df, report_file=report_file)
utils.drop_columns(df, ['name', 'base_name', 'pixels_y'], variable_lists, report_file=report_file)
utils.smooth_handling(df, cat_vars, target, report_file=report_file)

estimator = xgb.XGBRegressor(n_estimators=200, max_depth=3, gamma=0.5, colsample_bytree=0.6, subsample=0.6, min_child_weight=10)
df_dif = utils.fit_predict(df, estimator, target, 'id', 'DIF', report_file=report_file)
report_file.write("\n\n\n")

# put together predictions
report_file.write("MERGING \n")

df_out = df_min.merge(df_max, on="id").set_index("id")
df_out = df_out.merge(df_dif, on="id").set_index("id")
df_out['MAX'] = df_out['MAX']*0.4 + (df_out['MIN'] + abs(df_out['DIF']))*0.6
df_out['MIN'] = df_out['MIN']*0.4 + (df_out['MAX'] - abs(df_out['DIF']))*0.6
df_out.loc[df_out['MAX'] < df_out['MIN'], ['MIN', 'MAX']] = df_out.loc[df_out['MAX'] < df_out['MIN'], ['MIN', 'MAX']].mean(axis=1)
df_out = df_out.loc[:, ['MIN', 'MAX']]
report_file.write("replace values where max smaller than min with mean of two values \n")
report_file.write("prediction is average of price predictions and predictions with difference \n")
report_file.close()
df_out.to_csv(directory + "/result.csv")