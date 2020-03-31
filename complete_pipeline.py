import pandas as pd
import utils
from pathlib import Path
from datetime import datetime
from joblib import load
from sklearn.base import clone


directory = "./results/" + datetime.now().strftime("%m-%d-%Y,%H:%M:%S")
Path(directory).mkdir(parents=True, exist_ok=True)
report_file = open(directory + "/report", "w+")

estimators = []
estimators.append(load("./estimators/min_estimator.joblib"))
estimators.append(load("./estimators/max_estimator.joblib"))
estimators.append(load("./estimators/dif_estimator.joblib"))


cat_vars = ['brand', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
one_hot_cat_vars = ['os', 'screen_surface']
smooth_cat_vars = ['brand', 'os_details', 'cpu', 'gpu']
decrease_cat_vars = []
cols_to_be_dropped = ['name', 'base_name', 'cpu_details']
weights = [0.3, 0.7]


# pre-CV-validate
df = pd.read_csv("train.csv")
_, mae = utils.full_CV_pipeline_m(df, estimators, cols_to_be_dropped, one_hot_cat_vars, smooth_cat_vars, decrease_cat_vars, weights=weights)
report_file.write(f"CV Score: {mae} \n\n\n")


##### min_price
report_file.write("MIN PRICE \n")
train_min = pd.read_csv("train.csv")
train_min = train_min.drop(train_min[~train_min['detachable_keyboard'].isin([0, 1])].index).reset_index()
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
utils.drop_columns(df, cols_to_be_dropped, variable_lists, report_file=report_file)
utils.decrease_cat_size_handling(df, decrease_cat_vars, target)
df = utils.one_hot_encoding(df, one_hot_cat_vars)
utils.smooth_handling(df, smooth_cat_vars, target, report_file=report_file)

estimator = clone(estimators[0])
df_min = utils.fit_predict(df, estimator, target, 'id', 'MIN', report_file=report_file)
df_comp_min = utils.get_predictions(df, estimator, target, 'id', 'min_price_pred')
report_file.write("\n\n\n")

##### max_price
report_file.write("MAX PRICE \n")
train_max = pd.read_csv("train.csv")
train_max = train_max.drop(train_max[~train_max['detachable_keyboard'].isin([0,1])].index).reset_index()
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
utils.drop_columns(df, cols_to_be_dropped, variable_lists, report_file=report_file)
df = df.merge(df_comp_min, on='id')
utils.decrease_cat_size_handling(df, decrease_cat_vars, target)
df = utils.one_hot_encoding(df, one_hot_cat_vars)
utils.smooth_handling(df, smooth_cat_vars, target, report_file=report_file)

estimator = clone(estimators[1])
df_max = utils.fit_predict(df, estimator, target, 'id', 'MAX', report_file=report_file)
df_comp_max = utils.get_predictions(df, estimator, target, 'id', 'max_price_pred')
report_file.write("\n\n\n")

##### difference
report_file.write("DIFFERENCE \n")
train_dif = pd.read_csv("train.csv")
train_dif = train_dif.drop(train_dif[~train_dif['detachable_keyboard'].isin([0,1])].index).reset_index()
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
utils.drop_columns(df, cols_to_be_dropped, variable_lists, report_file=report_file)
# utils.feature_engineering(df, report_file=report_file)
utils.decrease_cat_size_handling(df, decrease_cat_vars, target)
df = utils.one_hot_encoding(df, one_hot_cat_vars)
utils.smooth_handling(df, smooth_cat_vars, target, report_file=report_file)

estimator = clone(estimators[2])
df_dif = utils.fit_predict(df, estimator, target, 'id', 'DIF', report_file=report_file)
report_file.write("\n\n\n")

# put together predictions
report_file.write("MERGING \n")

df_out = df_min.merge(df_max, on="id").set_index("id")
df_out = df_out.merge(df_dif, on="id").set_index("id")
df_out['MAX'] = df_out['MAX']*weights[0] + (df_out['MIN'] + abs(df_out['DIF']))*weights[1]
df_out['MIN'] = df_out['MIN']*weights[0] + (df_out['MAX'] - abs(df_out['DIF']))*weights[1]
df_out.loc[df_out['MAX'] < df_out['MIN'], ['MIN', 'MAX']] = df_out.loc[df_out['MAX'] < df_out['MIN'], ['MIN', 'MAX']].mean(axis=1)
df_out = df_out.loc[:, ['MIN', 'MAX']]
report_file.write("replace values where max smaller than min with mean of two values \n")
report_file.write("prediction is average of price predictions and predictions with difference \n")
report_file.close()
df_out.to_csv(directory + "/result.csv")