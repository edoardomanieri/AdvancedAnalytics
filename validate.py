import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import utils
from sklearn.base import clone


cat_vars = ['brand', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
one_hot_cat_vars = ['os', 'screen_surface', 'brand']
smooth_cat_vars = ['os_details', 'cpu', 'gpu']
decrease_cat_vars = []
cols_to_be_dropped = ['name', 'base_name',  'cpu_details']
weights = [0.3, 0.7]

# CV

params = {
        'learning_rate': [0.05, 0.1, 0.15, 0.20, 0.3],
        'n_estimators': [50, 100, 200, 300, 500],
        'min_child_weight': [3, 5, 10, 15],
        'gamma': [0.3, 0.5, 0.7, 1, 1.5],
        'subsample': [0.4, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'max_depth': [2, 3, 4, 5]
        }

df = pd.read_csv("train.csv")
cat_vars = ['brand', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
xgb_reg = xgb.XGBRegressor()
estimators = utils.randomizedsearch_CV_m(df, [xgb_reg, xgb_reg, xgb_reg], cols_to_be_dropped, one_hot_cat_vars, smooth_cat_vars, decrease_cat_vars, [params, params, params], weights=weights, trials=1)
utils.save_estimators(estimators)

params_min = {
        'learning_rate': [0.03, 0.05, 0.07, 0.1],
        'n_estimators': [100, 200, 300, 400],
        'min_child_weight': [3, 5, 7],
        'gamma': [0.1, 0.3, 0.5, 1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

params_max = {
        'learning_rate': [0.05, 0.10, 0.15, 0.20],
        'n_estimators': [50, 100, 200, 300],
        'min_child_weight': [3, 5, 7, 9],
        'gamma': [0.3, 0.5, 1, 1.5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
        'max_depth': [2, 3, 4]
        }

params_dif = {
        'learning_rate': [0.10, 0.15, 0.20, 0.3, 0.35],
        'n_estimators': [150, 250, 350, 450],
        'min_child_weight': [7, 10, 13, 15],
        'gamma': [1, 1.5, 1.8, 2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

df = pd.read_csv("train.csv")
cat_vars = ['brand', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
xgb_reg = xgb.XGBRegressor()
estimators = utils.randomizedsearch_CV_m(df, [xgb_reg, xgb_reg, xgb_reg], cols_to_be_dropped, one_hot_cat_vars,
                                         smooth_cat_vars, decrease_cat_vars, [params_min, params_max, params_dif], weights=weights, trials=1)
utils.save_estimators(estimators)
# utils.gridsearch_CV(df, xgb_reg, cat_vars, utils.smooth_handling, params)
# utils.full_CV_pipeline(df, xgb_reg, cat_vars, utils.smooth_handling, weights=[0.4, 0.6])


df = pd.read_csv("train.csv")
df = df.drop(df[~df['detachable_keyboard'].isin([0, 1])].index).reset_index(drop=True)
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
utils.drop_columns(df_min_in, cols_to_be_dropped + ['max_price'], variable_lists)
utils.decrease_cat_size_handling(df_min_in, decrease_cat_vars, target)
df_min_in = utils.one_hot_encoding(df_min_in, one_hot_cat_vars)
utils.smooth_handling(df_min_in, smooth_cat_vars, target)

estimator = clone(estimators[0])

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
utils.drop_columns(df_max_in, cols_to_be_dropped + ['min_price'], variable_lists)
df_max_in = df_max_in.merge(df_comp_min, on='id')

utils.decrease_cat_size_handling(df_max_in, decrease_cat_vars, target)
df_max_in = utils.one_hot_encoding(df_max_in, one_hot_cat_vars)
utils.smooth_handling(df_max_in, smooth_cat_vars, target)

estimator = clone(estimators[1])

df_max, mae_max = utils.fit_mae(df_max_in, estimator, target, 'id', 'MAX')
df_comp_max = utils.get_predictions(df_max_in, estimator, target, 'id', 'max_price_pred')

# difference


df_dif_in = df.copy()
df_dif_in['dif'] = df_dif_in['max_price'] - df_dif_in['min_price']

cat_vars = ['name', 'brand', 'base_name', 'cpu', 'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price', 'dif']
target = 'dif'
num_vars = [col for col in df.columns if col not in cat_vars + dummy_vars + target_vars]
variable_lists = [cat_vars, dummy_vars, target_vars, num_vars]

df_dif_in = utils.imputation(df_dif_in)
utils.drop_columns(df_dif_in, cols_to_be_dropped + ['min_price', 'max_price'], variable_lists)
#df_dif_in = df_dif_in.merge(df_comp_min, on='id')
#df_dif_in = df_dif_in.merge(df_comp_max, on='id')
utils.decrease_cat_size_handling(df_dif_in, decrease_cat_vars, target)
df_dif_in = utils.one_hot_encoding(df_dif_in, one_hot_cat_vars)
utils.smooth_handling(df_dif_in, smooth_cat_vars, target)

estimator = clone(estimators[2])
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
df_out['MAX'] = df_out['MAX']*weights[0] + (df_out['MIN'] + abs(df_out['DIF']))*weights[1]
df_out['MIN'] = df_out['MIN']*weights[0] + (df_out['MAX'] - abs(df_out['DIF']))*weights[1]
df_out.loc[df_out['MAX'] < df_out['MIN'], ['MIN', 'MAX']] = df_out.loc[df_out['MAX'] < df_out['MIN'], ['MIN', 'MAX']].mean(axis=1)
tot_mae = mean_absolute_error(df_out['TRUE_MAX'], df_out['MAX']) + mean_absolute_error(df_out['TRUE_MIN'], df_out['MIN'])