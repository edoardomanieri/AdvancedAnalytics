import numpy as np
import pandas as pd
from CustomImputer import DataFrameImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from joblib import dump
import itertools
import random
from sklearn.base import clone


def merge_train_test(train, test, target):
    train['train'] = 1
    test['train'] = 0
    test.insert(loc=0, column=target, value=[0 for _ in range(len(test))])
    return pd.concat([train, test])


def train_test_index(df, test_size=0.2, random_state=20):
    train, _ = train_test_split(df, test_size=0.2, random_state=random_state)
    df['train'] = 0
    df.loc[train.index, 'train'] = 1


def smooth_handling(df, cat_vars, on, report_file=None, m=10):
    for var in cat_vars:
        replace_dict, mean = calc_smooth_mean(df[df['train'] == 1], by=var, on=on, m=10)
        df.loc[df['train'] == 1, :] = df.loc[df['train'] == 1, :].replace(replace_dict)
        df.loc[df['train'] == 0, var] = np.where(df.loc[df['train'] == 0, var].isin(replace_dict.keys()), df.loc[df['train'] == 0, var].map(replace_dict), mean).astype(float)
    if report_file is not None:
        report_file.write(f"smooth mean to handle categorical variables. Params: m={m} \n")


def decrease_cat_size_handling(df, cat_vars, on):
    for var in cat_vars:
        replace_dict = reduce_categories(df[df['train'] == 1], var, on)
        df.replace(replace_dict, inplace=True)


def imputation(df, report_file=None):
    imputer = DataFrameImputer()
    imputer.fit(df[df['train'] == 1])
    if report_file is not None:
        report_file.write("replace nan with mode, mean and median \n")
    return imputer.transform(df)


def drop_columns(df, cols, variable_lists, report_file=None):
    df.drop(columns=cols, inplace=True)
    for col in cols:
        for l in variable_lists:
            if col in l:
                l.remove(col)
    if report_file is not None:
        report_file.write(f"removed columns: {cols} \n")


def one_hot_encoding(df, cat_vars):
    to_categorical(df, cat_vars)
    df = generate_dummies(df, cat_vars)
    return df


def split_train_test(df, target, id_col):
    y_train = df.loc[df['train'] == 1, target].values
    y_test = df.loc[df['train'] == 0, target].values
    X_train = df[df['train'] == 1].drop(columns=[target, id_col, 'train']).values
    X_test = df[df['train'] == 0].drop(columns=[target, id_col, 'train']).values
    return X_train, y_train, X_test, y_test


def split_train_test_res(df, target, id_col):
    y_train = df.loc[df['train'] == 1, target].values
    X_train = df[df['train'] == 1].drop(columns=[target, id_col, 'train']).values
    X_test = df[df['train'] == 0].drop(columns=[target, id_col, 'train']).values
    return X_train, y_train, X_test


def fit_predict(df, estimator, target, id_col, target_out, report_file=None):
    if report_file is not None:
        report_file.write(f"estimator: {estimator.__class__.__name__}, params: {estimator.get_params()} \n")
    X_train, y_train, X_test = split_train_test_res(df, target, 'id')
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    test = df[df['train'] == 0].drop(columns=['train'])
    test[target_out] = predictions
    df_out = test[['id', target_out]]
    return df_out


def fit_mae(df, estimator, target, id_col, target_out):
    X_train, y_train, X_test, y_test = split_train_test(df, target, 'id')
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    test = df[df['train'] == 0].drop(columns=['train'])
    test[target_out] = predictions
    test['TRUE_' + target_out] = y_test
    df_out = test[['id', target_out, 'TRUE_' + target_out]]
    return df_out, mae


def get_predictions(df, estimator, target, id_col, target_out, report_file=None):
    if report_file is not None:
        report_file.write("use min predictions to predict max. For train use y_train \n")
    X_train, y_train, X_test = split_train_test_res(df, target, 'id')
    estimator.fit(X_train, y_train)
    predictions_test = estimator.predict(X_test)
    predictions_train = y_train
    test = df[df['train'] == 0].drop(columns=['train'])
    train = df[df['train'] == 1].drop(columns=['train'])
    test[target_out] = predictions_test
    train[target_out] = predictions_train
    df_out = pd.concat([train, test]).loc[:, ['id', target_out]]
    return df_out


def calc_smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()
    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean']).reset_index()
    agg['smooth'] = (agg['count'] * agg['mean'] + m * mean) / (agg['count'] + m)
    d = pd.Series(agg['smooth'].values, index=agg[by]).to_dict()
    # Replace each value by the according smoothed mean
    return d, mean


def huber_approx_obj(preds, dtrain):
    d = preds - dtrain
    h = 1  #h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = -d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


def reduce_categories(df_train, col, on, min_cat_extension=10):
    replace_dict = {}
    # dict with category and mean of target variable
    dict_categories = df_train.groupby(col)[on].mean().to_dict()
    # sort it
    dict_categories = {k: v for k, v in sorted(dict_categories.items(), key=lambda item: item[1])}
    for cat in df_train[col].unique():
        # find category to be replaced
        if len(df_train.loc[df_train[col] == cat]) < (len(df_train) / min_cat_extension):
            min_dif = np.inf
            cat_replace = cat
            for value in dict_categories.keys():
                # find the best match based on the mean of the target variable
                # if len(df_train.loc[df_train[col] == value]) >= (len(df_train) / min_cat_extension):
                if len(df_train.loc[df_train[col] == value]) > len(df_train.loc[df_train[col] == cat]):
                    dif = abs(dict_categories[value] - dict_categories[cat])
                    if dif != 0 and dif <= min_dif:
                        min_dif = dif
                        cat_replace = value
            if cat_replace != cat:
                replace_dict[cat] = cat_replace
    already_done = []
    for key, value in replace_dict.items():
        if key not in already_done:
            replacement = value
            keys_to_modify = [key]
            while replacement in replace_dict.keys():
                keys_to_modify.append(replacement)
                replacement = replace_dict[replacement]
            for k in keys_to_modify:
                replace_dict[k] = replacement
                already_done.append(k)
    # return a replacement dict useful for train and test
    return replace_dict


def to_categorical(df, features_list):
    for feature in features_list:
        df[feature] = df[feature].astype('category')


def generate_dummies(df, features_list):
    df_new = df.copy()
    for feature in features_list:
        df_new = pd.concat([df_new, pd.get_dummies(df[feature], drop_first=True, prefix=feature)], axis=1).copy()
        df_new = df_new.drop(columns=[feature])
    return df_new


def CV_pipeline(df, estimator, cat_vars, cat_handler, target, cv=5, imputer=None, scaler=None):
    kf = KFold(n_splits=cv)
    mae_folds = []
    for train_index, _ in kf.split(df):
        df_temp = df.copy()
        df_temp['train'] = 0
        df_temp.loc[train_index, 'train'] = 1
        df_temp = imputation(df_temp)
        df_temp.drop(columns=['name', 'base_name', 'pixels_y', 'max_price'], inplace=True)
        cat_handler(df_temp, cat_vars, target)
        if cat_handler == decrease_cat_size_handling:
            df_temp = one_hot_encoding(df_temp, cat_vars)
        _, mae = fit_mae(df_temp, estimator, target, 'id', 'TMP')
        mae_folds.append(mae)
    return mae_folds, sum(mae_folds)/cv


def feature_engineering(df, report_file=None):
    if report_file is not None:
        report_file.write("isApple for dif \n")
    df['isApple'] = np.where(df['brand'] == 'Apple', 1, 0)


def full_CV_pipeline(df, estimator, cat_vars, cat_handler, weights=None, cv=5, imputer=None, scaler=None):
    kf = KFold(n_splits=cv)
    mae_folds = []
    if weights is None:
        weights = [0.5, 0.5]
    for train_index, _ in kf.split(df):
        df_temp = df.copy()

        df_temp['train'] = 0
        df_temp.loc[train_index, 'train'] = 1
        df_temp = imputation(df_temp)

        df_min = df_temp.drop(columns=['name', 'max_price'])
        cat_handler(df_min, cat_vars, 'min_price')
        if cat_handler == decrease_cat_size_handling:
            df_min = one_hot_encoding(df_min, cat_vars)
        df_min_out, _ = fit_mae(df_min, estimator, 'min_price', 'id', 'MIN')
        df_comp_min = get_predictions(df_min, estimator, 'min_price', 'id', 'min_price_pred')

        df_max = df_temp.drop(columns=['name', 'min_price'])
        df_max = df_max.merge(df_comp_min, on='id')
        cat_handler(df_max, cat_vars, 'max_price')
        if cat_handler == decrease_cat_size_handling:
            df_max = one_hot_encoding(df_max, cat_vars)
        df_max_out, _ = fit_mae(df_max, estimator, 'max_price', 'id', 'MAX')
        df_comp_max = get_predictions(df_max, estimator, 'max_price', 'id', 'max_price_pred')

        df_temp['dif'] = df_temp['max_price'] - df_temp['min_price']
        df_dif = df_temp.drop(columns=['name', 'min_price', 'max_price'])
        feature_engineering(df_dif)
        cat_handler(df_dif, cat_vars, 'dif')
        if cat_handler == decrease_cat_size_handling:
            df_dif = one_hot_encoding(df_dif, cat_vars)
        df_dif_out, _ = fit_mae(df_dif, estimator, 'dif', 'id', 'DIF')

        df_fin = df_min_out.merge(df_max_out, on="id")
        df_fin = df_fin.merge(df_dif_out, on="id") 
        df_fin['tmp'] = df_fin['MAX']*weights[0] + (df_fin['MIN'] + abs(df_fin['DIF']))*weights[1]
        df_fin['MIN'] = df_fin['MIN']*weights[0] + (df_fin['MAX'] - abs(df_fin['DIF']))*weights[1]
        df_fin['MAX'] = df_fin['tmp']
        df_fin.loc[df_fin['MAX'] < df_fin['MIN'], ['MIN', 'MAX']] = df_fin.loc[df_fin['MAX'] < df_fin['MIN'], ['MIN', 'MAX']].mean(axis=1)
        tot_mae = mean_absolute_error(df_fin['TRUE_MAX'], df_fin['MAX']) + mean_absolute_error(df_fin['TRUE_MIN'], df_fin['MIN'])
        mae_folds.append(tot_mae)
    return mae_folds, sum(mae_folds)/cv


def gridsearch_CV(df, estimator, cat_vars, cat_handler, param_dist, cv=5):
    m = np.inf
    best_params = {}
    for list_of_params in itertools.product(*param_dist.values()):
        param_dict = {x: y for x, y in zip(param_dist.keys(), list_of_params)}
        estimator.set_params(**param_dict)
        _, res = full_CV_pipeline(df, estimator, cat_vars, cat_handler, cv)
        print(param_dict)
        print(res)
        if res < m:
            m = res
            best_params = param_dict
    return m, best_params


def randomizedsearch_CV(df, estimator, cat_vars, cat_handler, param_dist, weights=None, cv=5, trials=20):
    m = np.inf
    best_params = {}
    param_dict_list = []
    for list_of_params in itertools.product(*param_dist.values()):
        param_dict = {x: y for x, y in zip(param_dist.keys(), list_of_params)}
        param_dict_list.append(param_dict)
    for _ in range(trials):
        param_dict = random.choice(param_dict_list)
        param_dict_list.remove(param_dict)
        estimator.set_params(**param_dict)
        _, res = full_CV_pipeline(df, estimator, cat_vars, cat_handler, cv=cv, weights=weights)
        print(param_dict)
        print(res)
        if res < m:
            m = res
            best_params = param_dict
    return m, best_params


def preprocessing(df):
    df = df.drop(df[~df['detachable_keyboard'].isin([0, 1])].index).reset_index(drop=True)
    df.loc[df['screen_surface'] == 'glossy', 'screen_surface'] = 'Glossy'
    df.loc[df['screen_surface'] == 'matte', 'screen_surface'] = 'Matte'
    df['gpu_d'] = 'other'
    df.loc[df['gpu'].str.contains('Intel').fillna(False), 'gpu_d'] = 'intel'
    df.loc[df['gpu'].str.contains('NVIDIA').fillna(False), 'gpu_d'] = 'nvidia'
    df.loc[df['gpu'].str.contains('AMD').fillna(False), 'gpu_d'] = 'amd'
    df['brand_d'] = 'other'
    df.loc[df['brand_d'].str.contains('Apple').fillna(False), 'brand_d'] = 'apple'
    df.loc[df['brand_d'].str.contains('Dell').fillna(False), 'brand_d'] = 'dell'
    df['cpu_d'] = 'other'
    df.loc[df['cpu_d'].str.contains('i7').fillna(False), 'cpu_d'] = 'i7'
    df.loc[df['cpu_d'].str.contains('i5').fillna(False), 'cpu_d'] = 'i5'
    return df


def full_CV_pipeline_m(df, estimators, col_to_drop, one_hot_cat_vars, smooth_cat_vars, decrease_cat_vars, weights=None, cv=5):
    df = preprocessing(df)
    kf = KFold(n_splits=cv, random_state=10)
    mae_folds = []
    if weights is None:
        weights = [0.5, 0.5]
    for train_index, _ in kf.split(df):
        df_temp = df.copy()
        df_temp['train'] = 0
        df_temp.loc[train_index, 'train'] = 1
        df_temp = imputation(df_temp)

        df_min = df_temp.drop(columns= col_to_drop + ['max_price'])
        decrease_cat_size_handling(df_min, decrease_cat_vars, 'min_price')
        smooth_handling(df_min, smooth_cat_vars, 'min_price')
        df_min = one_hot_encoding(df_min, one_hot_cat_vars)
        df_min_out, _ = fit_mae(df_min, estimators[0], 'min_price', 'id', 'MIN')
        df_comp_min = get_predictions(df_min, estimators[0], 'min_price', 'id', 'min_price_pred')

        df_max = df_temp.drop(columns= col_to_drop + ['min_price'])
        df_max = df_max.merge(df_comp_min, on='id')
        decrease_cat_size_handling(df_max, decrease_cat_vars, 'max_price')
        smooth_handling(df_max, smooth_cat_vars, 'max_price')
        df_max = one_hot_encoding(df_max, one_hot_cat_vars)
        df_max_out, _ = fit_mae(df_max, estimators[1], 'max_price', 'id', 'MAX')
        df_comp_max = get_predictions(df_max, estimators[1], 'max_price', 'id', 'max_price_pred')

        df_temp['dif'] = df_temp['max_price'] - df_temp['min_price']
        df_dif = df_temp.drop(columns=col_to_drop + ['min_price', 'max_price'])
        decrease_cat_size_handling(df_dif, decrease_cat_vars, 'dif')
        smooth_handling(df_dif, smooth_cat_vars, 'dif')
        df_dif = one_hot_encoding(df_dif, one_hot_cat_vars)
        df_dif_out, _ = fit_mae(df_dif, estimators[2], 'dif', 'id', 'DIF')

        df_fin = df_min_out.merge(df_max_out, on="id")
        df_fin = df_fin.merge(df_dif_out, on="id")
        df_fin['tmp'] = df_fin['MAX']*weights[0] + (df_fin['MIN'] + abs(df_fin['DIF']))*weights[1]
        df_fin['MIN'] = df_fin['MIN']*weights[0] + (df_fin['MAX'] - abs(df_fin['DIF']))*weights[1]
        df_fin['MAX'] = df_fin['tmp']
        df_fin.loc[df_fin['MAX'] < df_fin['MIN'], ['MIN', 'MAX']] = df_fin.loc[df_fin['MAX'] < df_fin['MIN'], ['MIN', 'MAX']].mean(axis=1)
        tot_mae = mean_absolute_error(df_fin['TRUE_MAX'], df_fin['MAX']) + mean_absolute_error(df_fin['TRUE_MIN'], df_fin['MIN'])
        mae_folds.append(tot_mae)
    return mae_folds, sum(mae_folds)/cv


def randomizedsearch_CV_m(df, estimators, col_to_drop, one_hot_cat_vars, smooth_cat_vars, decrease_cat_vars, param_dists, weights=None, cv=5, trials=20):
    m = np.inf
    best_params = {}
    best_estimators = []
    param_dict_list_min = []
    param_dict_list_max = []
    param_dict_list_dif = []
    for list_of_params in itertools.product(*param_dists[0].values()):
        param_dict = {x: y for x, y in zip(param_dists[0].keys(), list_of_params)}
        param_dict_list_min.append(param_dict)
    for list_of_params in itertools.product(*param_dists[1].values()):
        param_dict = {x: y for x, y in zip(param_dists[1].keys(), list_of_params)}
        param_dict_list_max.append(param_dict)
    for list_of_params in itertools.product(*param_dists[2].values()):
        param_dict = {x: y for x, y in zip(param_dists[2].keys(), list_of_params)}
        param_dict_list_dif.append(param_dict)
    for _ in range(trials):
        param_dict_min = random.choice(param_dict_list_min)
        param_dict_max = random.choice(param_dict_list_max)
        param_dict_dif = random.choice(param_dict_list_dif)
        estimators[0].set_params(**param_dict_min)
        estimator_min = clone(estimators[0])
        estimators[1].set_params(**param_dict_max)
        estimator_max = clone(estimators[1])
        estimators[2].set_params(**param_dict_dif)
        estimator_dif = clone(estimators[2])
        folds, res = full_CV_pipeline_m(df, [estimator_min, estimator_max, estimator_dif], col_to_drop, one_hot_cat_vars, smooth_cat_vars, decrease_cat_vars, cv=cv, weights=weights)
        print(param_dict_min)
        print(param_dict_max)
        print(param_dict_dif)
        print(res)
        print(folds)
        print(_)
        if res < m:
            m = res
            best_params = [param_dict_min, param_dict_max, param_dict_dif]
            best_estimators = [estimator_min, estimator_max, estimator_dif]
    print(m)
    print(best_params)
    return best_estimators


def save_estimators(estimators):
    dump(estimators[0], './estimators/min_estimator.joblib')
    dump(estimators[1], './estimators/max_estimator.joblib')
    dump(estimators[2], './estimators/dif_estimator.joblib')