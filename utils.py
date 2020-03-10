import numpy as np
import pandas as pd


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
    l = len(df_train[col].unique())
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
    df1 = df.copy()
    for feature in features_list:
        df1[feature] = df1[feature].astype('category',copy=False)
    return df1

def generate_dummies(df, features_list):
    df_new = df.copy()
    for feature in features_list:
        df_new = pd.concat([df_new,pd.get_dummies(df[feature], drop_first=True, prefix=feature)], axis=1).copy()
        df_new = df_new.drop(columns=[feature])
    return df_new
