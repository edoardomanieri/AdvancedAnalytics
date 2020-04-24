import shap
import pandas as pd
import utils
import numpy as np
from joblib import load
from sklearn.base import clone
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

%matplotlib agg

shap.initjs()

estimators = []
estimators.append(load("./estimators/min_estimator.joblib"))
estimators.append(load("./estimators/max_estimator.joblib"))
estimators.append(load("./estimators/dif_estimator.joblib"))

cat_vars = ['brand', 'cpu', 'cpu_details',
            'gpu', 'os', 'os_details', 'screen_surface']
one_hot_cat_vars = ['os', 'screen_surface', 'gpu_d',  'cpu_d', 'brand_d']
smooth_cat_vars = ['brand', 'os_details', 'cpu', 'gpu']
decrease_cat_vars = []
cols_to_be_dropped = ['name', 'base_name', 'cpu_details', 'pixels_y']
weights = [0.6, 0.4]

train_min = pd.read_csv("train.csv")
train_min = utils.preprocessing(train_min)
train_min.drop(columns=['max_price'], inplace=True)
test_min = pd.read_csv("test.csv")
test_min = utils.preprocessing(test_min)
df = utils.merge_train_test(train_min, test_min, 'min_price')


cat_vars = ['name', 'brand', 'base_name', 'cpu',
            'cpu_details', 'gpu', 'os', 'os_details', 'screen_surface']
dummy_vars = ['touchscreen', 'detachable_keyboard', 'discrete_gpu']
target_vars = ['min_price', 'max_price']
target = 'min_price'
num_vars = [col for col in df.columns if col not in cat_vars +
            dummy_vars + target_vars]
variable_lists = [cat_vars, dummy_vars, target_vars, num_vars]


df = utils.imputation(df)
utils.drop_columns(df, cols_to_be_dropped, variable_lists)
utils.decrease_cat_size_handling(df, decrease_cat_vars, target)
df = utils.one_hot_encoding(df, one_hot_cat_vars)
utils.smooth_handling(df, smooth_cat_vars, target)

estimator = clone(estimators[0])
X_train, y_train, X_test = utils.split_train_test_res(df, target, 'id')
estimator.fit(X_train, y_train)
X_train = df[df['train'] == 1].drop(columns=[target, 'id', 'train'])

explainer = shap.TreeExplainer(estimator)
shap_values = explainer.shap_values(X_train.values)

shap.force_plot(explainer.expected_value,
                shap_values[0, :], X_train.iloc[0, :])

shap.summary_plot(shap_values, X_train, show=False)
plt.tight_layout()
plt.savefig("summary_plot_shap.png")

shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("summary_plot_shap_bar.png")


# Permutation importance
result = permutation_importance(estimator, X_train.values, y_train, n_repeats=50,
                                random_state=42)
perm_sorted_idx = result.importances_mean.argsort()
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.boxplot(result.importances[perm_sorted_idx].T, vert=False,
           labels=X_train.columns[perm_sorted_idx])
plt.tight_layout()
fig.savefig("perm_imp.png")
