import os
os.chdir(os.path.abspath("./.."))

import numpy as np
import matplotlib.pyplot as plt
import datetime

import tensorboard
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

import shap
import lime

def data_zero_masking(data):
    data[np.isnan(data)] = 0
    return data

logs = os.listdir('./models/Conv1D/logs/')
log_ts = [datetime.datetime.strptime(log_ts[:15], "%Y%m%d-%H%M%S") for log_ts in logs]
log_latest = logs[log_ts.index(max(log_ts))]

# %load_ext tensorboard
# %tensorboard --logdir ./models/Conv1D/logs/{log_latest}

model = keras.models.load_model(f"models/Conv1D/models/{log_latest}")

ds_pos = tf.data.experimental.load('processed/patient_data/dx_pred_0_12_3src/tf_dataset/dataset_train_pos')
ds_neg = tf.data.experimental.load('processed/patient_data/dx_pred_0_12_3src/tf_dataset/dataset_train_neg')

X = ds_pos.take(500).concatenate(ds_neg.take(500))
X = list(X.as_numpy_iterator())

Xt = [x[0]['time_series_inputs'] for x in X]
Xs = [x[0]['static_inputs'] for x in X]
Xt = np.array(Xt)
Xs = np.array(Xs)

data_folder = 'processed/patient_data/dx_pred_0_12_3src'

info_test, data_test, label_test = np.load(os.path.join(data_folder, 'test_data.npy'), allow_pickle=True)
info_test = data_zero_masking(info_test)
data_test = data_zero_masking(data_test)[:,:,1:]

N_val = int(len(label_test)/2)
# dataset_val = ({'static_inputs': info_test[:N_val], 'time_series_inputs': data_test[:N_val]}, label_test[:N_val])
dataset_test = ({'static_inputs': info_test[N_val:], 'time_series_inputs': data_test[N_val:]}, label_test[N_val:])
Xt_test = dataset_test[0]['time_series_inputs']
Xs_test = dataset_test[0]['static_inputs']

shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
explainer = shap.DeepExplainer(model, [Xs, Xt])
shap_values = explainer.shap_values([Xs_test[:10], Xt_test[:10]])

feature_names = [
    'Cr', 'NIBP_d','RBC', 'NIBP_s', 'GLC', 'MPV', 'MCHC',
    'UREA', 'MCV', 'NA', 'MCH', 'RDW', 'HR', 'WBC', 'CL',
    'SAO2', 'PLT', 'CA', 'HCO3', 'NIBP_m', 'URINE', 'MG', 'K', 'RESPIRATION', 'HBG'
]
shap.initjs()
shap.summary_plot(
    shap_values[0][1],
    Xt_test[:10],
    feature_names=np.array(feature_names),
    max_display=50,
    plot_type='bar')

print(111)