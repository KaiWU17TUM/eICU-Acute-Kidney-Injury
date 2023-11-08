import os

os.chdir(os.path.abspath("./.."))

import datetime

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input, Masking, Dropout, Dense, Conv1D, GRU, LSTM, BatchNormalization
import tensorflow_addons as tfa


tf.config.run_functions_eagerly(True)

import gc
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.callbacks import Callback

from src.dx5849_training_data_3src import *


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


def data_zero_masking(data):
    data[np.isnan(data)] = 0
    return data


def build_model(kernel_size=10, n_filter=3, dense_layer=[128]):
    # Model Setup
    static_input_shape = (4)
    time_series_shape = (144, 25)

    static_inputs = Input(shape=static_input_shape, dtype='float64', name='static_inputs')
    time_series_inputs = Input(shape=time_series_shape, dtype='float64', name='time_series_inputs')

    x = time_series_inputs
    x = Masking(mask_value=.0)(x)
    x = BatchNormalization(axis=2, name='norm_input')(x)
    # print('Masked Input: ', x.shape)
    x = Conv1D(filters=n_filter * 25, kernel_size=kernel_size, groups=25,
               activation='relu',
               name='conv1d')(x)
    # print('Conv1D: ', x.shape)

    x = tf.reshape(x, [-1, x.shape[1] * x.shape[2]])
    x = layers.concatenate([static_inputs, x])
    # print('Concanated input for Dense layers: ', x.shape)

    for dense_i, dense_size in enumerate(dense_layer):
        x = Dense(dense_size, activation='relu', name=f'dense_{dense_i}')(x)
        # print('Dense1: ', x.shape)
        x = Dropout(0.5)(x)
        x = BatchNormalization(name=f'norm_dense_{dense_i}')(x)

    output = Dense(1, activation='sigmoid', name='output')(x)
    # print('Output: ', output.shape)

    model = keras.Model(inputs=[static_inputs, time_series_inputs],
                        outputs=[output])
    return model

root_folder = os.path.abspath('.')
data_folder = os.path.join(root_folder, 'processed/patient_data/dx_pred_0_12_3src')

_, X_test, _, y_test = np.load(os.path.join(data_folder, 'data_split_8020.npy'), allow_pickle=True)
X_test = np.array(X_test)
y_test = np.array(y_test)

info_test, data_test, label_test = np.load(os.path.join(data_folder, 'test_data.npy'), allow_pickle=True)
info_test = data_zero_masking(info_test)
data_test = data_zero_masking(data_test)[:,:,1:]

N_val = int(len(label_test)/2)
dataset_val = ({'static_inputs': info_test[:N_val], 'time_series_inputs': data_test[:N_val]}, label_test[:N_val])
dataset_test = ({'static_inputs': info_test[N_val:], 'time_series_inputs': data_test[N_val:]}, label_test[N_val:])

EPOCH = 100
BATCH_SIZE = 256
# ds_test = tf.data.experimental.load('processed/patient_data/dx_pred_0_12_3src/tf_dataset/dataset_test')
ds_pos = tf.data.experimental.load(os.path.join(data_folder, 'tf_dataset/dataset_train_pos'))
ds_neg = tf.data.experimental.load(os.path.join(data_folder, 'tf_dataset/dataset_train_neg'))
ds_train = generate_equally_resampled_tf_dataset(ds_pos, ds_neg)
ds_train = ds_train.batch(BATCH_SIZE).prefetch(2)
STEP_PER_EPOCH = np.ceil(2 * ds_neg.cardinality().numpy() / BATCH_SIZE)

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

LOSS = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.75, gamma=2, name='loss')

FILTER_NUMS = [1, 2, 3, 4, 5]
KERNEL_SIZES = [10, 20, 30, 40, 50]
DENSE_LAYERS = [[128], [256], [128, 64]]

for n_filter in FILTER_NUMS:
    for kernel_size in KERNEL_SIZES:
        for dense_layer in DENSE_LAYERS:
            model = build_model(n_filter=n_filter, kernel_size=kernel_size, dense_layer=dense_layer)
            model.summary()

            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            CALLBACKS = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-5,
                    patience=5,
                    verbose=1,
                ),

                keras.callbacks.ModelCheckpoint(
                    filepath=f'./models/Conv1D/models/{current_time}_F{n_filter}_K{kernel_size}_D{dense_layer}',
                    save_weights_only=False,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    save_freq='epoch',
                ),

                keras.callbacks.TensorBoard(
                    log_dir=f"./models/Conv1D/logs/{current_time}_F{n_filter}_K{kernel_size}_D{dense_layer}"),

                ClearMemory(),
            ]

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss=LOSS,
                metrics=METRICS,
                run_eagerly=True,
            )

            history = model.fit(
                ds_train,
                epochs=EPOCH,
                steps_per_epoch=STEP_PER_EPOCH,
                # validation_steps=int(0.1 * STEP_PER_EPOCH),
                # validation_freq=1,
                validation_data=(dataset_val[0], dataset_val[1]),
                callbacks=CALLBACKS,
            )
            np.save(os.path.join(os.path.abspath("./models/Conv1D/logs"),
                                 f'{current_time}_F{n_filter}_K{kernel_size}_D{dense_layer}', 'train_history'),
                    history,
                    allow_pickle=True)

            del model