import os
os.chdir(os.path.abspath("./.."))

import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Masking, Dropout, Dense, Conv1D, GRU, LSTM, BatchNormalization
import tensorflow_addons as tfa

tf.config.run_functions_eagerly(True)

import gc
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback

from src.dx5849_training_data_3src import *


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


def data_zero_masking(data):
    data[np.isnan(data)] = 0
    return data

def build_model(kernel_size=10, N_filter_per_feature=3):
    # Model Setup
    static_input_shape = (4)
    time_series_shape = (144, 25)

    static_inputs = Input(shape=static_input_shape, dtype='float64', name='static_inputs')
    time_series_inputs = Input(shape=time_series_shape, dtype='float64', name='time_series_inputs')

    x = time_series_inputs
    x = Masking(mask_value=.0)(x)
    x = BatchNormalization(axis=2, name='norm_input')(x)
    # print('Masked Input: ', x.shape)
    x = Conv1D(filters=N_filter_per_feature*25, kernel_size=kernel_size, groups=25,
               activation='relu',
               name='conv1d')(x)
    # print('Conv1D: ', x.shape)

    x = tf.reshape(x, [-1, x.shape[1]*x.shape[2]])
    x = layers.concatenate([static_inputs, x])
    # print('Concanated input for Dense layers: ', x.shape)

    x = Dense(64, activation='relu', name='dense1')(x)
    # print('Dense1: ', x.shape)
    x = Dropout(0.5)(x)
    x = BatchNormalization(name='norm_dense1')(x)

    x = Dense(32, activation='relu', name='dense2')(x)
    # print('Dense2: ', x.shape)
    x = Dropout(0.5)(x)
    x = BatchNormalization(name='norm_dense2')(x)

    output = Dense(1, activation='sigmoid', name='output')(x)
    # print('Output: ', output.shape)

    model = keras.Model(inputs=[static_inputs, time_series_inputs],
                        outputs=[output])
    return model

root_folder = os.path.abspath('.')
data_folder = os.path.join(root_folder, 'processed/patient_data/dx_pred_6_12_3src')

X_train, X_test, y_train, y_test = np.load(os.path.join(data_folder, 'data_split_8020.npy'), allow_pickle=True)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
# X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=.5, random_state=5849, shuffle=True, stratify=y_test)

info_test, data_test, label_test = np.load(os.path.join(data_folder, 'test_data.npy'), allow_pickle=True)
info_test = data_zero_masking(info_test)
data_test = data_zero_masking(data_test)[:,:,1:]

N_val = int(len(label_test)/2)
dataset_val = ({'static_inputs': info_test[:N_val], 'time_series_inputs': data_test[:N_val]}, label_test[:N_val])
dataset_test = ({'static_inputs': info_test[N_val:], 'time_series_inputs': data_test[N_val:]}, label_test[N_val:])


EPOCH = 100
BATCH_SIZE = 128
# ds_test = tf.data.experimental.load('processed/patient_data/dx_pred_0_12_3src/tf_dataset/dataset_test')
ds_pos = tf.data.experimental.load(os.path.join(data_folder, 'tf_dataset/dataset_train_pos'))
ds_neg = tf.data.experimental.load(os.path.join(data_folder, 'tf_dataset/dataset_train_neg'))
ds_train = generate_equally_resampled_tf_dataset(ds_pos, ds_neg)
ds_train = ds_train.batch(BATCH_SIZE).prefetch(2)
STEP_PER_EPOCH = np.ceil(2 * ds_neg.cardinality().numpy() / BATCH_SIZE)

print(tf.test.gpu_device_name())


model = build_model()
model.summary()

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
# log_names = ['loss', 'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 'auc', 'prc']

# LOSS = keras.losses.BinaryCrossentropy(name='loss')
LOSS = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.75, gamma=0, name='loss')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CALLBACKS = [
    # keras.callbacks.EarlyStopping(
    #     monitor="val_loss",
    #     min_delta=1e-5,
    #     patience=5,
    #     verbose=1,
    # ),

    keras.callbacks.ModelCheckpoint(
        filepath=f'./models/Conv1D/models/{current_time}',
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_freq='epoch',
    ),

    keras.callbacks.TensorBoard(log_dir=f"./models/Conv1D/logs/{current_time}"),

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
np.save(os.path.join(os.path.abspath("/home/dhm/workspace/conference_projects/MICCAI2022_AKF_feature_importance/models/Conv1D/logs"), current_time, 'train_history'), allow_pickle=True)


# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#
# train_log_dir = 'models/Conv1D/logs/' + current_time + '/train'
# test_log_dir = 'models/Conv1D/logs/' + current_time + '/test'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)
#
# class_weight = {
#     0: sum(y_train) / len(y_train),
#     1: 1 - sum(y_train) / len(y_train),
# }
#
# batches_processed_counter = 0
#
# for epoch_i in range(epoch):
#     batch_generator = generate_balanced_data_batch_raw(X_train, y_train,
#                                               batchsize=batch_size,
#                                               resample_freq=None)
#
#     for batch_i, (info, data, label) in enumerate(batch_generator):
#         batches_processed_counter = batches_processed_counter + 1
#         info = data_zero_masking(info)
#         data = data_zero_masking(data)[:, :, 1:]
#
#         log_train = model.train_on_batch({'static_inputs': info, 'time_series_inputs': data},
#                                          label,
#                                          reset_metrics=False, return_dict=True)
#
#         with train_summary_writer.as_default():
#             for log in log_train:
#                 tf.summary.scalar(log, log_train[log], step=batches_processed_counter)
#
#
#     log_test = model.evaluate({'static_inputs': info_test, 'time_series_inputs': data_test},
#                               {'output': label_test}, return_dict=True)
#
#     with test_summary_writer.as_default():
#         for log in log_test:
#             tf.summary.scalar(log, log_test[log], step=batches_processed_counter)
#
#     print(f'Epoch: {epoch_i}')
#     print('\tTrainLoss: ', log_train['loss'], 'TrainAcc: ', log_train['accuracy'], 'TrainAUC: ',
#           log_train['auc'])
#     print('\tTestLoss: ', log_test['loss'], 'TestAcc: ', log_test['accuracy'], 'TestAUC: ', log_test['auc'],
#           '\n')
#
# model.save('models/Conv1D/logs/' + current_time + '/model')

