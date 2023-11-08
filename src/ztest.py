import os
from pathlib import Path
os.chdir(os.path.join(Path(__file__).parent.resolve(), ".."))

import sys
sys.path.append(os.path.dirname(os.path.realpath(Path(__file__).parent.resolve())))

import numpy as np
import pandas as pd
from tqdm import tqdm

# from create_dataset_torch import *

from statsmodels.stats.weightstats import CompareMeans
import statsmodels.stats.weightstats as ws

# ############################# torch-py36 env => create data for z test #####################################
# def generate_stacked_dataset(data_folder, X, y, s_names, ts_names, imp=None, resample=None):
#     data_len = 144
#
#     ds = PatientDataset(data_folder, X, y, imputation=imp, resample_rate=resample)
#     loader = DataLoader(ds, batch_size=128, num_workers=8)
#
#     data_pos = {}
#     data_neg = {}
#     for fn in s_names:
#         data_pos[fn] = []
#         data_neg[fn] = []
#     for fn in ts_names:
#         data_pos[fn] = []
#         data_neg[fn] = []
#
#     for data in tqdm(loader):
#         data_s = data['data']['static'].numpy()
#         data_t = data['data']['time_series'].numpy()
#         label = data['label'].numpy()
#
#         for li, l in enumerate(label):
#             if l == 1:
#                 for fi, fn in enumerate(s_names):
#                     data_pos[fn] += list(data_s[li, :, fi][~np.isnan(data_s[li, :, fi])])
#                 for fi, fn in enumerate(ts_names):
#                     data_pos[fn] += list(data_t[li, :, fi][~np.isnan(data_t[li, :, fi])])
#             elif l == 0:
#                 for fi, fn in enumerate(s_names):
#                     data_neg[fn] += list(data_s[li, :, fi][~np.isnan(data_s[li, :, fi])])
#                 for fi, fn in enumerate(ts_names):
#                     data_neg[fn] += list(data_t[li, :, fi][~np.isnan(data_t[li, :, fi])])
#
#     np.savez(os.path.join(root_folder, f'ztest-{pred}.npz'),
#              data_pos=data_pos, data_neg=data_neg,
#              allow_pickle=True)
#
#     return 1
#
# for pred in [12, 6, 0]:
#
#     root_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src')
#     data_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src/ts_aligned')
#
#     data_split = np.load(os.path.join(root_folder, 'data_split.npy'), allow_pickle=True).item()
#     X_train = data_split['X_train'] + data_split['X_val']
#     y_train = data_split['y_train'] + data_split['y_val']
#     X_test = data_split['X_test']
#     y_test = data_split['y_test']
#
#     feature_static = pd.read_csv(
#         os.path.join(data_folder, 'pos', 'info', os.listdir(os.path.join(data_folder, 'pos', 'info'))[0]),
#         sep=';', header=0, index_col=[0]).columns.to_list()
#     feature_ts = pd.read_csv(
#         os.path.join(data_folder, 'pos', 'data', os.listdir(os.path.join(data_folder, 'pos', 'data'))[0]),
#         sep=';', header=0, index_col=[0, 1]).columns.to_list()
#
#     generate_stacked_dataset(data_folder, X_train, y_train, feature_static, feature_ts)
#
# ############################################################################################################


if __name__=='__main__':

    for pred in [12, 6, 0]:

        root_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src')
        data_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src/ts_aligned')

        feature_static = pd.read_csv(
            os.path.join(data_folder, 'pos', 'info', os.listdir(os.path.join(data_folder, 'pos', 'info'))[0]),
            sep=';', header=0, index_col=[0]).columns.to_list()
        feature_ts = pd.read_csv(
            os.path.join(data_folder, 'pos', 'data', os.listdir(os.path.join(data_folder, 'pos', 'data'))[0]),
            sep=';', header=0, index_col=[0, 1]).columns.to_list()

        data = np.load(os.path.join(root_folder, f'ztest-{pred}.npz'), allow_pickle=True)
        data_pos = data['data_pos'].item()
        data_neg = data['data_neg'].item()

        ztest_result = {}

        for fn in tqdm(feature_static):
            z_res = CompareMeans(ws.DescrStatsW(data_pos[fn]), ws.DescrStatsW(data_neg[fn])).ztest_ind(alternative='two-sided', usevar='unequal', value=0)
            ztest_result[fn] = z_res

        for fn in tqdm(feature_ts):
            z_res = CompareMeans(ws.DescrStatsW(data_pos[fn]), ws.DescrStatsW(data_neg[fn])).ztest_ind(alternative='two-sided', usevar='unequal', value=0)
            ztest_result[fn] = z_res

        print(ztest_result)

        np.savez(os.path.join(root_folder, f'ztest-result-{pred}.npz'),  ztest_result, allow_pickle=True)
