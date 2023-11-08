import os
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.utils.data import DataLoader

from src.classPatientDataset import PatientDatasetSimple



def load_input_uid_dict():
    print(os.path.abspath("."))
    input_table = pd.read_csv(
        'processed/DatasetOverview-Inputs-icd9code.csv',
        header=0, sep=',')
    input_uid_dict = {uid: input_table.loc[input_table['TableUID'] == uid, 'ParamNameNew'].iloc[0]
                      for uid in input_table.loc[input_table['TableUID'] < 7e5, 'TableUID'].unique()}
    for uid in input_table.loc[input_table['TableUID'] > 7e5, 'TableUID'].unique():
        input_uid_dict[uid] = input_table.loc[input_table['TableUID'] == uid, 'ParamNameOrigin'].iloc[0]
    return input_uid_dict


def load_patient_data_dsv(file_path, patient_id):
    data_file = os.path.join(file_path, 'data_' + str(patient_id) + '.dsv')
    info_file = os.path.join(file_path, 'info_' + str(patient_id) + '.dsv')
    patient_data = pd.read_csv(data_file, header=0, sep='$')
    patient_info = pd.read_csv(info_file, header=0, sep='$')
    return patient_info, patient_data


def get_pid_5849_all(file_path, pid_all, dx_list):
    pid_5849_all = []
    for pid in tqdm(pid_all):
        data_file = os.path.join(file_path, 'data_'+str(pid)+'.dsv')
        data = pd.read_csv(data_file, header=0, sep='$')
        data_dx = data[data['UID']>7e5]
        data_dx['Value'] = data_dx['Value'].astype(float)
        for dx in dx_list:
            if data_dx[(data_dx['UID']==dx)].shape[0] > 0:
                pid_5849_all.append(pid)
                break
    return pid_5849_all


def generate_neg_data_sample(info, data, ts, input_n_hour, selected_input_uid):
    data_i = []
    for uid in selected_input_uid:
        # gender
        if uid == 3:
            if info.loc[info['UID'] == uid, 'Value'].iloc[0] == 'Male':
                data_i.append(1)
            elif info.loc[info['UID'] == uid, 'Value'].iloc[0] == 'Female':
                data_i.append(2)
            else:
                data_i.append(np.nan)
        # age
        if uid == 4:
            val = info.loc[info['UID'] == uid, 'Value'].iloc[0]
            if pd.isnull(val):
                data_i.append(np.nan)
                continue
            if val.replace(' ', '') == '>89':
                data_i.append(90)
            elif val.replace(' ', '').isdigit():
                data_i.append(int(val))
            else:
                data_i.append(np.nan)
        # admission weight
        if uid == 23:
            val = info.loc[info['UID'] == uid, 'Value'].iloc[0]
            try:
                data_i.append(float(val))
            except:
                data_i.append(np.nan)
        # admission height
        if uid == 9:
            val = info.loc[info['UID'] == uid, 'Value'].iloc[0]
            try:
                data_i.append(float(val))
            except:
                data_i.append(np.nan)
        # vital sign
        if uid > 1e5 and uid < 3e5:
            try:
                val = data.loc[(data['UID'] == uid) & (data['Offset'] > ts - input_n_hour * 60) & (data['Offset'] <= ts), 'Value'].astype(float)
            except:
                break
            if val.shape[0] == 0:
                data_i += [np.nan] * 3
            else:
                mean = val.mean()
                std = val.std()
                trend = val[-10:].mean() - val[:10].mean()
                data_i += [mean, std, trend]
        # urine
        if uid == 300017:
            try:
                val = data.loc[(data['UID'] == uid) & (data['Offset'] > ts - input_n_hour * 60) & (data['Offset'] <= ts), 'Value'].astype(float)
            except:
                break
            if val.shape[0] == 0:
                data_i.append(np.nan)
            else:
                data_i.append(val.mean())
        # lab
        if uid > 4e5 and uid < 5e5:
            try:
                val = data.loc[(data['UID'] == uid) & (data['Offset'] > ts - input_n_hour * 60) & (data['Offset'] <= ts), 'Value'].astype(float)
            except:
                break
            if val.shape[0] == 0:
                data_i.append(np.nan)
            else:
                data_i.append(val.iloc[-1])

    return data_i


def generate_neg_data_sample_raw(info, data, ts, input_n_hour, selected_input_uid, uid_dict):
    data_neg = {}
    info_neg = {}
    for uid in selected_input_uid:
        # gender
        if uid == 3:
            if info.loc[info['UID'] == uid, 'Value'].iloc[0] == 'Male':
                info_neg[uid_dict[uid].upper()] = 1
            elif info.loc[info['UID'] == uid, 'Value'].iloc[0] == 'Female':
                info_neg[uid_dict[uid].upper()] = 2
            else:
                info_neg[uid_dict[uid].upper()] = np.nan
        # age
        if uid == 4:
            val = info.loc[info['UID'] == uid, 'Value'].iloc[0]
            if pd.isnull(val):
                info_neg[uid_dict[uid].upper()] = np.nan
                continue
            if val.replace(' ', '') == '>89':
                info_neg[uid_dict[uid].upper()] = 90
            elif val.replace(' ', '').isdigit():
                info_neg[uid_dict[uid].upper()] = int(val)
            else:
                info_neg[uid_dict[uid].upper()] = np.nan
        # admission weight
        if uid == 23:
            val = info.loc[info['UID'] == uid, 'Value'].iloc[0]
            try:
                info_neg[uid_dict[uid].upper()] = float(val)
            except:
                info_neg[uid_dict[uid].upper()] = np.nan
        # admission height
        if uid == 9:
            val = info.loc[info['UID'] == uid, 'Value'].iloc[0]
            try:
                info_neg[uid_dict[uid].upper()] = float(val)
            except:
                info_neg[uid_dict[uid].upper()] = np.nan
        # vital sign
        if uid > 1e5 and uid < 3e5:
            try:
                val = data.loc[(data['UID']==uid) & (data['Offset'] > ts - input_n_hour * 60) & (data['Offset'] <= ts), 'Value'].astype(float)
                time = data.loc[(data['UID']==uid) & (data['Offset'] > ts - input_n_hour * 60) & (data['Offset'] <= ts), 'Offset'].astype(float)
            except:
                break

            if uid == 100004:
                feat_name = 'SPO2'
            else:
                feat_name = uid_dict[uid].upper()
            if val.shape[0] == 0:
                data_neg[feat_name] = np.array([])
                data_neg[feat_name+'_T'] = np.array([])
            else:
                data_neg[feat_name] = val.values.flatten()
                data_neg[feat_name+'_T'] = time.values.flatten()

        # urine
        if uid == 300017:
            try:
                val = data.loc[(data['UID']==uid) & (data['Offset'] > ts - input_n_hour * 60) & (data['Offset'] <= ts), 'Value'].astype(float)
                time = data.loc[(data['UID']==uid) & (data['Offset'] > ts - input_n_hour * 60) & (data['Offset'] <= ts), 'Offset'].astype(float)
            except:
                break
            if val.shape[0] == 0:
                data_neg[uid_dict[uid].upper()] = np.array([])
                data_neg[uid_dict[uid].upper()+'_T'] = np.array([])
            else:
                data_neg[uid_dict[uid].upper()] = val.values.flatten()
                data_neg[uid_dict[uid].upper()+'_T'] = time.values.flatten()
        # lab
        if uid > 4e5 and uid < 5e5:
            try:
                val = data.loc[(data['UID']==uid) & (data['Offset'] > ts - input_n_hour * 60) & (data['Offset'] <= ts), 'Value'].astype(float)
                time = data.loc[(data['UID']==uid) & (data['Offset'] > ts - input_n_hour * 60) & (data['Offset'] <= ts), 'Offset'].astype(float)
            except:
                break
            if val.shape[0] == 0:
                data_neg[uid_dict[uid].upper()] = np.array([])
                data_neg[uid_dict[uid].upper()+'_T'] = np.array([])
            else:
                data_neg[uid_dict[uid].upper()] = val.values.flatten()
                data_neg[uid_dict[uid].upper()+'_T'] = time.values.flatten()

    return info_neg, data_neg


def generate_pos_data_sample(info, data, dx_ts, pred_n_hour, input_n_hour, selected_input_uid):
    data_i = []
    for uid in selected_input_uid:
        #gender
        if uid == 3:
            if info.loc[info['UID']==uid, 'Value'].iloc[0] == 'Male':
                data_i.append(1)
            elif info.loc[info['UID']==uid, 'Value'].iloc[0] == 'Female':
                data_i.append(2)
            else:
                data_i.append(np.nan)
        #age
        if uid == 4:
            val = info.loc[info['UID']==uid, 'Value'].iloc[0]
            if val.replace(' ', '') == '>89':
                data_i.append(90)
            elif val.replace(' ', '').isdigit():
                data_i.append(int(val))
            else:
                data_i.append(np.nan)
        #admission weight
        if uid == 23:
            val = info.loc[info['UID']==uid, 'Value'].iloc[0]
            try:
                data_i.append(float(val))
            except:
                data_i.append(np.nan)
        #admission height
        if uid == 9:
            val = info.loc[info['UID']==uid, 'Value'].iloc[0]
            try:
                data_i.append(float(val))
            except:
                data_i.append(np.nan)
        #vital sign
        # manually calculated feature
        if uid > 1e5 and uid < 3e5:
            try:
                val = data.loc[(data['UID']==uid) & (data['Offset']>dx_ts-(pred_n_hour+input_n_hour)*60) & (data['Offset']<=dx_ts-pred_n_hour*60), 'Value'].astype(float)
            except:
                break
            if val.shape[0] == 0:
                data_i += [np.nan] * 3
            else:
                mean = val.mean()
                std = val.std()
                trend = val[-10:].mean() - val[:10].mean()
                data_i += [mean, std, trend]
        #urine
        if uid == 300017:
            try:
                val = data.loc[(data['UID']==uid) & (data['Offset']>dx_ts-(pred_n_hour+input_n_hour)*60) & (data['Offset']<=dx_ts-pred_n_hour*60), 'Value'].astype(float)
            except:
                break
            if val.shape[0] == 0:
                data_i.append(np.nan)
            else:
                data_i.append(val.mean())
        #lab
        if uid > 4e5 and uid < 5e5:
            try:
                val = data.loc[(data['UID']==uid) & (data['Offset']>dx_ts-(pred_n_hour+input_n_hour)*60) & (data['Offset']<=dx_ts-pred_n_hour*60), 'Value'].astype(float)
            except:
                break
            if val.shape[0] == 0:
                data_i.append(np.nan)
            else:
                data_i.append(val.iloc[-1])

    return data_i


def generate_pos_data_sample_raw(info, data, dx_ts, pred_n_hour, input_n_hour, selected_input_uid, uid_dict):
    info_pos = {}
    data_pos = {}
    for uid in selected_input_uid:
        #gender
        if uid == 3:
            if info.loc[info['UID']==uid, 'Value'].iloc[0] == 'Male':
                info_pos[uid_dict[uid].upper()] = 1
            elif info.loc[info['UID']==uid, 'Value'].iloc[0] == 'Female':
                info_pos[uid_dict[uid].upper()] = 2
            else:
                info_pos[uid_dict[uid].upper()] = np.nan
        #age
        if uid == 4:
            val = info.loc[info['UID']==uid, 'Value'].iloc[0]
            if pd.isnull(val):
                info_pos[uid_dict[uid].upper()] = np.nan
                continue
            if val.replace(' ', '') == '>89':
                info_pos[uid_dict[uid].upper()] = 90
            elif val.replace(' ', '').isdigit():
                info_pos[uid_dict[uid].upper()] = int(val)
            else:
                info_pos[uid_dict[uid].upper()] = np.nan
        #admission weight
        if uid == 23:
            val = info.loc[info['UID']==uid, 'Value'].iloc[0]
            try:
                info_pos[uid_dict[uid].upper()] = float(val)
            except:
                info_pos[uid_dict[uid].upper()] = np.nan
        #admission height
        if uid == 9:
            val = info.loc[info['UID']==uid, 'Value'].iloc[0]
            try:
                info_pos[uid_dict[uid].upper()] = float(val)
            except:
                info_pos[uid_dict[uid].upper()] = np.nan
        #vital sign
        # raw data
        if uid > 1e5 and uid < 3e5:
            try:
                val = data.loc[(data['UID']==uid) & (data['Offset']>dx_ts-(pred_n_hour+input_n_hour)*60) & (data['Offset']<dx_ts-pred_n_hour*60), 'Value'].astype(float)
                ts = data.loc[(data['UID']==uid) & (data['Offset']>dx_ts-(pred_n_hour+input_n_hour)*60) & (data['Offset']<dx_ts-pred_n_hour*60), 'Offset'].astype(float)
            except:
                break

            if uid == 100004:
                feat_name = 'SPO2'
            else:
                feat_name = uid_dict[uid].upper()
            if val.shape[0] == 0:
                data_pos[feat_name] = np.array([])
                data_pos[feat_name+'_T'] = np.array([])
            else:
                data_pos[feat_name] = val.values.flatten()
                data_pos[feat_name+'_T'] = ts.values.flatten()

        #urine
        if uid == 300017:
            try:
                val = data.loc[(data['UID']==uid) & (data['Offset']>dx_ts-(pred_n_hour+input_n_hour)*60) & (data['Offset']<dx_ts-pred_n_hour*60), 'Value'].astype(float)
                ts = data.loc[(data['UID']==uid) & (data['Offset']>dx_ts-(pred_n_hour+input_n_hour)*60) & (data['Offset']<dx_ts-pred_n_hour*60), 'Offset'].astype(float)
            except:
                break
            if val.shape[0] == 0:
                data_pos[uid_dict[uid].upper()] = np.array([])
                data_pos[uid_dict[uid].upper()+'_T'] = np.array([])
            else:
                data_pos[uid_dict[uid].upper()] = val.values.flatten()
                data_pos[uid_dict[uid].upper()+'_T'] = ts.values.flatten()
        #lab
        if uid > 4e5 and uid < 5e5:
            try:
                val = data.loc[(data['UID']==uid) & (data['Offset']>dx_ts-(pred_n_hour+input_n_hour)*60) & (data['Offset']<dx_ts-pred_n_hour*60), 'Value'].astype(float)
                ts = data.loc[(data['UID']==uid) & (data['Offset']>dx_ts-(pred_n_hour+input_n_hour)*60) & (data['Offset']<dx_ts-pred_n_hour*60), 'Offset'].astype(float)
            except:
                break
            if val.shape[0] == 0:
                data_pos[uid_dict[uid].upper()] = np.array([])
                data_pos[uid_dict[uid].upper()+'_T'] = np.array([])
            else:
                data_pos[uid_dict[uid].upper()] = val.values.flatten()
                data_pos[uid_dict[uid].upper()+'_T'] = ts.values.flatten()

    return info_pos, data_pos


def generate_dataset_processed(dsv_path, pid_list, dx_list, selected_input_uid, pred_n_hour=3, input_n_hour=12, neg_data=False, save_data=False):
    np.random.seed(666)

    if save_data:
        save_path = os.path.abspath('processed/patient_data/dx_pred_' + str(pred_n_hour) + '_' + str(input_n_hour))
        Path(save_path).mkdir(parents=True, exist_ok=True)

    pid_5849_all = pid_list[0]
    pid_non_5849 = pid_list[1]

    #positive
    data_pos = []
    for pid in tqdm(pid_5849_all):
        info, data = load_patient_data_dsv(dsv_path, pid)
        dx_events = data[data['UID'].isin(dx_list)]
        #Ensure that there is enough data before the diagnosis
        dx_events_ts = dx_events.loc[(dx_events['Offset'] > (pred_n_hour+input_n_hour)*60) & (dx_events['Offset'].diff() > (pred_n_hour+input_n_hour)*60), 'Offset']

        for di, dx_ts in enumerate(dx_events_ts):
            data_i = generate_pos_data_sample(info, data, dx_ts, pred_n_hour, input_n_hour, selected_input_uid)
            data_pos.append(data_i)

    data_pos = np.array(data_pos)
    if save_data:
        np.save(os.path.abspath(os.path.join(save_path, 'data_positive_processed')), data_pos)

    if neg_data:
        #negative (patient without occurance of kidney failure during the ICU stay)
        data_neg = []
        for pid in tqdm(pid_non_5849):
            info, data = load_patient_data_dsv(dsv_path, pid)
            #select time period with valid recording of vital signs
            ts_range = data.loc[data['UID'] == 100002, 'Offset'].values
            if len(ts_range) == 0:
                continue
            if ts_range[-1] - ts_range[0] < 2 * input_n_hour * 60:
                continue

            ts = ts_range[0] + input_n_hour * 60
            data_i = generate_neg_data_sample(info, data, ts, input_n_hour, selected_input_uid)
            data_neg.append(data_i)

        data_neg = np.array(data_neg)
        if save_data:
            np.save(os.path.abspath(os.path.join(save_path, 'data_negative_processed')), data_neg)
        return data_pos, data_neg

    return data_pos, None


def generate_dataset_raw(dsv_path, pid_list, dx_list, selected_input_uid, feature_names, uid_dict, pred_n_hour=3, input_n_hour=12, neg_data=False, save_data=True):
    feat_static = feature_names[:4]
    feat_ts = []
    for feat in feature_names[4:]:
        feat_ts += [feat, feat+'_T']

    if save_data:
        save_path = os.path.abspath('processed/patient_data/dx_pred_' + str(pred_n_hour) + '_' + str(input_n_hour) + '/raw/')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path, 'pos')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path, 'neg')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path, 'pos', 'info')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path, 'pos', 'data')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path, 'neg', 'info')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path, 'neg', 'data')).mkdir(parents=True, exist_ok=True)

    pid_5849_all = pid_list[0]
    pid_non_5849 = pid_list[1]

    #positive
    for pid in tqdm(pid_5849_all):
        info, data = load_patient_data_dsv(dsv_path, pid)
        dx_events = data[data['UID'].isin(dx_list)]
        #Ensure that there is enough data before the diagnosis
        dx_events_ts = dx_events.loc[(dx_events['Offset'] > (pred_n_hour+input_n_hour)*60) & (dx_events['Offset'].diff() > (pred_n_hour+input_n_hour)*60), 'Offset']

        for di, dx_ts in enumerate(dx_events_ts):
            info_pos, data_pos = generate_pos_data_sample_raw(info, data, dx_ts, pred_n_hour, input_n_hour, selected_input_uid, uid_dict)

            df_pos = pd.DataFrame.from_dict(data_pos, orient='index').transpose()
            df_pos = df_pos[feat_ts]
            df_info_pos = pd.DataFrame.from_dict(info_pos, orient='index').transpose()
            df_info_pos = df_info_pos[feat_static]
            if save_data:
                df_pos.to_csv(os.path.abspath(os.path.join(save_path, 'pos', 'data', f'{pid}_{di}.csv')), sep=';')
                df_info_pos.to_csv(os.path.abspath(os.path.join(save_path, 'pos', 'info', f'{pid}_{di}.csv')), sep=';')


    if neg_data:
        #negative (patient without occurance of kidney failure during the ICU stay)
        for pid in tqdm(pid_non_5849):
            info, data = load_patient_data_dsv(dsv_path, pid)
            #select time period with valid recording of vital signs
            ts_range = data.loc[data['UID'] == 100002, 'Offset'].values
            if len(ts_range) == 0:
                continue
            if ts_range[-1] - ts_range[0] < 2 * input_n_hour * 60:
                continue

            ts = ts_range[0] + input_n_hour * 60
            info_neg, data_neg = generate_neg_data_sample_raw(info, data, ts, input_n_hour, selected_input_uid, uid_dict)

            df_neg = pd.DataFrame.from_dict(data_neg, orient='index').transpose()
            df_neg = df_neg[feat_ts]
            df_info_neg = pd.DataFrame.from_dict(info_neg, orient='index').transpose()
            df_info_neg = df_info_neg[feat_static]
            if save_data:
                df_neg.to_csv(os.path.abspath(os.path.join(save_path, 'neg', 'data', f'{pid}_0.csv')), sep=';')
                df_info_neg.to_csv(os.path.abspath(os.path.join(save_path, 'neg', 'info', f'{pid}_0.csv')), sep=';')


        return 1

    return 1


def align_timestamps_raw(file_folder, feature_names, input_n_hour=12):
    pos_path = os.path.join(file_folder, 'pos/data')
    neg_path = os.path.join(file_folder, 'neg/data')

    save_pos_path = os.path.join(file_folder, '../ts_aligned/pos/data')
    save_neg_path = os.path.join(file_folder, '../ts_aligned/neg/data')
    Path(save_pos_path).mkdir(parents=True, exist_ok=True)
    Path(save_neg_path).mkdir(parents=True, exist_ok=True)

    sample_list = {
        'pos': os.listdir(pos_path),
        'neg': os.listdir(neg_path),
    }
    for k in sample_list:
        sample_list[k] = [os.path.splitext(sample)[0] for sample in sample_list[k]]
        sample_list[k] = sorted(list(set(sample_list[k])))

    for path, samples, save_path in zip([pos_path, neg_path], [sample_list['pos'], sample_list['neg']], [save_pos_path, save_neg_path]):
        for sample in tqdm(samples):
            data_origin = pd.read_csv(os.path.join(path, sample+'.csv'), sep=';', header=0, index_col=0)
            ts_start = data_origin.filter(regex='_T').min().min()
            ts_end = data_origin.filter(regex='_T').max().max()
            if pd.isnull(ts_start):
                continue
            ts_start = ts_start - ts_start % 5
            if ts_end - ts_start >= 720:
                ts_start += 5

            # columns = ['Timestamp'] + list(set([col.replace('_T', '') for col in data_origin.columns]))
            columns = ['Timestamp'] + feature_names
            data_aligned = pd.DataFrame(columns=columns)

            data_aligned['Timestamp'] = np.arange(ts_start, ts_start + 5 * input_n_hour*60/5, 5)

            for col in data_aligned.columns.to_list():
                if col == 'Timestamp':
                    continue
                for ind, d in data_origin[[col, col+'_T']].dropna().iterrows():
                    if int((d[col+'_T']-ts_start)//5) > 143:
                        print(data_origin[col+'_T'])
                    if d[col+'_T'] < ts_start:
                        continue
                    data_aligned.at[int((d[col+'_T']-ts_start)//5), col] = d[col]

            data_aligned.to_csv(os.path.join(save_path, sample+'.csv'), sep=';')

    return 1


def train_test_split_aligned_data(file_folder):
    pos_path = os.path.join(file_folder, 'pos/data')
    neg_path = os.path.join(file_folder, 'neg/data')

    sample_list = {
        'pos': os.listdir(pos_path),
        'neg': os.listdir(neg_path),
    }
    for k in sample_list:
        # sample_list[k] = [os.path.splitext(sample)[0].replace('_info', '').replace('_data', '') for sample in sample_list[k]]
        sample_list[k] = [os.path.splitext(sample)[0] for sample in sample_list[k]]
        sample_list[k] = sorted(list(set(sample_list[k])))
    sample_label = [1] * len(sample_list['pos']) + [0] * len(sample_list['neg'])
    sample_list = sample_list['pos'] + sample_list['neg']

    X_train, X_test, y_train, y_test = train_test_split(sample_list, sample_label, train_size=.8, random_state=0, shuffle=True, stratify=sample_label)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=.5, random_state=0, shuffle=True, stratify=y_test)

    data_split = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
    np.save(os.path.join(file_folder, '../data_split'), data_split, allow_pickle=True)

    return data_split


def data_zero_masking(data):
    data[np.isnan(data)] = 0
    return data


def calculate_normalization_params(data_split, data_folder):
    X_train = data_split['X_train']
    y_train = data_split['y_train']
    X_val = data_split['X_val']
    y_val = data_split['y_val']

    X_all = X_train + X_val
    y_all = y_train + y_val

    ds = PatientDatasetSimple(data_folder, X_all, y_all)
    ds_loader = DataLoader(ds, 1000, num_workers=6)

    dim_t = 34  # input dimension of temporal data
    dim_s = 4  # input dimension of static data

    mean = torch.zeros(dim_t)
    std = torch.zeros(dim_t)
    count = torch.zeros(dim_t)

    mean_s = torch.zeros(dim_s)
    std_s = torch.zeros(dim_s)
    count_s = torch.zeros(dim_s)

    for batch in tqdm(ds_loader):
        info = batch['data']['static']
        info = info.view(-1, dim_s)
        mask = info.isnan()
        info = torch.masked_fill(info, mask, 0)
        count_s += (~mask).sum(dim=0)
        mean_s += info.sum(dim=0)

        data = batch['data']['time_series']
        data = data.view(-1, dim_t)
        mask = data.isnan()
        data = torch.masked_fill(data, mask, 0)
        count += (~mask).sum(dim=0)
        mean += data.sum(dim=0)
    mean_s = mean_s / count_s
    mean = mean / count

    for batch in tqdm(ds_loader):
        info = batch['data']['static']
        info = info.view(-1, dim_s)
        mask = info.isnan()
        info = torch.masked_fill(info, mask, 0)
        std_s += (info - mean_s).pow(2).sum(dim=0)

        data = batch['data']['time_series']
        data = data.view(-1, dim_t)
        mask = data.isnan()
        data = torch.masked_fill(data, mask, 0)
        std += (data - mean).pow(2).sum(dim=0)
    std_s = torch.sqrt(std_s / count_s)
    std = torch.sqrt(std / count)

    dataset_norm_params = {
        'static_mean': mean_s,
        'static_std': std_s,
        'data_mean': mean,
        'data_std': std,
    }

    return dataset_norm_params