import os
from pathlib import Path
os.chdir(os.path.join(Path(__file__).parent.resolve(), ".."))

import sys
sys.path.append(os.path.dirname(os.path.realpath(Path(__file__).parent.resolve())))
# print(os.path.abspath("."))

from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold

import tensorflow as tf

from utils.data_io import *
from utils.common import *

uid_dict = load_input_uid_dict()
uid_tables = {
    'Vital Signs': [1e5, 3e5],
    'Intake Output': [3e5, 4e5],
    'Lab Tests': [4e5, 5e5],
}

dx_5849 = [700003, 700163, 700499, 700571, 700596, 700854]

selected_input_uid = [
    #Demographics
    3, 4, 23, 9,
    #VitalPeriodic
    100002, 100004, 100003, 100001,
    #VitalAperiodic
    200003, 200004, 200005,
    #Urine
    300017,
    #Lab
    400008, 400012, 400011, 400019, 400031, 400006, 400004, 400002, 400010,
    400027, 400028, 400033, 400014, 400015, 400013, 400029, 400017, 400018,
    400026, 400025, 400024, 400001, 400030, 400003, 400016, 400005
]

columns = ['Gender', 'Age', 'AdmissionWeight', 'AdimissionHeight',
           'HR_mean', 'HR_std', 'HR_trend',
           'SpO2_mean', 'SpO2_std', 'SpO2_trend',
           'Respiration_mean', 'Respiration_std', 'Respiration_trend',
           'CVP_mean', 'CVP_std', 'CVP_trend',
           'NIBP_d_mean', 'NIBP_d_std', 'NIBP_d_trend',
           'NIBP_m_mean', 'NIBP_m_std', 'NIBP_m_trend',
           'NIBP_s_mean', 'NIBP_s_std', 'NIBP_s_trend',
           'Urine',
           'GLC', 'K', 'HCO3', 'Na', 'Urea', 'Cr', 'Cl', 'Ca', 'HBG',
           'PLT', 'RBC', 'WBC', 'MCHC', 'MCV', 'MCH', 'RDW', 'Mg', 'MPV',
           'PH', 'PAO2', 'PACO2', 'Baseexcess', 'SAO2', 'Caionized', 'METHB', 'COHB',
]

pid_all = np.load(os.path.abspath('processed/pid_all.npy'))
pid_5849_all = np.load('processed/pid_5849_all.npy')
pid_non_5849 = np.load('processed/pid_non_5849.npy')


def get_pid_5849_all(file_path='/home/dhm/workspace/eicu/data/all_unit_type_structured_dsv_data_v1/'):
    pid_5849_all = []
    for pid in tqdm(pid_all):
        data_file = os.path.join(file_path, 'data_'+str(pid)+'.dsv')
        data = pd.read_csv(data_file, header=0, sep='$')
        data_dx = data[data['UID']>7e5]
        data_dx['Value'] = data_dx['Value'].astype(float)
        for dx in dx_5849:
            if data_dx[(data_dx['UID']==dx)].shape[0] > 0:
                pid_5849_all.append(pid)
                break
    return pid_5849_all


def generate_neg_data_sample(info, data, ts, input_n_hour):
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


def generate_neg_data_sample_raw(info, data, ts, input_n_hour):
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
            if val.shape[0] == 0:
                data_neg[uid_dict[uid].upper()] = np.array([])
                data_neg[ uid_dict[uid].upper()+'_T'] = np.array([])
            else:
                data_neg[uid_dict[uid].upper()] = val.values.flatten()
                data_neg[uid_dict[uid].upper()+'_T'] = time.values.flatten()

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


def generate_pos_data_sample(info, data, dx_ts, pred_n_hour, input_n_hour):
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


def generate_pos_data_sample_raw(info, data, dx_ts, pred_n_hour, input_n_hour):
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
            if val.shape[0] == 0:
                data_pos[uid_dict[uid].upper()] = np.array([])
                data_pos[uid_dict[uid].upper()+'_T'] = np.array([])
            else:
                data_pos[uid_dict[uid].upper()] = val.values.flatten()
                data_pos[uid_dict[uid].upper()+'_T'] = ts.values.flatten()

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


def generate_data_dx5849(pred_n_hour=3, input_n_hour=12, neg_data=False, save_data=False):
    np.random.seed(666)

    if save_data:
        save_path = os.path.abspath('processed/patient_data/dx_pred_' + str(pred_n_hour) + '_' + str(input_n_hour) + '_3src')
        Path(save_path).mkdir(parents=True, exist_ok=True)

    #positive
    data_pos = []
    for pid in tqdm(pid_5849_all):
        info, data = load_patient_data_dsv('/home/dhm/workspace/eicu/data/all_unit_type_structured_dsv_data_v1/', pid)
        dx_events = data[data['UID'].isin(dx_5849)]
        #Ensure that there is enough data before the diagnosis
        dx_events_ts = dx_events.loc[(dx_events['Offset'] > (pred_n_hour+input_n_hour)*60) & (dx_events['Offset'].diff() > (pred_n_hour+input_n_hour)*60), 'Offset']

        for di, dx_ts in enumerate(dx_events_ts):
            data_i = generate_pos_data_sample(info, data, dx_ts, pred_n_hour, input_n_hour)
            data_pos.append(data_i)

    data_pos = np.array(data_pos)
    df_pos = pd.DataFrame(data_pos, columns=columns, dtype=float)
    if save_data:
        np.save(os.path.abspath(os.path.join(save_path, 'data_positive')), data_pos)
        df_pos.to_csv(os.path.abspath(os.path.join(save_path, 'data_positive.csv')), sep=';')

    if neg_data:
        #negative (patient without occurance of kidney failure during the ICU stay)
        data_neg = []
        for pid in tqdm(pid_non_5849):
            info, data = load_patient_data_dsv('/home/dhm/workspace/eicu/data/all_unit_type_structured_dsv_data_v1/', pid)
            #select time period with valid recording of vital signs
            ts_range = data.loc[data['UID'] == 100002, 'Offset'].values
            if len(ts_range) == 0:
                continue
            if ts_range[-1] - ts_range[0] < 2 * input_n_hour * 60:
                continue

            ts = ts_range[0] + input_n_hour * 60
            data_i = generate_neg_data_sample(info, data, ts, input_n_hour)
            data_neg.append(data_i)

        data_neg = np.array(data_neg)
        df_neg = pd.DataFrame(data_neg, columns=columns, dtype=float)
        if save_data:
            np.save(os.path.abspath(os.path.join(save_path, 'data_negative')), data_neg)
            df_neg.to_csv(os.path.abspath(os.path.join(save_path, 'data_negative.csv')), sep=';')

        return data_pos, data_neg

    return data_pos


def generate_data_dx5849_raw(dsv_path, pred_n_hour=3, input_n_hour=12, neg_data=False, save_data=True):
    if save_data:
        save_path = os.path.abspath('processed/patient_data/dx_pred_' + str(pred_n_hour) + '_' + str(input_n_hour) + '_3src/raw/')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path, 'pos')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path, 'neg')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path, 'pos', 'info')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path, 'pos', 'data')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path, 'neg', 'info')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path, 'neg', 'data')).mkdir(parents=True, exist_ok=True)

    #positive
    for pid in tqdm(pid_5849_all):
        info, data = load_patient_data_dsv(dsv_path, pid)
        dx_events = data[data['UID'].isin(dx_5849)]
        #Ensure that there is enough data before the diagnosis
        dx_events_ts = dx_events.loc[(dx_events['Offset'] > (pred_n_hour+input_n_hour)*60) & (dx_events['Offset'].diff() > (pred_n_hour+input_n_hour)*60), 'Offset']

        for di, dx_ts in enumerate(dx_events_ts):
            info_pos, data_pos = generate_pos_data_sample_raw(info, data, dx_ts, pred_n_hour, input_n_hour)

            df_pos = pd.DataFrame.from_dict(data_pos, orient='index').transpose()
            df_info_pos = pd.DataFrame.from_dict(info_pos, orient='index').transpose()
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
            info_neg, data_neg = generate_neg_data_sample_raw(info, data, ts, input_n_hour)

            df_neg = pd.DataFrame.from_dict(data_neg, orient='index').transpose()
            df_info_neg = pd.DataFrame.from_dict(info_neg, orient='index').transpose()
            if save_data:
                df_neg.to_csv(os.path.abspath(os.path.join(save_path, 'neg', 'data', f'{pid}_0.csv')), sep=';')
                df_info_neg.to_csv(os.path.abspath(os.path.join(save_path, 'neg', 'info', f'{pid}_0.csv')), sep=';')


        return 1

    return 1


def align_timestamps_raw(input_n_hour=12, file_folder=os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_0_12_3src/raw')):
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
        # sample_list[k] = [os.path.splitext(sample)[0].replace('_info', '').replace('_data', '') for sample in sample_list[k]]
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

            columns = ['Timestamp'] + list(set([col.replace('_T', '') for col in data_origin.columns]))
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

                check1 = data_aligned[['Timestamp', col]].dropna()
                check2 = data_origin[[col+'_T', col]].dropna()

            data_aligned.to_csv(os.path.join(save_path, sample+'.csv'), sep=';')

    return 1


def resample_aligned_data(sample_data, resample_freq='10Min'):
    sample_data['Timestamp'] = pd.to_datetime(sample_data['Timestamp'], unit='m')
    sample_data = sample_data.set_index('Timestamp')
    resampled_data = sample_data.resample(resample_freq).mean()

    return resampled_data


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

    return 1


def generate_data_batch_raw(sample_list, sample_label,
                            file_folder=os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_0_12_3src/ts_aligned'),
                            batchsize=128,
                            resample_freq=None):
    pos_path = os.path.join(file_folder, 'pos/data')
    neg_path = os.path.join(file_folder, 'neg/data')
    info_pos_path = os.path.join(file_folder, '../raw/pos/info')
    info_neg_path = os.path.join(file_folder, '../raw/neg/info')

    # sample_list = {
    #     'pos': os.listdir(pos_path),
    #     'neg': os.listdir(neg_path),
    # }
    # for k in sample_list:
    #     sample_list[k] = [os.path.splitext(sample)[0].replace('_info', '').replace('_data', '') for sample in sample_list[k]]
    #     sample_list[k] = sorted(list(set(sample_list[k])))
    # sample_label = [1] * len(sample_list['pos']) + [0] * len(sample_list['neg'])
    # sample_list = sample_list['pos'] + sample_list['neg']
    # sample_list, sample_label = shuffle(sample_list, sample_label, random_state=0)

    # sample_list, _, sample_label, _ = np.load(data_split, allow_pickle=True)

    sample_count = 0
    N_sample = len(sample_label)

    while sample_count < N_sample:
        if sample_count + batchsize < N_sample:
            N_batch = batchsize
        else:
            N_batch = N_sample-sample_count

        info_batch = []
        data_batch = []
        label_batch = []

        for i in range(N_batch):
            sample = sample_list[sample_count]
            label = sample_label[sample_count]
            if label == 1:
                info_path = info_pos_path
                path = pos_path
            else:
                info_path = info_neg_path
                path = neg_path

            info = pd.read_csv(os.path.join(info_path, sample+'.csv'), sep=';', header=0, index_col=0)
            data = pd.read_csv(os.path.join(path, sample+'.csv'), sep=';', header=0, index_col=0)

            if resample_freq:
                data = resample_aligned_data(data, resample_freq)

            info = info.to_numpy().flatten()
            data = data.to_numpy()
            info_batch.append(info)
            data_batch.append(data)
            label_batch.append(label)

            sample_count += 1

        yield np.array(info_batch), np.array(data_batch), np.array(label_batch).reshape(-1,1)


def generate_balanced_data_batch_raw(sample_list, sample_label,
                                     file_folder=os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_0_12_3src/ts_aligned'),
                                     batchsize=128,
                                     resample_freq=None):
    if batchsize % 2 != 0:
        raise ValueError('Batch size should be even number.')
    bs_class = int(batchsize / 2)

    pos_path = os.path.join(file_folder, 'pos/data')
    neg_path = os.path.join(file_folder, 'neg/data')
    info_pos_path = os.path.join(file_folder, '../raw/pos/info')
    info_neg_path = os.path.join(file_folder, '../raw/neg/info')

    idx_pos = np.where(np.array(sample_label) == 1)[0]
    idx_neg = np.where(np.array(sample_label) == 0)[0]
    sample_count = 0
    N_sample = len(sample_label)
    N_sample_pos = sum(sample_label)
    N_sample_neg = N_sample - N_sample_pos

    while sample_count < N_sample_neg:
        if sample_count + bs_class < N_sample_neg:
            N_batch = bs_class
        else:
            N_batch = N_sample_neg - sample_count

        info_batch = []
        data_batch = []
        label_batch = []

        for i in range(N_batch):

            for iii, (idx, info_path, path) in enumerate(zip([idx_pos, idx_neg], [info_pos_path, info_neg_path], [pos_path, neg_path])):
                if iii == 0:
                    sample = sample_list[idx[sample_count % N_sample_pos]]
                    label = sample_label[idx[sample_count % N_sample_pos]]
                else:
                    sample = sample_list[idx[sample_count]]
                    label = sample_label[idx[sample_count]]
                info = pd.read_csv(os.path.join(info_path, sample + '.csv'), sep=';', header=0, index_col=0)
                data = pd.read_csv(os.path.join(path, sample + '.csv'), sep=';', header=0, index_col=0)

                if resample_freq:
                    data = resample_aligned_data(data, resample_freq)

                info = info.to_numpy().flatten()
                data = data.to_numpy()
                info_batch.append(info)
                data_batch.append(data)
                label_batch.append(label)

            sample_count += 1

        yield np.array(info_batch), np.array(data_batch), np.array(label_batch).reshape(-1, 1)


def data_zero_masking(data):
    data[np.isnan(data)] = 0
    return data


def generate_tf_dataset_raw(sample_list, sample_label,
                            file_folder=os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_0_12_3src/ts_aligned'),
                            resample_freq=None):
    path = {
        'pos': os.path.join(file_folder, 'pos/data'),
        'neg': os.path.join(file_folder, 'neg/data'),
    }
    info_path = {
        'pos': os.path.join(file_folder, 'pos/info'),
        'neg': os.path.join(file_folder, 'neg/info'),
    }

    N_sample = {}
    N_sample['total'] = len(sample_label)
    N_sample['pos'] = sum(sample_label)
    N_sample['neg'] = N_sample['total'] - N_sample['pos']

    idx = {
        'pos': np.where(np.array(sample_label) == 1)[0],
        'neg': np.where(np.array(sample_label) == 0)[0]
    }

    info_set = {'pos': [], 'neg': []}
    data_set = {'pos': [], 'neg': []}
    label_set = {'pos': [], 'neg': []}

    tf_dataset = {
        'pos': {},
        'neg': {},
    }

    for data_label in ['pos', 'neg']:
        for i in tqdm(range(N_sample[data_label])):
            sample = sample_list[idx[data_label][i]]
            label = sample_label[idx[data_label][i]]
            info = pd.read_csv(os.path.join(info_path[data_label], sample + '.csv'), sep=';', header=0, index_col=0)
            data = pd.read_csv(os.path.join(path[data_label], sample + '.csv'), sep=';', header=0, index_col=[0, 1])

            if resample_freq:
                data = resample_aligned_data(data, resample_freq)

            info = info.to_numpy().flatten()
            info = data_zero_masking(info)
            data = data.to_numpy()
            data = data_zero_masking(data)

            info_set[data_label].append(info)
            data_set[data_label].append(data)
            label_set[data_label].append(label)

        tf_dataset[data_label] = tf.data.Dataset.from_tensor_slices(({'static_inputs': tf.convert_to_tensor(info_set[data_label], dtype=tf.float64),
                                                                      'time_series_inputs': tf.convert_to_tensor(data_set[data_label], dtype=tf.float64)},
                                                                     tf.convert_to_tensor(label_set[data_label], dtype=tf.int32)))

    return tf_dataset


def generate_equally_resampled_tf_dataset(dataset_pos, dataset_neg):
    BUFFER_SIZE = max(dataset_pos.cardinality().numpy(), dataset_neg.cardinality().numpy())
    dataset_pos = dataset_pos.shuffle(BUFFER_SIZE).repeat()
    dataset_neg = dataset_neg.shuffle(BUFFER_SIZE).repeat()
    resampled_ds = tf.data.Dataset.sample_from_datasets([dataset_pos, dataset_neg], weights=[0.5, 0.5])

    return resampled_ds


def generate_validation_data_raw(X, y, idx,
                                 file_folder=os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_0_12_3src/ts_aligned'),
                                 resample_freq=None):
    pos_path = os.path.join(file_folder, 'pos/data')
    neg_path = os.path.join(file_folder, 'neg/data')
    info_pos_path = os.path.join(file_folder, '../raw/pos/info')
    info_neg_path = os.path.join(file_folder, '../raw/neg/info')

    info_ = []
    data_ = []
    label_ = []

    for i in tqdm(idx):
        sample = X[i]
        label = y[i]
        if label == 1:
            info_path = info_pos_path
            path = pos_path
        else:
            info_path = info_neg_path
            path = neg_path

        info = pd.read_csv(os.path.join(info_path, sample+'.csv'), sep=';', header=0, index_col=0)
        data = pd.read_csv(os.path.join(path, sample+'.csv'), sep=';', header=0, index_col=0)

        if resample_freq:
            data = resample_aligned_data(data, resample_freq)

        info = info.to_numpy().flatten()
        data = data.to_numpy()
        info_.append(info)
        data_.append(data)
        label_.append(label)

    return np.array(info_), np.array(data_), np.array(label_)



if __name__ == "__main__":

# ######################################################################################################
# # Generate raw data and aligned data
# ######################################################################################################
#     DATA_INTERVAL = 12
#     PRED_INTERVAL = 12
#
#     file_folder_dsv = '/home/dhm/workspace/eicu/data/all_unit_type_structured_dsv_data_v1/'
#     file_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{PRED_INTERVAL}_{DATA_INTERVAL}_3src')
#     file_folder_raw = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{PRED_INTERVAL}_{DATA_INTERVAL}_3src/raw')
#     file_folder_aligned = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{PRED_INTERVAL}_{DATA_INTERVAL}_3src/ts_aligned')
#
#     # pid_all = os.listdir(file_folder_dsv)
#     # pid_all = [os.path.splitext(p)[0].replace('data_', '').replace('info_', '') for p in pid_all]
#     # pid_all = sorted(list(set(pid_all)))
#     # np.save('processed/pid_all', pid_all)
#     #
#     # pid_5849_all = get_pid_5849_all('/home/dhm/workspace/eicu/data/all_unit_type_structured_dsv_data_v1')
#     # np.save('processed/pid_5849_all', pid_5849_all)
#     #
#     # pid_non_5849 = [pid for pid in pid_all if pid not in pid_5849_all]
#     # np.save('processed/pid_non_5849', pid_non_5849)
#
#     pid_all = np.load(os.path.abspath('processed/pid_all.npy'))
#     pid_5849_all = np.load('processed/pid_5849_all.npy')
#     pid_non_5849 = np.load('processed/pid_non_5849.npy')
#
#     print('Generating raw data...')
#     generate_data_dx5849_raw(file_folder_dsv, pred_n_hour=PRED_INTERVAL, input_n_hour=DATA_INTERVAL, neg_data=True, save_data=True)
#
#     print('Generating aligned data...')
#     align_timestamps_raw(input_n_hour=12, file_folder=file_folder_raw)
#
#     train_test_split_aligned_data(file_folder_aligned)
#
# ######################################################################################################



######################################################################################################
# Generate processed data
######################################################################################################
    for pred in [0, 6, 12]:
        print(f'Prediction interval: {pred}')

        if pred == 0:
            generate_data_dx5849(pred_n_hour=pred, input_n_hour=12, neg_data=True, save_data=True)
        else:
            generate_data_dx5849(pred_n_hour=pred, input_n_hour=12, neg_data=False, save_data=True)
######################################################################################################



    # X_train, X_test, y_train, y_test = np.load(os.path.join(file_folder, 'data_split_8020.npy'), allow_pickle=True)
    # X_train = np.array(X_train)
    # X_test = np.array(X_test)
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)

    # info_test, data_test, label_test = generate_validation_data_raw(X_test, y_test, range(len(y_test)), file_folder=file_folder_aligned)
    # test_data = [info_test, data_test, label_test]
    # test_arr = np.empty(3, dtype=object)
    # test_arr[:] = test_data
    #
    # np.save(os.path.join(file_folder, 'test_data'), test_arr, allow_pickle=True)
    # info_test, data_test, label_test = np.load(os.path.join(file_folder, 'test_data.npy'), allow_pickle=True)


    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    # kf_splits = {}
    # for i, (train_i, val_i) in enumerate(kf.split(X_train, y_train)):
    #     kf_splits[i] = [train_i, val_i]
    #
    # split_id = 0
    # info_val, data_val, label_val = generate_validation_data_raw(X_train, y_train, kf_splits[split_id][1])
    # batch_generator = generate_data_batch_raw(X_train[kf_splits[split_id][0]], y_train[kf_splits[split_id][0]])
    # for iii in range(5):
    #     info, data, label = next(batch_generator)



    # X_train, X_test, y_train, y_test = np.load(os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_0_12_3src/data_split_8020.npy'),
    #                                            allow_pickle=True)
    # X_train = np.array(X_train)
    # X_test = np.array(X_test)
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)
    #
    # batch_generator = generate_balanced_data_batch_raw(X_train, y_train, batchsize=256)
    # while True:
    #     info, data, label = next(batch_generator)
    #
    # print(1)




    # X_train, X_test, y_train, y_test = np.load(os.path.join(file_folder, 'data_split_8020.npy'),
    #                                            allow_pickle=True)
    # X_train = np.array(X_train)
    # X_test = np.array(X_test)
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)

    # dataset_test = generate_tf_dataset_raw(X_test, y_test)
    # dataset_test = dataset_test['pos'].concatenate(dataset_test['neg'])
    # tf.data.experimental.save(dataset_test, os.path.join(file_folder, 'tf_dataset/dataset_test'))
    #
    # dataset_train = generate_tf_dataset_raw(X_train, y_train)
    # tf.data.experimental.save(dataset_train['pos'], os.path.join(file_folder, 'tf_dataset/dataset_train_pos'))
    # tf.data.experimental.save(dataset_train['neg'], os.path.join(file_folder, 'tf_dataset/dataset_train_neg'))


    # ds_test = tf.data.experimental.load('processed/patient_data/dx_pred_0_12_3src/tf_dataset/dataset_test')
    # ds_pos = tf.data.experimental.load('processed/patient_data/dx_pred_0_12_3src/tf_dataset/dataset_train_pos')
    # ds_neg = tf.data.experimental.load('processed/patient_data/dx_pred_0_12_3src/tf_dataset/dataset_train_neg')
    # ds_resampled = generate_equally_resampled_tf_dataset(ds_pos, ds_neg)
    # ds_resampled.batch(32).prefetch(2)
    #
    # ds_batch = ds_resampled.take(1)
    # ds_batch = list(ds_batch.as_numpy_iterator())``