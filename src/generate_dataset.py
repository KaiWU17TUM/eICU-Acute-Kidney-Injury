import os
from pathlib import Path
import shutil

os.chdir(os.path.join(Path(__file__).parent.resolve(), ".."))

import sys
sys.path.append(os.path.dirname(os.path.realpath(Path(__file__).parent.resolve())))
# print(os.path.abspath("."))

from utils.data_io import *
from utils.common import *
from src.classPatientDataset import PatientDataset

import warnings
warnings.filterwarnings("ignore")

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

feature_names = [
    'GENDER', 'AGE', 'ADMISSIONWEIGHT', 'ADMISSIONHEIGHT',
    'HEARTRATE', 'SPO2', 'RESPIRATION', 'CVP',
    'NONINVASIVEDIASTOLIC', 'NONINVASIVEMEAN', 'NONINVASIVESYSTOLIC',
    'URINE',
    'GLC', 'K', 'HCO3', 'NA', 'UREA', 'CR', 'CL', 'CA', 'HBG',
    'PLT', 'RBC', 'WBC', 'MCHC', 'MCV', 'MCH', 'RDW', 'MG', 'MPV',
     'PH', 'PAO2', 'PACO2', 'BASEEXCESS', 'SAO2', 'CAIONIZED', 'METHB', 'COHB'
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


print("Start generating dataset...")


if __name__ == "__main__":
    PRED_INTERVAL = 6
    DATA_INTERVAL = 12

    ######################################################################################################
    # Generate raw data and aligned data
    ######################################################################################################
    # local machine
    # file_folder_dsv = '/home/kai/workspace/DHM/finished_projects/MICCAI2022_AKF_feature_importance/data/all_unit_type_structured_dsv_data_v1'
    # dhmgpuserver docker
    # file_folder_dsv = '/workspace/data/EICU/data/all_unit_type_structured_dsv_data_v1'
    # dhmgpuserver virtual env
    file_folder_dsv = '/home/kai/DigitalICU/EICU/data/all_unit_type_structured_dsv_data_v1/'
    file_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{PRED_INTERVAL}_{DATA_INTERVAL}')
    file_folder_raw = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{PRED_INTERVAL}_{DATA_INTERVAL}/raw')
    file_folder_aligned = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{PRED_INTERVAL}_{DATA_INTERVAL}/ts_aligned')

    # pid_all = os.listdir(file_folder_dsv)
    # pid_all = [os.path.splitext(p)[0].replace('data_', '').replace('info_', '') for p in pid_all]
    # pid_all = sorted(list(set(pid_all)))
    # np.save('processed/pid_all', pid_all)
    #
    # pid_5849_all = get_pid_5849_all(file_folder_dsv, pid_all, dx_5849)
    # np.save('processed/pid_5849_all', pid_5849_all)
    #
    # pid_non_5849 = [pid for pid in pid_all if pid not in pid_5849_all]
    # np.save('processed/pid_non_5849', pid_non_5849)

    pid_all = np.load(os.path.abspath('processed/pid_all.npy'))
    pid_5849_all = np.load('processed/pid_5849_all.npy')
    pid_non_5849 = np.load('processed/pid_non_5849.npy')
    pid_list = [pid_5849_all, pid_non_5849]

    print('Generating raw data...')
    generate_dataset_raw(dsv_path=file_folder_dsv, pid_list=pid_list, dx_list=dx_5849,
                         selected_input_uid=selected_input_uid, uid_dict=uid_dict, feature_names=feature_names,
                         pred_n_hour=PRED_INTERVAL, input_n_hour=DATA_INTERVAL,
                         neg_data=True, save_data=True)

    print('Generating aligned data...')
    align_timestamps_raw(file_folder=file_folder_raw, feature_names=feature_names, input_n_hour=DATA_INTERVAL)
    print('Copying patient static data to aligned folder...')
    shutil.copytree(os.path.join(file_folder_raw, 'pos/info'), os.path.join(file_folder_aligned, 'pos/info'))
    shutil.copytree(os.path.join(file_folder_raw, 'neg/info'), os.path.join(file_folder_aligned, 'neg/info'))

    print('Data split: train, validation, test...')
    data_split = train_test_split_aligned_data(file_folder_aligned)
    data_split = np.load(os.path.join(file_folder, 'data_split.npy'), allow_pickle=True).item()
    ######################################################################################################


    ######################################################################################################
    # Generate processed data
    ######################################################################################################
    print('Generating processed data...')
    generate_dataset_processed(dsv_path=file_folder_dsv, pid_list=pid_list, dx_list=dx_5849,
                               selected_input_uid=selected_input_uid,
                               pred_n_hour=PRED_INTERVAL, input_n_hour=DATA_INTERVAL,
                               neg_data=True, save_data=True)
    ######################################################################################################


    ######################################################################################################
    # Calculate calibration params
    ######################################################################################################
    print('Calculating normalization parameters...')
    norm_params = calculate_normalization_params(data_split, file_folder_aligned)
    torch.save(norm_params, os.path.join(file_folder, 'norm_params.pt'))
    ######################################################################################################

    ######################################################################################################
    # save data in npz
    ######################################################################################################
    print('Save data in npz file...')
    X_train = data_split['X_train'] + data_split['X_val']
    y_train = data_split['y_train'] + data_split['y_val']
    X_test = data_split['X_test']
    y_test = data_split['y_test']

    data_np = {
        'train': {'static': None, 'time_series': None, 'label': None},
        'test': {'static': None, 'time_series': None, 'label': None},
    }
    for X, y, type in zip([X_train, X_test], [y_train, y_test], ['train', 'test']):
        ds = PatientDataset(file_folder_aligned, X, y, imputation='zero')
        ds_loader = DataLoader(ds, batch_size=128, num_workers=16)

        data_static = torch.zeros((0, 4))
        data_ts = torch.zeros((0, 144, 34))
        data_label = torch.zeros((0, 1))
        for batch in tqdm(ds_loader):
            data_static = torch.concat((data_static, batch['data']['static'].reshape(batch['data']['static'].shape[0], -1)), dim=0)
            data_ts = torch.concat((data_ts, batch['data']['time_series']))
            data_label = torch.concat((data_label, batch['label'].reshape(batch['label'].shape[0], 1)), dim=0)
        data_np[type]['static'] = data_static
        data_np[type]['time_series'] = data_ts
        data_np[type]['label'] = data_label

    np.savez(os.path.join(file_folder, f'ts_aligned_{PRED_INTERVAL}_5_zero.npz'),
             data_train=data_np['train'],
             data_test=data_np['test'],
             allow_pickle=True)
    ######################################################################################################
