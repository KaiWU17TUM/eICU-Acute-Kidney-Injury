import os
from pathlib import Path
os.chdir(os.path.join(Path(__file__).parent.resolve(), ".."))

import sys
sys.path.append(os.path.dirname(os.path.realpath(Path(__file__).parent.resolve())))

import time
import pickle
import numpy as np
import argparse
import tqdm

from create_dataset_torch import *

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

import shap

import warnings
warnings.filterwarnings("ignore")


def generate_stacked_dataset(data_folder, X_train, y_train, X_test, y_test, imp='mean', resample=None):
    if not resample:
        data_len = 144
    elif resample == 30:
        data_len = 24
    elif resample == 60:
        data_len = 12

    ds_train = PatientDataset(data_folder, X_train, y_train ,
                              model_type='SVM', imputation=imp, resample_rate=resample)
    loader_train = DataLoader(ds_train, batch_size=128,
                              sampler=ImbalancedDatasetSampler(ds_train), num_workers=8)

    ds_test = PatientDataset(data_folder, X_test, y_test,
                             model_type='SVM', imputation=imp, resample_rate=resample)
    loader_test = DataLoader(ds_test, batch_size=128, num_workers=8)



    data_train = torch.empty((0, 4 + data_len * 33))
    label_train = torch.empty(0)
    for data in loader_train:
        data_ = data['data']['static'].reshape(data['data']['static'].shape[0], -1)
        data_ = torch.concat((data_,
                              data['data']['time_series'].reshape(data['data']['time_series'].shape[0], -1)),
                             dim=1)
        data_train = torch.concat((data_train, data_), dim=0)
        label_train = torch.concat((label_train, data['label']), dim=0)

    data_test = torch.empty((0, 4 + data_len * 33))
    label_test = torch.empty(0)
    for data in tqdm(loader_test):
        data_ = data['data']['static'].reshape(data['data']['static'].shape[0], -1)
        data_ = torch.concat((data_,
                              data['data']['time_series'].reshape(data['data']['time_series'].shape[0], -1)),
                             dim=1)
        data_test = torch.concat((data_test, data_), dim=0)
        label_test = torch.concat((label_test, data['label']), dim=0)

    data_train = data_train.numpy()
    label_train = label_train.numpy()

    data_test = data_test.numpy()
    label_test = label_test.numpy()

    if resample == None:
        resample = 5
    np.savez(os.path.join(root_folder, f'ts_aligned_stacked_{pred}_{imp}_{resample}.npz'),
             X_train=data_train, y_train=label_train,
             X_test=data_test, y_test=label_test,
             allow_pickle=True)

    return 1


if __name__=='__main__':
    np.random.seed(0)
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--pred",
        nargs="*",
        type=int,
        default=[0, 6, 12],
    )
    CLI.add_argument(
        "--imp",
        nargs="*",
        type=str,
        default=["mean", "zero"],
    )
    CLI.add_argument(
        "--rr",
        nargs="*",
        type=int,
        default=[5, 30, 60]
    )
    args = CLI.parse_args()

    pred_list = args.pred
    imp_list = args.imp
    rr_list = args.rr

    for pred in pred_list:
        for imp in imp_list:
            for resample_rate in rr_list:
                print(f'##################### PRED {pred} - IMP {imp} - SAMPLERATE {resample_rate} #####################')

                root_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src')
                data_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src/ts_aligned')
                data_split = np.load(os.path.join(root_folder, 'data_split.npy'), allow_pickle=True).item()
                X_train = data_split['X_train']
                y_train = data_split['y_train']
                X_val = data_split['X_val']
                y_val = data_split['y_val']
                X_test = data_split['X_test']
                y_test = data_split['y_test']

                if not os.path.exists(os.path.join(root_folder, f'ts_aligned_stacked_{pred}_{imp}_{resample_rate}.npz')):
                    if resample_rate == 5:
                        resample_rate = None
                    generate_stacked_dataset(data_folder, X_train+X_val, y_train+y_val, X_test, y_test, imp=imp, resample=resample_rate)

                if resample_rate == None:
                    resample_rate = 5

                data = np.load(os.path.join(root_folder, f'ts_aligned_stacked_{pred}_{imp}_{resample_rate}.npz'))
                X_train = data['X_train']
                y_train = data['y_train']
                X_test = data['X_test']
                y_test = data['y_test']

                clf = svm.SVC()
                train_time_start = time.time()
                clf.fit(X_train, y_train)
                train_time_total = time.time() - train_time_start



                y_pred = clf.predict(X_test)
                METRICS = {
                    'acc': accuracy_score,
                    'recall': recall_score,
                    'precision': precision_score,
                    'f1': f1_score,
                    'auroc': roc_auc_score
                }
                for metric in METRICS:
                    print(f'{metric}: {METRICS[metric](y_test, y_pred)}')

                save_model_path = 'models/SVM/'
                Path(save_model_path).mkdir(parents=True, exist_ok=True)
                save_model_path = save_model_path + f'pred{pred}-imp{imp}-sampling{resample_rate}-raw.sav'
                pickle.dump(clf, open(save_model_path, 'wb'))
                with open('models/SVM/train_time.txt', 'a') as train_time_file:
                    train_time_file.write(f'pred{pred}-imp{imp}-sampling{resample_rate}-raw: {train_time_total}\n')

                # Create Tree Explainer object that can calculate shap values
                X_shap_summary = shap.kmeans(X_train, 20)
                X_shap, _, y_shap, _ = train_test_split(X_test, y_test, train_size=.02,
                                                        random_state=0, shuffle=True, stratify=y_test)

                shap_explainer = shap.KernelExplainer(clf.predict, X_shap_summary)
                shap_values = shap_explainer.shap_values(X_shap, nsamples=100)

                np.save(f'pred{pred}-imp{imp}-sampling{resample_rate}-raw-shap.npy', shap_values)
