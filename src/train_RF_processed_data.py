import os
import time
from pathlib import Path
os.chdir(os.path.join(Path(__file__).parent.resolve(), ".."))

import sys
sys.path.append(os.path.dirname(os.path.realpath(Path(__file__).parent.resolve())))

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

import shap

import warnings
warnings.filterwarnings("ignore")


def preprocess(X, y, imputation='mean'):
    nor = StandardScaler().fit(X)

    if imputation == 'mean':
        imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
        X = imp.transform(X)
    elif imputation == 'zero':
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit(X)
        X = imp.transform(X)

    X = nor.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, shuffle=True, stratify=y)

    return X_train, X_test, y_train, y_test

if __name__=='__main__':
    np.random.seed(0)
    # pred = int(sys.argv[1])
    # imp = sys.argv[2]

    for pred in [0, 6, 12]:

        for imp in ['zero', 'mean']:
            print(f'##################### PRED {pred} - IMP {imp} #####################')

            data_pos = np.load(f'processed/patient_data/dx_pred_{pred}_12_3src/data_positive.npy')
            data_neg = np.load(f'processed/patient_data/dx_pred_{pred}_12_3src/data_negative.npy')
            X_raw = np.vstack((data_pos, data_neg))
            y_raw = [1] * len(data_pos) + [0] * len(data_neg)
            X_train, X_test, y_train, y_test = preprocess(X_raw, y_raw, imputation=imp)
            neg_weight = sum(y_raw) / len(y_raw)
            pos_weight = 1 - neg_weight
            sample_weight = np.ones_like(y_train) * neg_weight
            sample_weight[np.array(y_train)==1] = pos_weight


            tree_depth = 25
            clf = RandomForestClassifier(max_depth=tree_depth, random_state=0)
            train_time_start = time.time()
            clf.fit(X_train, y_train, sample_weight=sample_weight)
            train_time_total = time.time() - train_time_start

            save_model_path = 'models/RF/RF_processed/'
            Path(save_model_path).mkdir(parents=True, exist_ok=True)
            with open('models/RF/RF_processed/train_time.txt', 'a') as train_time_file:
                train_time_file.write(f'pred{pred}-imp{imp}-processed: {train_time_total}\n')

            save_model_path = save_model_path + f'pred{pred}-imp{imp}-processed.sav'
            pickle.dump(clf, open(save_model_path, 'wb'))


            save_path = os.path.join('models/RF/', 'model_performance_processed')
            Path(save_path).mkdir(parents=True, exist_ok=True)

            y_pred = clf.predict(X_test)
            METRICS = {
                'auroc': roc_auc_score,
                'recall': recall_score,
                'precision': precision_score,
                'f1': f1_score,
                'acc': accuracy_score,
            }

            columns = ['test_roc', 'test_recall', 'test_precision', 'test_f1', 'test_acc', 'test_loss']

            res = []
            for metric in METRICS:
                metric_val = METRICS[metric](y_test, y_pred)
                res.append(metric_val)
                print(f'{metric}: {metric_val}')
            res.append(np.nan)
            res_dict = {(pred, imp, None): res}
            df = pd.DataFrame.from_dict(res_dict, orient='index', columns=columns)
            df.to_csv(os.path.join(save_path, f'{pred}-{imp}-{None}.csv'), sep=';')


            # Create Tree Explainer object that can calculate shap values
            X_shap, _, y_shap, _ = train_test_split(X_test, y_test, train_size=.02, stratify=y_test, random_state=0)
            shap_explainer = shap.TreeExplainer(clf)
            shap_values = shap_explainer.shap_values(X_shap)

            save_path = os.path.join('models/RF/', 'RF_processed_shap')
            Path(save_path).mkdir(parents=True, exist_ok=True)
            np.save(os.path.join(save_path, f'pred{pred}-imp{imp}-processed-shap.npy'), shap_values)