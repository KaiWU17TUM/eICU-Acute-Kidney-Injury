import os

os.chdir('./..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from src.create_dataset_torch import *

import warnings

warnings.filterwarnings("ignore")

import shap

import pickle


def preprocess(X, y):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
    X = imp.transform(X)

    nor = Normalizer().fit(X)  # fit does nothing.
    X = nor.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test

root_folder = os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_0_12_3src')
data_folder = os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_0_12_3src/ts_aligned')
data = np.load(os.path.join(root_folder, 'ts_aligned_stacked.npz'))
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)



svm_model_file = 'models/SVM/full_data.sav'
# pickle.dump(clf, open(svm_model_file, 'wb'))

clf = pickle.load(open(svm_model_file, 'rb'))

tmp_file = os.listdir(os.path.join(data_folder, 'pos', 'data'))[0]
tmp_file = os.path.join(data_folder, 'pos', 'data', tmp_file)
feature_names = pd.read_csv(tmp_file, sep=';', header=0, index_col=[0,1]).columns.to_list()
# feature_names = [
#     'RDW', 'RESPIRATION', 'NIBP_m', 'NA', 'MCHC', 'MG', 'URINE',
#     'NIBP_s', 'HBG', 'HCO3', 'UREA', 'MCH', 'SAO2', 'MCV', 'GLC',
#     'PLT', 'HR', 'MPV', 'CL', 'K', 'CA', 'RBC', 'CR', 'NIBP_d', 'WBC'
# ]

# Create Tree Explainer object that can calculate shap values
X_shap, _, y_shap, _ = train_test_split(X_test, y_test, train_size=.004,
                                        random_state=0, shuffle=True, stratify=y_test)

shap_explainer = shap.KernelExplainer(clf.predict, X_shap)
shap_values = shap_explainer.shap_values(X_shap, nsamples=100)

shap.summary_plot(shap_values, X_shap, feature_names=feature_names)