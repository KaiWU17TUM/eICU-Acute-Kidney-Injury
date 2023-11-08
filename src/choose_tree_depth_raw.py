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


cv_score_hist = {}
val_score_hist = {}

for resample in [0, 30, 60]:
    data = np.load(os.path.join(root_folder, f'ts_aligned_stacked_{pred}_{imp}_{resample}.npz'))
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    cv_score_hist[resample] = []
    val_score_hist[resample] = []


    for tree_depth in tqdm(range(20, 41)):
        clf = RandomForestClassifier(max_depth=tree_depth, random_state=0)
        cv_score = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        cv_score_hist += [cv_score.mean()]

        clf.fit(X_train, y_train)
        val_score = clf.score(X_test, y_test)
        val_score_hist += [val_score]
        print(f'Tree Depth: {tree_depth} - CV Score: {cv_score} - Val Score: {val_score}')