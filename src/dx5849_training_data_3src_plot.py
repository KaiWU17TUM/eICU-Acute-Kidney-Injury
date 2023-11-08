import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import icd10
from icd9cms.icd9 import search

from collections import Counter

import sys
sys.path.insert(1, os.path.join(os.path.abspath('.'), '..'))

from utils.data_io import *
from utils.common import *

uid_dict = load_input_uid_dict()
uid_tables = {
    'Vital Signs': [1e5, 3e5],
    'Intake Output': [3e5, 4e5],
    'Lab Tests': [4e5, 5e5],
    # 'Infusion': [5e5, 6e5],
    # 'Nurse Charting': [6e5, 7e5],
}

dx_5849 = [700003, 700163, 700499, 700571, 700596, 700854]
pid_5849 = np.load('processed/pid_5849_from_structured_dsv.npy')

pred_n_hour = 6
input_n_hour = 6

save_path = 'processed/icd9_5849/dx_pred_' + str(pred_n_hour) + '_' + str(input_n_hour) + '_3src'
Path(save_path).mkdir(parents=True, exist_ok=True)



for pid in pid_5849:
    info, data = load_patient_data_dsv(pid)
    dx_events = data[data['UID'].isin(dx_5849)]
    dx_events_ts = dx_events.loc[(dx_events['Offset'] > (pred_n_hour+input_n_hour)*60) & (dx_events['Offset'].diff() > (pred_n_hour+input_n_hour)*60), 'Offset']

    for di, dx_ts in enumerate(dx_events_ts):
        fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        for ti, (table, uid_range) in enumerate(uid_tables.items()):
            select_uid = data.loc[(data['UID'] > uid_range[0]) & (data['UID'] < uid_range[1]), 'UID']
            select_uid = select_uid.unique()

            for uid in select_uid:
                try:
                    select_data = data.loc[(data['UID'] == uid) & (data['Offset']>dx_ts-(pred_n_hour+input_n_hour)*60) & (data['Offset']<dx_ts), 'Value'].astype(float)
                except:
                    break
                if select_data.shape[0] == 0:
                    continue
                select_ts = data.loc[(data['UID'] == uid) & (data['Offset']>dx_ts-(pred_n_hour+input_n_hour)*60) & (data['Offset']<dx_ts), 'Offset'] / 60
                select_label = uid_dict[uid]
                if 'Weight' in select_label:
                    continue
                if select_data.shape[0] == 1:
                    ax[ti % 3].scatter(select_ts, select_data, label=select_label)
                else:
                    ax[ti % 3].plot(select_ts, select_data, label=select_label)
                # if table=='Vital Signs':
                #     ax[ti // 3][ti % 3].plot(select_ts, select_data, label=select_label)
                # else:
                #     ax[ti // 3][ti % 3].scatter(select_ts, select_data, label=select_label)

                ax[ti % 3].set_title(table, fontsize=12)
                ax[ti % 3].set_xlabel('Time after admission[h]', fontsize=10)
                ax[ti % 3].legend(loc='upper right', fontsize=8, borderaxespad=0.)
                # ax[ti // 3][ti % 3].legend(loc='center right', fontsize=12, bbox_to_anchor=(1.15, 0.5), borderaxespad=0.)

        fig.savefig(os.path.join(save_path, str(pid)+'_'+str(dx_ts//60)+'.png'), transparent=False, bbox_inches='tight')
        plt.cla()
        plt.close()











