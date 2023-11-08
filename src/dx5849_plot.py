import os
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(1, os.path.join(os.path.abspath('.'), '..'))

from utils.data_io import *
from utils.common import *

uid_dict = load_input_uid_dict()
uid_tables = {
    'Vital Signs': [1e5, 3e5],
    'Intake Output': [3e5, 4e5],
    'Lab Tests': [4e5, 5e5],
    'Infusion': [5e5, 6e5],
    'Nurse Charting': [6e5, 7e5],
}

dx_5849 = [700003, 700163, 700499, 700571, 700596, 700854]
pid_5849 = np.load('processed/pid_5849_from_structured_dsv.npy')

save_path = 'processed/icd9_5849/'
for table in uid_tables:
    Path(os.path.join(save_path, table.replace(' ', '_').lower())).mkdir(parents=True, exist_ok=True)




for pid in pid_5849:
    info, data = load_patient_data_dsv(pid)
    for table, uid_range in uid_tables.items():
        select_uid = data.loc[(data['UID'] > uid_range[0]) & (data['UID'] < uid_range[1]), 'UID']
        select_uid = select_uid.unique()

        fig, ax = plt.subplots(figsize=(12, 6))
        for uid in select_uid:
            try:
                select_data = data.loc[data['UID'] == uid, 'Value'].astype(float)
            except:
                continue
            select_ts = data.loc[data['UID'] == uid, 'Offset'] / 60
            select_label = uid_dict[uid]
            ax.plot(select_ts, select_data, label=select_label)

        for dx in dx_5849:
            if data.loc[data['UID'] == dx].shape[0] > 0:
                dx_ts = data.loc[data['UID'] == dx, 'Offset'] / 60
                for ts in dx_ts:
                    ax.axvline(ts, c='r', linewidth=2)

        ax.set_title(table, fontsize=20)
        ax.set_xlabel('Time after admission[h]', fontsize=14)
        plt.legend(loc='center right', fontsize=12, bbox_to_anchor=(1.15, 0.5), borderaxespad=0.)

        save_to = os.path.join(save_path, table.replace(' ', '_').lower(), str(pid)+'.png')
        fig.savefig(save_to, transparent=False, bbox_inches='tight')
