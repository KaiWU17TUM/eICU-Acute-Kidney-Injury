import os.path

import numpy as np
import pandas as pd

DEMO_ITEMS = ['patientunitstayid', 'age', 'gender', 'apacheadmissiondx', 'unitdischargestatus', 'hospitaldischargestatus']

DX_SELECTED= [
    'Infarction, acute myocardial (MI)',
    'CABG alone, coronary artery bypass grafting',
    'CABG with aortic valve replacement',
    'CHF, congestive heart failure',
    'Aortic valve replacement (isolated)',
    'Cardiac arrest (with or without respiratory arrest; for respiratory arrest see Respiratory System)',
    'Arrest, respiratory (without cardiac arrest)',
    'Angina, unstable (angina interferes w/quality of life or meds are tolerated poorly)',
    'Angina, stable (asymp or stable pattern of symptoms w/meds)',
    'Rhythm disturbance (atrial, supraventricular)',
    'Rhythm disturbance (conduction defect)',
    'Rhythm disturbance (ventricular)',
    'Cardiovascular medical, other',
    'Cardiomyopathy',
    'Mitral valve repair',
    'Mitral valve replacement',
    'Shock, cardiogenic',
    'Cardiovascular surgery, other',
    'Ablation or mapping of cardiac conduction pathway',
    'Thrombus, arterial',
    'Pericardial effusion/tamponade',
    'Efffusion, pericardial',
    'Aortic and Mitral valve replacement',
    'Hypertension, uncontrolled (for cerebrovascular accident-see Neurological System)',
]

DX_NAME_PATH = [dx.replace(" ", "_").replace(',', '').replace('/', '_') for dx in DX_SELECTED]

AGE_RANGES = [str(i) for i in range(0, 90, 10)] + ['> 89']



def add_list_elem_in_dict(dict, key, value):
    if key in dict:
        dict[key].append(value)
    else:
        dict[key] = [value]

def sort_dict(dictionary, key):
    sorted_idx = np.argsort(np.array(dictionary[key]))
    for k in dictionary:
        dictionary[k] = np.array(dictionary[k])[sorted_idx].tolist()
    return dictionary


def calculate_delta(data, timestamp, max_interval=24):
    delta = []
    valid_idx = [i for i in range(len(data))
                 if data[i] is not None and data[i] == data[i]]

    delta += [None] * valid_idx[0]
    for i in range(len(valid_idx) - 1):
        interval = valid_idx[i + 1] - valid_idx[i]
        if interval < max_interval:
            delta += [(data[valid_idx[i + 1]] - data[valid_idx[i]]) / interval] * interval
        else:
            delta += [None] * interval
    delta += [None] * (len(timestamp) - valid_idx[-1])
    return delta


def sort_sub_list(l:list, mask, idx):
    l = np.array(l)
    l[mask] = l[mask][idx]
    return list(l)


def argsort_sub_list(l:list, mask):
    l = np.array(l)
    sort_idx = np.argsort(l[mask])
    return sort_idx


