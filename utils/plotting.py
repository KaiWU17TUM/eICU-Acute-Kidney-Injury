import math

import matplotlib.pyplot as plt
import numpy as np
import pathlib

def plot_patient_data(patients, entry_group, entry, entry_time, title='', save_path=''):
    x_live = []
    y_live = []
    x_expired = []
    y_expired = []
    for pid in patients:
        if patients[pid]['unitdischargestatus'] == 'Alive':
            x_live += patients[pid][entry_group][entry_time]
            y_live += patients[pid][entry_group][entry]
        elif patients[pid]['unitdischargestatus'] == 'Expired':
            x_expired += patients[pid][entry_group][entry_time]
            y_expired += patients[pid][entry_group][entry]

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.gca()

    x_live_valid = []
    y_live_valid = []
    for x, y in zip(x_live, y_live):
        if y == y and y != None and type(y)==float:
            x_live_valid.append(x)
            y_live_valid.append(y)
    x_live_valid = np.array(x_live_valid)
    y_live_valid = np.array(y_live_valid)
    x_live_valid = x_live_valid[np.abs(y_live_valid) < 50*np.median(y_live_valid)]
    y_live_valid = y_live_valid[np.abs(y_live_valid) < 50*np.median(y_live_valid)]

    x_expired_valid = []
    y_expired_valid = []
    for x, y in zip(x_expired, y_expired):
        if y == y and y != None and type(y)==float:
            x_expired_valid.append(x)
            y_expired_valid.append(y)
    x_expired_valid = np.array(x_expired_valid)
    y_expired_valid = np.array(y_expired_valid)
    # x_expired_valid = x_expired_valid[np.abs(y_expired_valid) < 50*np.median(y_expired_valid)]
    # y_expired_valid = y_expired_valid[np.abs(y_expired_valid) < 50*np.median(y_expired_valid)]

    ax.plot(x_live_valid/60, y_live_valid, marker='.', ms=1, linestyle='None', label='Alive')
    ax.plot(x_expired_valid/60, y_expired_valid, marker='.', ms=1, linestyle='None', label='Expired')
    plt.title(f'{title} Distribution over Time')
    ax.set_xlabel('Time (hour)')
    ax.set_ylabel(title)
    ax.legend()
    ax.grid()

    if save_path:
        file_path = save_path + title.replace(' ', '_') + '.png'
        plt.savefig(file_path)
    plt.close(fig)
    return 1
