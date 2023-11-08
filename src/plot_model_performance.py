import os
from pathlib import Path
os.chdir(os.path.join(Path(os.path.abspath('')).parent.resolve()))
print(os.getcwd())

import sys
sys.path.append(os.path.dirname(os.path.realpath(Path(os.path.abspath('')).parent.resolve())))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    rf_result_processed_folder = 'models/RF/model_performance_processed'
    svm_result_processed_folder = 'models/SVM/model_performance_processed'
    rf_result_raw_folder = 'models/RF/model_performance_raw'
    svm_result_raw_folder = 'models/SVM/model_performance_raw'
    cnn_result_folder = 'models/Conv1D_torch/model_performance'
    lstm_result_folder = 'models/LSTM_torch/model_performance'
    lstmatt_result_folder = 'models/LSTMATT_torch/model_performance'

    model_list = ['RF_compact', 'SVM_compact', 'RF_raw', 'SVM_raw', 'CNN', 'LSTM', 'LSTMATT']
    result_list = [rf_result_processed_folder, svm_result_processed_folder,
                   rf_result_raw_folder, svm_result_raw_folder,
                   cnn_result_folder, lstm_result_folder, lstmatt_result_folder]

    index = ['AUROC', 'Recall', 'Precision', 'F1', 'Accuracy', 'Loss']
    result_tab = pd.DataFrame(index=index)

    for model, result_folder in zip(model_list, result_list):
        for imp in ['zero', 'mean']:
            for rr in [5, 30, 60]:
                for pred in [0, 6, 12]:
                    try:
                        result = pd.read_csv(os.path.join(result_folder, f'{pred}-{imp}-{rr}.csv'),
                                             sep=';', header=0, index_col=0).transpose()
                    except:
                        result = pd.read_csv(os.path.join(result_folder, f'{pred}-{imp}-None.csv'),
                                             sep=';', header=0, index_col=0).transpose()
                    result.index = index
                    result_tab[(model, pred, imp, rr)] = result[result.columns[0]]

    result_tab.columns = pd.MultiIndex.from_tuples(result_tab.columns, names=['model', 'pred', 'imp', 'rr'])
    result_tab = result_tab.T

    index_model = result_tab.index.get_level_values('model')
    index_pred = result_tab.index.get_level_values('pred')
    index_imp = result_tab.index.get_level_values('imp')
    index_rr = result_tab.index.get_level_values('rr')



    score = 'Precision'

    cmap = plt.get_cmap("tab10")
    markers = ['o', 'x', 'v', ]
    lines = ['solid', 'dotted', 'dashdot']

    for ii, imp in enumerate(['zero', 'mean']):
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(left=0.1, bottom=None, right=0.7, top=0.95, wspace=None, hspace=None)

        for mi, model in enumerate(model_list):
            for ri, rr in enumerate([5, 30, 60]):
                if 'compact' in model and rr != 5:
                    continue
                y = result_tab.iloc[(index_model == model) &
                                    (index_imp == imp) &
                                    (index_rr == rr)][score]
                x = [0, 6, 12]
                if 'compact' in model:
                    ax.plot(x, y, c=cmap(7-mi), linestyle=lines[ri], marker=markers[ri],
                            label=f'{model}')
                else:
                    ax.plot(x, y, c=cmap(7-mi), linestyle=lines[ri], marker=markers[ri],
                            label=f'{model}-{rr}min')
                ax.set_xticks([0, 6, 12])
                ax.set_xlabel('Prediction Interval [h]', fontsize=18)
                ax.set_ylabel(f'{score}', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
        save_path = f'plots/model_performance/model_performance_{score}_imp{imp}'
        fig.savefig(save_path)

    # for mi, model in enumerate(model_list):
    #     for si, score in enumerate(['AUROC', 'Recall', 'Precision']):
    #         for ri, rr in enumerate([5, 30, 60]):
    #             y = result_tab.iloc[(index_model == model) &
    #                                 (index_imp == imp) &
    #                                 (index_rr == rr)][score]
    #             x = [0, 6, 12]
    #             if mi > 1:
    #                 ax[si][mi].plot(x, y, label=f'{rr}min')
    #             if mi == 6:
    #                 # ax[si][mi].legend(loc='lower right');
    #                 ax[si][mi].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #             if mi <= 1 and ri == 0:
    #                 ax[si][mi].plot(x, y)
    #
    #             # if si == 2:
    #             #     ax[si][mi].set_xlabel('prediction interval [h]')
    #             if mi == 0:
    #                 ax[si][mi].set_ylabel(f'{score}')
    #             if si == 0:
    #                 ax[si][mi].set_title(f'{model}')
    #
    #             ax[si][mi].set_xticks([0, 6, 12])

    plt.show()
    print('111')