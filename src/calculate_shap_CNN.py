import os
from pathlib import Path

import pandas as pd

os.chdir(os.path.join(Path(__file__).parent.resolve(), ".."))

import sys
sys.path.append(os.path.dirname(os.path.realpath(Path(__file__).parent.resolve())))

import glob

from sklearn.model_selection import train_test_split
import shap
from tqdm import tqdm

from train_CNN import *


if __name__ == '__main__':
    # model_path = 'models/Conv1D_torch/2layer'
    model_path = 'models/Conv1D_torch/CNN_2layer'
    model_versions = os.listdir(model_path)
    for version in model_versions:
        v = version.split('_')
        if len(v) < 5:
            continue
        file_path = os.path.join(model_path, version)
        model_file = glob.glob(file_path + '/*.ckpt')
        print(model_file)
        if len(model_file) == 1:
            model_file = model_file[0]
        else:
            assert "more than one model file in the folder!"

        pred = int(v[2])
        imp = v[3]
        if imp == 'm':
            imp = 'mean'
        elif imp == 'z':
            imp = 'zero'
        else:
            assert f"Unkown imputation: {imp}"
        rr = int(v[4])
        if rr == 5:
            resample_rate = None
        else:
            resample_rate = rr

        # save_path = os.path.join(model_path, '..', 'shap')
        # if os.path.exists(os.path.join(save_path, f'pred{pred}-imp{imp}-sampling{resample_rate}.npz')):
        #     continue

        print(f'##################### PRED {pred} - IMP {imp} - SAMPLERATE {rr} #####################')

        root_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src')
        data_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src/ts_aligned')
        data_split = np.load(os.path.join(root_folder, 'data_split.npy'), allow_pickle=True).item()
        X_test = data_split['X_test']
        y_test = data_split['y_test']
        X_train = data_split['X_train'] + data_split['X_val']
        y_train = data_split['y_train'] + data_split['y_val']

        n_input_static = 4
        n_input_seq = 33
        n_filters_per_channel = 3

        if rr == 5:
            cnn_kernel_size = 8
            cnn_stride = 4
            pool_kernel_size = 4
        elif rr == 30:
            cnn_kernel_size = 2
            cnn_stride = 2
            pool_kernel_size = 3
        elif rr == 60:
            cnn_kernel_size = 2
            cnn_stride = 2
            pool_kernel_size = 2

        config = {
            "n_input_static": n_input_static,
            "n_input_seq": n_input_seq,
            "n_classes": 1,
            "cnn_kernel_size": cnn_kernel_size,
            "cnn_stride": cnn_stride,
            "n_group_cnn": n_input_seq,
            "n_hidden": n_filters_per_channel * n_input_seq,
            "pool_kernel_size": pool_kernel_size,
            "dropout_dense": 0.5,
            "lr": 1e-3,
            "batch_size": 64,
            'resample_rate': resample_rate,
            'imputation': imp,
        }
        model = CNN_CLF.load_from_checkpoint(checkpoint_path=model_file,
                                             config=config, data_folder=data_folder, data_split=data_split)
        model.eval()

        trainer = Trainer(gpus=1)
        test_result = trainer.test(model)[0]

        # save_path = os.path.join(model_path, '..', 'model_performance')
        save_path = os.path.join(model_path, '..', 'sensitive_test')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        columns = ['test_roc', 'test_recall', 'test_precision', 'test_f1', 'test_acc', 'test_loss']

        res = []
        for col in columns:
            res.append(test_result[col])
        res_dict = {(pred, imp, rr): res}
        df = pd.DataFrame.from_dict(res_dict, orient='index', columns=columns)
        # df.to_csv(os.path.join(save_path, f'{pred}-{imp}-{rr}.csv'), sep=';')
        df.to_csv(os.path.join(save_path, f'{pred}-{imp}-{rr}-{v[1]}.csv'), sep=';')

        # model.to(device='cpu')
        #
        #
        # ds_background = PatientDataset(data_folder, X_train, y_train, resample_rate=resample_rate, imputation=imp)
        # loader_background = DataLoader(ds_background, batch_size=500,
        #                                sampler=ImbalancedDatasetSampler(ds_background), num_workers=2)
        # data_background = next(iter(loader_background))
        #
        # ds_shap = PatientDataset(data_folder, X_test, y_test, resample_rate=resample_rate, imputation=imp)
        # loader_shap = DataLoader(ds_shap, batch_size=int(.02*(ds_shap.__len__())),
        #                          sampler=ImbalancedDatasetSampler(ds_shap), num_workers=2)
        # data = next(iter(loader_shap))
        #
        # for d in data_background['data']:
        #     data_background['data'][d] = data_background['data'][d].to(device='cpu')
        # data_background['label'].to(device='cpu')
        #
        # for d in data['data']:
        #     data['data'][d] = data['data'][d].to(device='cpu')
        # data['label'].to(device='cpu')
        #
        #
        # e = shap.DeepExplainer(
        #     model,
        #     [data_background['data']['static'], data_background['data']['time_series']],
        # )
        #
        # shap_values = e.shap_values(
        #     [data['data']['static'], data['data']['time_series']]
        # )
        # sv = np.empty(len(shap_values), dtype=object)
        # sv[:] = shap_values
        #
        # save_path = os.path.join(model_path, '..', 'shap')
        # Path(save_path).mkdir(parents=True, exist_ok=True)
        #
        # np.savez(os.path.join(save_path, f'pred{pred}-imp{imp}-sampling{rr}.npz'),
        #          shap=sv, label=data['label'], data=data['data'],
        #          allow_pickle=True)
