import os
from pathlib import Path
os.chdir(os.path.join(Path(__file__).parent.resolve(), ".."))

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from tqdm import tqdm



def resample_aligned_data(sample_data, resample_freq='10Min'):
    sample_data['Timestamp'] = pd.to_datetime(sample_data['Timestamp'], unit='m')
    sample_data = sample_data.set_index('Timestamp')
    resampled_data = sample_data.resample(resample_freq).mean()

    return resampled_data

def data_zero_masking(data):
    data[np.isnan(data)] = 0
    return data

class PatientDataset(Dataset):
    def __init__(self, folder, sample_list, label_list,):
        self.folder = folder
        self.sample_list = sample_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        y = self.label_list[idx]
        if y == 0:
            file_path = os.path.join(self.folder, 'neg')
        else:
            file_path = os.path.join(self.folder, 'pos')

        x_info = pd.read_csv(os.path.join(file_path, 'info', self.sample_list[idx] + '.csv'), sep=';', header=0,
                             index_col=0).to_numpy()
        x_data = pd.read_csv(os.path.join(file_path, 'data', self.sample_list[idx] + '.csv'), sep=';', header=0,
                             index_col=[0, 1]).to_numpy()

        return {
            'data': {
                'static': x_info,
                'time_series': x_data,
            },
            'label': y
        }



if __name__=='__main__':
    ##################################################################################################
    # Calculate Normalization Parameters
    ##################################################################################################

    for pred in [6, 12]:

        root_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src')
        data_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src/ts_aligned')

        data_split = np.load(os.path.join(root_folder, 'data_split.npy'), allow_pickle=True).item()
        X_train = data_split['X_train']
        y_train = data_split['y_train']
        X_val = data_split['X_val']
        y_val = data_split['y_val']
        X_test = data_split['X_test']
        y_test = data_split['y_test']

        X_all = X_train + X_val
        y_all = y_train + y_val

        ds = PatientDataset(data_folder, X_all, y_all)
        ds_loader = DataLoader(ds, 1000, num_workers=6)

        dim_t = 33
        dim_s = 4

        mean = torch.zeros(dim_t)
        std = torch.zeros(dim_t)
        count = torch.zeros(dim_t)

        mean_s = torch.zeros(dim_s)
        std_s = torch.zeros(dim_s)
        count_s = torch.zeros(dim_s)

        for batch in tqdm(ds_loader):
            info = batch['data']['static']
            info = info.view(-1, dim_s)
            mask = info.isnan()
            info = torch.masked_fill(info, mask, 0)
            count_s += (~mask).sum(dim=0)
            mean_s += info.sum(dim=0)

            data = batch['data']['time_series']
            data = data.view(-1, dim_t)
            mask = data.isnan()
            data = torch.masked_fill(data, mask, 0)
            count += (~mask).sum(dim=0)
            mean += data.sum(dim=0)
        mean_s = mean_s / count_s
        mean = mean / count

        for batch in tqdm(ds_loader):
            info = batch['data']['static']
            info = info.view(-1, dim_s)
            mask = info.isnan()
            info = torch.masked_fill(info, mask, 0)
            std_s += (info - mean_s).pow(2).sum(dim=0)

            data = batch['data']['time_series']
            data = data.view(-1, dim_t)
            mask = data.isnan()
            data = torch.masked_fill(data, mask, 0)
            std += (data - mean).pow(2).sum(dim=0)
        std_s = torch.sqrt(std_s / count_s)
        std = torch.sqrt(std / count)

        dataset_norm_params = {
            'static_mean': mean_s,
            'static_std': std_s,
            'data_mean': mean,
            'data_std': std,
        }


        save_path = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src', 'norm_params.pt')
        torch.save(dataset_norm_params, save_path)

        # norm_params = torch.load(save_path)

    ##################################################################################################














