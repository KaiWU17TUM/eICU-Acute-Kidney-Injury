import os
from pathlib import Path
os.chdir(os.path.join(Path(__file__).parent.resolve(), ".."))

import sys
sys.path.append(os.path.dirname(os.path.realpath(Path(__file__).parent.resolve())))

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torchsampler import ImbalancedDatasetSampler

def resample_aligned_data(sample_data, resample_freq='10Min'):
    sample_data['Timestamp'] = pd.to_datetime(sample_data['Timestamp'], unit='m')
    sample_data = sample_data.set_index('Timestamp')
    resampled_data = sample_data.resample(resample_freq).last()

    return resampled_data

def data_zero_masking(data):
    data[np.isnan(data)] = 0
    return data

class PatientDataset(Dataset):
    def __init__(self, folder, sample_list, label_list, resample_rate=None, imputation='mean', model_type='otherThanGRUD'):
        self.folder = folder
        self.sample_list = sample_list
        self.label_list = label_list
        self.resample_rate = resample_rate
        self.imputation = imputation
        self.model_type = model_type
        self.model_type = model_type

        self._get_normalization_params(folder)

        if self.imputation == 'zero':
            static_mean = self.normalization_params['static_mean']
            self.static_fill = (torch.zeros_like(static_mean) - static_mean) / self.normalization_params['static_std']
            self.static_fill = self.static_fill.view((1,-1))

            data_mean = self.normalization_params['data_mean']
            self.data_fill = (torch.zeros_like(data_mean) - data_mean) / self.normalization_params['data_std']

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
        if self.resample_rate:
            x_data = pd.read_csv(os.path.join(file_path, 'data', self.sample_list[idx] + '.csv'), sep=';', header=0,
                                 index_col=0)
            x_data = resample_aligned_data(x_data, str(self.resample_rate)+'Min')
            x_len = int(144//(self.resample_rate/5))
            if x_data.shape[0] > x_len:
                x_data = x_data.iloc[:x_len, :]
            if x_data.shape[0] < x_len:
                print(x_data.shape[0])
            x_data = x_data.to_numpy()
        else:
            x_data = pd.read_csv(os.path.join(file_path, 'data', self.sample_list[idx] + '.csv'), sep=';', header=0,
                                 index_col=[0,1]).to_numpy()

        x_info = self._normalize_data(x_info, self.normalization_params['static_mean'],
                                      self.normalization_params['static_std'])
        x_data = self._normalize_data(x_data, self.normalization_params['data_mean'],
                                      self.normalization_params['data_std'])


        # if self.model_type.upper() == 'GRUD':
        #     x_data = torch.Tensor(x_data)
        #     mask = torch.isnan(x_data)
        #     delta_t = torch.zeros_like(x_data)
        #     x_last = x_data.detach().clone()
        #     for t in range(1, x_data.shape[0]):
        #         delta_t[t,:][mask[t-1,:]] = delta_t[t-1,:][mask[t-1,:]] + 1
        #         delta_t[t,:][~mask[t-1,:]] = 1
        #         x_last[t, :][mask[t, :]] = x_last[t-1,:][mask[t,:]]
        #         # tmp = x_last[t-1:t+1,:][mask[t,:].repeat(2,1)]
        #         # if (~torch.isnan(tmp)).sum()>0:
        #         #     print(x_last[t-1:t+1,:])
        #
        #     if self.imputation == 'zero':
        #         x_info[torch.isnan(x_info)] = self.static_fill[torch.isnan(x_info)]
        #         x_data[np.isnan(x_data)] = self.data_fill.repeat(x_data.shape[0], 1)[np.isnan(x_data)]
        #         x_last[np.isnan(x_last)] = self.data_fill.repeat(x_last.shape[0], 1)[np.isnan(x_last)]
        #
        #     elif self.imputation == 'mean':
        #         x_info[torch.isnan(x_info)] = 0
        #         x_data[torch.isnan(x_data)] = 0
        #         x_last[torch.isnan(x_last)] = 0
        #
        #     return {
        #         'data': {
        #             'static': x_info,
        #             'time_series': x_data,
        #             'mask': (~mask).float(),
        #             'delta_t': delta_t,
        #             'x_last': x_last,
        #         },
        #         'label': y
        #     }

        if self.imputation == 'zero':
            x_info[torch.isnan(x_info)] = self.static_fill[torch.isnan(x_info)]
            x_data[torch.isnan(x_data)] = self.data_fill.repeat(x_data.shape[0], 1)[torch.isnan(x_data)]

        elif self.imputation == 'mean':
            x_info[torch.isnan(x_info)] = 0
            x_data[torch.isnan(x_data)] = 0

        return {
            'data': {
                'static': x_info,
                'time_series': x_data,
            },
            'label': y
        }


    def _get_normalization_params(self, data_path):
        file_path = os.path.join(data_path, '..', 'norm_params.pt')
        self.normalization_params = torch.load(file_path)


    def _normalize_data(self, data, mean, std):
        try:
            x = (torch.Tensor(data) - mean) / std
        except:
            x = (torch.Tensor(data) - mean) / std
        return x

    def get_labels(self):
        return self.label_list

    def get_original_data(self, normalized_data, type):
        normalized_data = torch.Tensor(normalized_data)
        if type == 'time_series':
            mean = self.normalization_params['data_mean']
            std = self.normalization_params['data_std']
        elif type == 'static':
            mean = self.normalization_params['static_mean']
            std = self.normalization_params['static_std']

        else:
            assert ValueError('Currently only support time_series or static data type.')

        data = normalized_data * std + mean

        return data






if __name__=='__main__':
    # ##################################################################################################
    # # Calculate Normalization Parameters
    # ##################################################################################################
    # # root_folder = os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_0_12_3src')
    # # data_folder = os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_0_12_3src/ts_aligned')
    # root_folder = os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_6_12_3src')
    # data_folder = os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_6_12_3src/ts_aligned')
    # # root_folder = os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_12_12_3src')
    # # data_folder = os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_12_12_3src/ts_aligned')
    # X_train, X_test, y_train, y_test = np.load(os.path.join(root_folder, 'data_split_8020.npy'), allow_pickle=True)
    # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=.5, random_state=0, shuffle=True,
    #                                                 stratify=y_test)
    # X_all = X_train + X_val
    # y_all = y_train + y_val
    #
    # ds = PatientDataset(data_folder, X_all, y_all, imputation=None)
    # ds_loader = DataLoader(ds, 1000, num_workers=6)
    #
    # mean = torch.zeros(25)
    # std = torch.zeros(25)
    # count = torch.zeros(25)
    #
    # mean_s = torch.zeros(4)
    # std_s = torch.zeros(4)
    # count_s = torch.zeros(4)
    #
    # for batch in tqdm(ds_loader):
    #     info = batch['data']['static']
    #     info = info.view(-1, 4)
    #     mask = info.isnan()
    #     info = torch.masked_fill(info, mask, 0)
    #     count_s += (~mask).sum(dim=0)
    #     mean_s += info.sum(dim=0)
    #
    #     data = batch['data']['time_series']
    #     data = data.view(-1, 25)
    #     mask = data.isnan()
    #     data = torch.masked_fill(data, mask, 0)
    #     count += (~mask).sum(dim=0)
    #     mean += data.sum(dim=0)
    # mean_s = mean_s / count_s
    # mean = mean / count
    #
    # for batch in tqdm(ds_loader):
    #     info = batch['data']['static']
    #     info = info.view(-1, 4)
    #     mask = info.isnan()
    #     info = torch.masked_fill(info, mask, 0)
    #     std_s += (info - mean_s).pow(2).sum(dim=0)
    #
    #     data = batch['data']['time_series']
    #     data = data.view(-1, 25)
    #     mask = data.isnan()
    #     data = torch.masked_fill(data, mask, 0)
    #     std += (data - mean).pow(2).sum(dim=0)
    # std_s = torch.sqrt(std_s / count_s)
    # std = torch.sqrt(std / count)
    #
    # dataset_norm_params = {
    #     'static_mean': mean_s,
    #     'static_std': std_s,
    #     'data_mean': mean,
    #     'data_std': std,
    # }
    #
    # # save_path = os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_0_12_3src', 'norm_params_0_12.pt')
    # save_path = os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_6_12_3src', 'norm_params_6_12.pt')
    # # save_path = os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_12_12_3src', 'norm_params_12_12.pt')
    # torch.save(dataset_norm_params, save_path)
    #
    # norm_params = torch.load(save_path)
    # ##################################################################################################




    ##################################################################################################
    # Check data
    ##################################################################################################
    pred = 12

    root_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src')
    data_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src/ts_aligned')
    data_split = np.load(os.path.join(root_folder, 'data_split.npy'), allow_pickle=True).item()
    X_train = data_split['X_train']
    y_train = data_split['y_train']
    X_val = data_split['X_val']
    y_val = data_split['y_val']
    X_test = data_split['X_test']
    y_test = data_split['y_test']

    file_path = os.path.join(root_folder, 'norm_params.pt')
    normalization_params = torch.load(file_path)

    ds = PatientDataset(data_folder, X_train, y_train, model_type='CNN')
    ds_grud = PatientDataset(data_folder, X_train, y_train, model_type='GRUD')
    ds_loader = DataLoader(ds, 1, num_workers=6)

    for i in range(len(ds_grud)):
        sample = ds_grud[i]
        info = sample['data']['static']
        data = sample['data']['time_series']
        label = sample['label']

        label_ = y_test[i]
        if label_ == 0:
            file_path = os.path.join(data_folder, 'neg')
        else:
            file_path = os.path.join(data_folder, 'pos')
        info_ = pd.read_csv(os.path.join(file_path, 'info', X_train[i]+'.csv'), sep=';', header=0, index_col=0).to_numpy()
        data_ = pd.read_csv(os.path.join(file_path, 'data', X_train[i] + '.csv'), sep=';', header=0, index_col=[0, 1]).to_numpy()

        info_normalized = (torch.Tensor(info_) - normalization_params['static_mean']) / normalization_params['static_std']
        data_normalized = (torch.Tensor(data_) - normalization_params['data_mean']) / normalization_params['data_std']


        print(i)






    # loader_test1 = DataLoader(ds, batch_size=32, sampler=ImbalancedDatasetSampler(ds_test))
    # loader_test2 = DataLoader(ds, batch_size=32)
    # for i, (batch1, batch2) in enumerate(zip(loader_test1, loader_test2)):
    #     data1 = batch1['data']
    #     label1 = batch1['label']
    #     data2 = batch2['data']
    #     label2 = batch2['label']
    #
    #     print(sum(label1), sum(label2))


