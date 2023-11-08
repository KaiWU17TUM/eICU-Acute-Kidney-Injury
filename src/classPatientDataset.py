import os
from pathlib import Path
os.chdir(os.path.join(Path(__file__).parent.resolve(), ".."))

import torch
from torch.utils.data import Dataset
import pandas as pd

def resample_aligned_data(sample_data, resample_freq='10Min'):
    sample_data['Timestamp'] = pd.to_datetime(sample_data['Timestamp'], unit='m')
    sample_data = sample_data.set_index('Timestamp')
    resampled_data = sample_data.resample(resample_freq).last()

    return resampled_data

class PatientDatasetSimple(Dataset):
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


class PatientDataset(Dataset):
    def __init__(self, folder, sample_list, label_list, resample_rate=None, imputation='mean'):
        self.folder = folder
        self.sample_list = sample_list
        self.label_list = label_list
        self.resample_rate = resample_rate
        self.imputation = imputation
        self._get_normalization_params(folder)

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

        x_info = torch.Tensor(x_info)
        x_data = torch.Tensor(x_data)
        if self.imputation == 'zero':
            x_info[torch.isnan(x_info)] = 0
            x_data[torch.isnan(x_data)] = 0

        elif self.imputation == 'mean':
            x_info[torch.isnan(x_info)] = self.normalization_params['static_mean'][torch.isnan(x_info)]
            x_data[torch.isnan(x_data)] = self.normalization_params['data_mean'][torch.isnan(x_data)]

        x_info = self._normalize_data(x_info, self.normalization_params['static_mean'],
                                      self.normalization_params['static_std'])
        x_data = self._normalize_data(x_data, self.normalization_params['data_mean'],
                                      self.normalization_params['data_std'])

        # if self.imputation == 'zero':
        #     x_info[torch.isnan(x_info)] = self.static_fill[torch.isnan(x_info)]
        #     x_data[torch.isnan(x_data)] = self.data_fill.repeat(x_data.shape[0], 1)[torch.isnan(x_data)]
        #
        # elif self.imputation == 'mean':
        #     x_info[torch.isnan(x_info)] = 0
        #     x_data[torch.isnan(x_data)] = 0

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



# class PatientDataset(Dataset):
#     def __init__(self, folder, sample_list, label_list, resample_rate=None, imputation='mean', model_type='otherThanGRUD'):
#         self.folder = folder
#         self.sample_list = sample_list
#         self.label_list = label_list
#         self.resample_rate = resample_rate
#         self.imputation = imputation
#         self.model_type = model_type
#
#         self._get_normalization_params(folder)
#
#         # if self.imputation == 'zero':
#         #     static_mean = self.normalization_params['static_mean']
#         #     self.static_fill = (torch.zeros_like(static_mean) - static_mean) / self.normalization_params['static_std']
#         #     self.static_fill = self.static_fill.view((1,-1))
#         #
#         #     data_mean = self.normalization_params['data_mean']
#         #     self.data_fill = (torch.zeros_like(data_mean) - data_mean) / self.normalization_params['data_std']
#
#     def __len__(self):
#         return len(self.label_list)
#
#     def __getitem__(self, idx):
#         y = self.label_list[idx]
#         if y == 0:
#             file_path = os.path.join(self.folder, 'neg')
#         else:
#             file_path = os.path.join(self.folder, 'pos')
#
#         x_info = pd.read_csv(os.path.join(file_path, 'info', self.sample_list[idx] + '.csv'), sep=';', header=0,
#                              index_col=0).to_numpy()
#         if self.resample_rate:
#             x_data = pd.read_csv(os.path.join(file_path, 'data', self.sample_list[idx] + '.csv'), sep=';', header=0,
#                                  index_col=0)
#             x_data = resample_aligned_data(x_data, str(self.resample_rate)+'Min')
#             x_len = int(144//(self.resample_rate/5))
#             if x_data.shape[0] > x_len:
#                 x_data = x_data.iloc[:x_len, :]
#             if x_data.shape[0] < x_len:
#                 print(x_data.shape[0])
#             x_data = x_data.to_numpy()
#         else:
#             x_data = pd.read_csv(os.path.join(file_path, 'data', self.sample_list[idx] + '.csv'), sep=';', header=0,
#                                  index_col=[0,1]).to_numpy()
#
#         x_info = torch.Tensor(x_info)
#         x_data = torch.Tensor(x_data)
#         if self.imputation == 'zero':
#             x_info[torch.isnan(x_info)] = 0
#             x_data[torch.isnan(x_data)] = 0
#
#         elif self.imputation == 'mean':
#             x_info[torch.isnan(x_info)] = self.normalization_params['static_mean'][torch.isnan(x_info)]
#             x_data[torch.isnan(x_data)] = self.normalization_params['data_mean'][torch.isnan(x_data)]
#
#         x_info = self._normalize_data(x_info, self.normalization_params['static_mean'],
#                                       self.normalization_params['static_std'])
#         x_data = self._normalize_data(x_data, self.normalization_params['data_mean'],
#                                       self.normalization_params['data_std'])
#
#         # if self.imputation == 'zero':
#         #     x_info[torch.isnan(x_info)] = self.static_fill[torch.isnan(x_info)]
#         #     x_data[torch.isnan(x_data)] = self.data_fill.repeat(x_data.shape[0], 1)[torch.isnan(x_data)]
#         #
#         # elif self.imputation == 'mean':
#         #     x_info[torch.isnan(x_info)] = 0
#         #     x_data[torch.isnan(x_data)] = 0
#
#         return {
#             'data': {
#                 'static': x_info,
#                 'time_series': x_data,
#             },
#             'label': y
#         }
#
#
#     def _get_normalization_params(self, data_path):
#         file_path = os.path.join(data_path, '..', 'norm_params.pt')
#         self.normalization_params = torch.load(file_path)
#
#
#     def _normalize_data(self, data, mean, std):
#         try:
#             x = (torch.Tensor(data) - mean) / std
#         except:
#             x = (torch.Tensor(data) - mean) / std
#         return x
#
#     def get_labels(self):
#         return self.label_list
#
#     def get_original_data(self, normalized_data, type):
#         normalized_data = torch.Tensor(normalized_data)
#         if type == 'time_series':
#             mean = self.normalization_params['data_mean']
#             std = self.normalization_params['data_std']
#         elif type == 'static':
#             mean = self.normalization_params['static_mean']
#             std = self.normalization_params['static_std']
#
#         else:
#             assert ValueError('Currently only support time_series or static data type.')
#
#         data = normalized_data * std + mean
#
#         return data