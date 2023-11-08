import os
os.chdir('./..')

from src.test_torch_lstm import *
from src.test_torch_cnn import *

import shap
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled=False


root_folder = os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_0_12_3src')
data_folder = os.path.join(os.path.abspath('.'), 'processed/patient_data/dx_pred_0_12_3src/ts_aligned')
X_train, X_test, y_train, y_test = np.load(os.path.join(root_folder, 'data_split_8020.npy'), allow_pickle=True)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=.5, random_state=0, shuffle=True,
                                                stratify=y_test)

# config = {
#         "n_input_static": 4,
#         "n_input_seq": 25,
#         "n_classes": 1,
#         "n_hidden": 64,
#         "n_layers": 3,
#         "hidden_dense": 32,
#         "lr": 1e-3,
#         "batch_size": 64,
#         "dropout": 0.75,
#         "dropout_dense": 0.5,
#     }
# model = FullNet.load_from_checkpoint(checkpoint_path='models/LSTM_torch/epoch112-val_roc0.91.ckpt', config=config, data_folder=data_folder)
# model.to(device='cuda')
#
#
# # trainer = Trainer(gpus=1)
# # trainer.test(model);
#
#
# X, _, y, _ = train_test_split(X_test, y_test, train_size=.02, random_state=0, shuffle=True, stratify=y_test)
#
# ds_shap = PatientDataset(data_folder, X, y)
# loader_shap = DataLoader(ds_shap, batch_size=32)
#
# loader_shap = DataLoader(ds_shap, batch_size=ds_shap.__len__())
# data = next(iter(loader_shap))
#
# for d in data['data']:
#     data['data'][d] = data['data'][d].to(device='cuda')
# y = data['label'].to(device='cuda')
#
#
# # model.train()
# e = shap.DeepExplainer(
#     model,
#     [data['data']['static'], data['data']['time_series']],
# )
#
# shap_values = e.shap_values(
#     [data['data']['static'], data['data']['time_series']]
# )
#
# feature_names = [
#     'RDW', 'RESPIRATION', 'NIBP_m', 'NA', 'MCHC', 'MG', 'URINE',
#     'NIBP_s', 'HBG', 'HCO3', 'UREA', 'MCH', 'SAO2', 'MCV', 'GLC',
#     'PLT', 'HR', 'MPV', 'CL', 'K', 'CA', 'RBC', 'CR', 'NIBP_d', 'WBC'
# ]
#
# # shap.summary_plot(shap_values, features=data['data']['time_series'], feature_names=np.array(feature_names), max_display=50,)
#
#
# data_ts = data['data']['time_series']
# data_mask = data_ts!=0
# data_mask_ = data_mask.sum(dim=1) == 0
# mask_sum = data_mask.sum(dim=1)
# mask_sum[data_mask_] = 1
# data_ts_mean = (data_ts * data_mask).sum(dim=1) / mask_sum
#
# shap_ts_mean = shap_values[1].mean(axis=1)
#
# shap.summary_plot(shap_ts_mean, features=data_ts_mean.cpu().detach().numpy(), feature_names=np.array(feature_names),)
#
#
# import seaborn as sns
#
# xlabels = [str(xlabel) for xlabel in np.arange(0, 144)]
# feat_scores = shap_values[1][300,:,:].T
# fig, ax = plt.subplots(figsize=(10,6), dpi=500)
# ax = sns.heatmap(feat_scores, yticklabels=feature_names, xticklabels=xlabels,
#                 cmap='vlag')
# plt.setp(ax.get_xticklabels(), rotation=0, ha="right",rotation_mode="anchor");
# plt.locator_params(axis='x', nbins=18)
# plt.show()



n_input_static = 4
n_input_seq = 25
n_filters_per_channel = 3

config = {
        "n_input_static": n_input_static,
        "n_input_seq": n_input_seq,
        "n_classes": 1,
        "cnn_kernel_size": 8,
        "cnn_stride": 4,
        "n_group_cnn": n_input_seq,
        "n_hidden": n_filters_per_channel*n_input_seq,
        "pool_kernel_size": 8,
        "dropout_dense": 0.5,
        "lr": 1e-3,
        "batch_size": 64,
    }
model = CNN_CLF.load_from_checkpoint(checkpoint_path='models/Conv1D_torch/epoch75-val_roc0.87.ckpt',
                                     config=config, data_folder=data_folder)


model.to(device='cpu')

X, _, y, _ = train_test_split(X_test, y_test, train_size=.02, random_state=0, shuffle=True, stratify=y_test)

ds_shap = PatientDataset(data_folder, X, y)
loader_shap = DataLoader(ds_shap, batch_size=32)

loader_shap = DataLoader(ds_shap, batch_size=ds_shap.__len__())
data = next(iter(loader_shap))

for d in data['data']:
    data['data'][d] = data['data'][d].to(device='cpu')
y = data['label'].to(device='cpu')

# model.train()
e = shap.DeepExplainer(
    model,
    [data['data']['static'], data['data']['time_series']],
)

shap_values = e.shap_values(
    [data['data']['static'], data['data']['time_series']]
)

feature_names = [
    'RDW', 'RESPIRATION', 'NIBP_m', 'NA', 'MCHC', 'MG', 'URINE',
    'NIBP_s', 'HBG', 'HCO3', 'UREA', 'MCH', 'SAO2', 'MCV', 'GLC',
    'PLT', 'HR', 'MPV', 'CL', 'K', 'CA', 'RBC', 'CR', 'NIBP_d', 'WBC'
]

data_ts = data['data']['time_series']
data_mask = data_ts!=0
data_mask_ = data_mask.sum(dim=1) == 0
mask_sum = data_mask.sum(dim=1)
mask_sum[data_mask_] = 1
data_ts_mean = (data_ts * data_mask).sum(dim=1) / mask_sum

shap_ts_mean = shap_values[1].mean(axis=1)