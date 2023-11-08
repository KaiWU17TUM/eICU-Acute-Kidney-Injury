import os
from pathlib import Path
os.chdir(os.path.join(Path(__file__).parent.resolve(), ".."))

import sys
sys.path.append(os.path.dirname(os.path.realpath(Path(__file__).parent.resolve())))

import time
import argparse

from src.create_dataset_torch import *

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics


from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger




class CNN_CLF(LightningModule):
    def __init__(self, config, data_folder, data_split):
        super().__init__()

        self.config = config
        self.data_folder = data_folder
        self.X_train = data_split['X_train']
        self.y_train = data_split['y_train']
        self.X_val = data_split['X_val']
        self.y_val = data_split['y_val']
        self.X_test = data_split['X_test']
        self.y_test = data_split['y_test']

        self.cnn_layer = nn.Conv1d(
            in_channels=config['n_input_seq'],
            out_channels=config['n_hidden'],
            kernel_size=config['cnn_kernel_size'],
            stride=config['cnn_stride'],
            groups=config['n_group_cnn'],
        )

        self.cnn_layer_ = nn.Conv1d(
            in_channels=config['n_hidden'],
            out_channels=config['n_hidden'],
            kernel_size=config['cnn_kernel_size'],
            stride=config['cnn_stride'],
            groups=1,
        )

        self.max_pool = nn.Sequential(nn.MaxPool1d(
            kernel_size=config['pool_kernel_size'],
            ceil_mode=True,
        ))

        self.output_layer = nn.LazyLinear(config['n_classes'])

        self.dropout_layer = nn.Dropout(p=config['dropout_dense'])

        self.METRICS = {
            'acc': torchmetrics.Accuracy().to(device='cuda'),
            'recall': torchmetrics.Recall().to(device='cuda'),
            'precision': torchmetrics.Precision().to(device='cuda'),
            'roc': torchmetrics.AUROC(pos_label=1).to(device='cuda'),
            # 'prc': torchmetrics.PrecisionRecallCurve(pos_label=1).to(device='cuda'),
            'f1': torchmetrics.F1().to(device='cuda'),
        }

    def forward(self, x_static, x_seq):
        x_static = x_static.float()
        x_sequence = x_seq.float().permute(0,2,1)

        x_cnn = self.cnn_layer(x_sequence)
        x_cnn = self.cnn_layer_(x_cnn)
        x_cnn = self.max_pool(x_cnn)
        x_cnn = F.relu(x_cnn)
        x_cnn = x_cnn.reshape(x_cnn.shape[0], -1)

        x_static = x_static.view(x_static.size(0), -1)
        x_ = torch.cat((x_static, x_cnn), dim=1)
        x_ = self.dropout_layer(x_)

        out = self.output_layer(x_)
        out = torch.sigmoid(out)
        return out


    def configure_optimizers(self):
        adam = optim.Adam(self.parameters(), lr=self.config['lr'])
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(adam, T_0=5, T_mult=2, eta_min=1e-6,)
        # return [adam], [lr_scheduler]
        return adam

    def training_step(self, batch, batch_idx):
        # lr_sch = self.lr_schedulers()
        # lr_sch.step()

        x = batch['data']
        y = batch['label'].view(-1, 1)
        pred = self.forward(x['static'], x['time_series'])
        loss = F.binary_cross_entropy(pred, y.float())

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            self.log("train_" + metric, self.METRICS[metric](pred, y), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        y = batch['label'].view(-1, 1)
        pred = self.forward(x['static'], x['time_series'])
        loss = F.binary_cross_entropy(pred, y.float())

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["val_"+metric] = self.METRICS[metric](pred, y)
            self.log("val_"+metric, outputs["val_"+metric], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        y = batch['label'].view(-1, 1)
        pred = self.forward(x['static'], x['time_series'])
        loss = F.binary_cross_entropy(pred, y.float())

        outputs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            outputs["test_" + metric] = self.METRICS[metric](pred, y)
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def train_dataloader(self):
        ds_train = PatientDataset(self.data_folder, self.X_train, self.y_train,
                                  resample_rate=self.config['resample_rate'], imputation=self.config['imputation'])
        loader_train = DataLoader(ds_train, batch_size=self.config['batch_size'],
                                  sampler=ImbalancedDatasetSampler(ds_train), num_workers=8)
        return loader_train

    def test_dataloader(self):
        ds_test = PatientDataset(self.data_folder, self.X_test, self.y_test,
                                 resample_rate=self.config['resample_rate'], imputation=self.config['imputation'])
        loader_test = DataLoader(ds_test, batch_size=self.config['batch_size'], num_workers=8)
        return loader_test

    def val_dataloader(self):
        ds_val = PatientDataset(self.data_folder, self.X_val, self.y_val,
                                resample_rate=self.config['resample_rate'], imputation=self.config['imputation'])
        loader_val = DataLoader(ds_val, batch_size=self.config['batch_size'], num_workers=8)
        return loader_val




if __name__=='__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--pred",
        nargs="*",
        type=int,
        default=[6],
    )
    CLI.add_argument(
        "--imp",
        nargs="*",
        type=str,
        default=["zero"],
    )
    CLI.add_argument(
        "--rr",
        nargs="*",
        type=int,
        default=[5]
    )
    args = CLI.parse_args()

    pred_list = args.pred
    imp_list = args.imp
    rr_list = args.rr

    for seed in [222, 333]:
        torch.manual_seed(seed)
        for pred in pred_list:
            for imputation in imp_list:
                for resample_rate in rr_list:
                    # for pred in [12]:
                    #     for imputation in ['mean']:
                    #         for resample_rate in [5]:
                    print(f'##################### PRED {pred} - IMP {imputation} - SAMPLERATE {resample_rate} #####################')

                    if resample_rate == 5:
                        resample_rate = None

                    root_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src')
                    data_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src/ts_aligned')
                    data_split = np.load(os.path.join(root_folder, 'data_split.npy'), allow_pickle=True).item()

                    n_input_static = 4
                    n_input_seq = 33
                    n_filters_per_channel = 3

                    if not resample_rate:
                        cnn_kernel_size = 8
                        cnn_stride = 4
                        pool_kernel_size = 4
                    elif resample_rate == 30:
                        cnn_kernel_size = 2
                        cnn_stride = 2
                        pool_kernel_size = 3
                    elif resample_rate == 60:
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
                        "n_hidden": n_filters_per_channel*n_input_seq,
                        "pool_kernel_size": pool_kernel_size,
                        "dropout_dense": 0.5,
                        "lr": 1e-3,
                        "batch_size": 64,
                        'resample_rate': resample_rate,
                        'imputation': imputation,
                    }

                    model_name = "CNN_2layer"
                    Path(f'models/Conv1D_torch/{model_name}').mkdir(parents=True, exist_ok=True)
                    model_version = os.listdir(f'models/Conv1D_torch/{model_name}')
                    version = 0
                    versions = [int(v.split('_')[1]) for v in model_version if 'version_' in v]
                    if len(versions)>0:
                        version = sorted(versions)[-1] + 1


                    if resample_rate:
                        modelfilename = f'pred{pred}-imp{imputation}-sampling{resample_rate}-'
                    else:
                        modelfilename = f'pred{pred}-imp{imputation}-sampling5-'
                    callbacks = [
                        ModelCheckpoint(
                            monitor='val_roc',
                            mode='max',
                            save_top_k=1,
                            dirpath=f'models/Conv1D_torch/{model_name}/version_{version}',
                            filename=modelfilename+'epoch{epoch:02d}-val_roc{val_roc:.2f}',
                            auto_insert_metric_name=False
                        ),
                        EarlyStopping(
                            monitor='val_loss',
                            mode='min',
                            patience=20,
                        )
                    ]

                    model = CNN_CLF(config, data_folder, data_split)

                    logger = TensorBoardLogger("models/Conv1D_torch",
                                               name=model_name,
                                               version=version)

                    trainer = Trainer(
                        max_epochs=250,
                        gpus=1,
                        callbacks=callbacks,
                        logger=logger,
                        # resume_from_checkpoint=os.path.join(checkpoint_dir, "checkpoint"),
                    )
                    train_time_start = time.time()
                    trainer.fit(model)
                    train_time_total = time.time() - train_time_start

                    with open(f'models/Conv1D_torch/{model_name}/train_time.txt', 'a') as train_time_file:
                        train_time_file.write(f'{modelfilename}: {train_time_total}\n')