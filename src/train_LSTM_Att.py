import os
from pathlib import Path
os.chdir(os.path.join(Path(__file__).parent.resolve(), ".."))

import sys
sys.path.append(os.path.dirname(os.path.realpath(Path(__file__).parent.resolve())))

import time
import argparse

from src.create_dataset_torch import *

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


class LSTM_ATT(LightningModule):
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

        self.lstm_layer = nn.LSTM(
            input_size=config['n_input_seq'],
            hidden_size=config['n_hidden'],
            num_layers=config['n_layers'],
            dropout=config['dropout'],
            batch_first=True,
        )

        self.attention_layer = nn.MultiheadAttention(
            embed_dim=config['n_input_seq'],
            num_heads=config['n_input_seq'],
            batch_first=True,
        )

        self.output_layer = nn.Linear(config['n_input_static'] + config['n_hidden'], config['n_classes'])

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
        x_seq = x_seq.float()

        x_seq, _ = self.attention_layer(x_seq, x_seq, x_seq)
        out, (h_n, c_n) = self.lstm_layer(x_seq)
        x_seq = out[:, -1, :]
        x_seq = x_seq.reshape(x_seq.size(0), -1)
        x_seq = F.relu(x_seq)

        x_static = x_static.view(x_static.size(0), -1)
        x_ = torch.cat((x_static, x_seq), dim=1)

        out = self.output_layer(x_)
        out = self.dropout_layer(out)
        out = torch.sigmoid(out)

        return out

    def configure_optimizers(self):
        adam = optim.Adam(self.parameters(), lr=self.config['lr'])
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, mode='min', factor=.5, patience=5)
        return [adam], {"scheduler": lr_scheduler, "monitor": "train_loss"}
        # return adam

    def training_epoch_end(self, outputs):
        lr_sch = self.lr_schedulers()
        lr_sch.step(self.trainer.callback_metrics["train_loss"])

    def training_step(self, batch, batch_idx):
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
        loader_train = DataLoader(ds_train, batch_size=self.config['batch_size'], sampler=ImbalancedDatasetSampler(ds_train), num_workers=8)
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

                    n_layer = 2
                    n_hidden = 64
                    n_hidden_dense = 32

                    config = {
                        "n_input_static": 4,
                        "n_input_seq": 33,
                        "n_classes": 1,
                        "n_hidden": n_hidden,
                        "n_layers": n_layer,
                        "hidden_dense": n_hidden_dense,
                        "lr": 1e-3,
                        "batch_size": 64,
                        "dropout": 0,
                        "dropout_dense": 0.5,
                        'resample_rate': resample_rate,
                        'imputation': imputation,
                    }

                    model_type = 'LSTMATT_torch'
                    model_name = f"{n_layer}layer"
                    model_save_path = f'models/{model_type}/{model_name}'
                    Path(model_save_path).mkdir(parents=True, exist_ok=True)
                    model_version = os.listdir(model_save_path)
                    version = 0
                    versions = [int(v.split('_')[1]) for v in model_version if 'version_' in v]
                    if len(versions) > 0:
                        version = sorted(versions)[-1] + 1

                    modelfilename = f'pred{pred}-imp{imputation}-sampling{resample_rate}-'
                    callbacks = [
                        ModelCheckpoint(
                            monitor='val_roc',
                            mode='max',
                            save_top_k=1,
                            dirpath=f'models/{model_type}/{model_name}/version_{version}',
                            filename=modelfilename + 'epoch{epoch:02d}-val_roc{val_roc:.2f}',
                            auto_insert_metric_name=False
                        ),
                        EarlyStopping(
                            monitor='val_loss',
                            mode='min',
                            patience=30,
                        )
                    ]

                    model = LSTM_ATT(config, data_folder, data_split)

                    logger = TensorBoardLogger(
                        f"models/{model_type}",
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

                    with open(f'models/{model_type}/{model_name}/train_time.txt', 'a') as train_time_file:
                        train_time_file.write(f'{modelfilename}: {train_time_total}\n')