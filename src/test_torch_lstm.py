import os
from pathlib import Path
os.chdir(os.path.join(Path(__file__).parent.resolve(), ".."))

import torch

from src.create_dataset_torch import *
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

torch.manual_seed(111)

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split, StratifiedKFold

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler



class FullNet(LightningModule):
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

        # self.dense_layer = nn.Linear(config['n_input_static']+config['n_hidden'], config['hidden_dense'])
        # self.output_layer = nn.Linear(config['hidden_dense'], config['n_classes'])
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
        x_sequence = x_seq.float()

        out, (h_n, c_n) = self.lstm_layer(x_sequence)
        x_seq = out[:, -1, :]
        x_seq = x_seq.reshape(x_seq.size(0), -1)
        x_seq = F.relu(x_seq)

        x_static = x_static.view(x_static.size(0), -1)
        x_ = torch.cat((x_static, x_seq), dim=1)

        # out = self.dense_layer(x_)
        # out = self.dropout_layer(out)
        # out = F.relu(out)
        #
        # out = self.output_layer(out)
        # out = self.dropout_layer(out)
        out = self.output_layer(x_)
        out =torch.sigmoid(out)
        return out


    def configure_optimizers(self):
        adam = optim.Adam(self.parameters(), lr=self.config['lr'])
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
        loader_train = DataLoader(ds_train, batch_size=self.config['batch_size'], sampler=ImbalancedDatasetSampler(ds_train), num_workers=6)
        return loader_train

    def test_dataloader(self):
        ds_test = PatientDataset(self.data_folder, self.X_test, self.y_test,
                                 resample_rate=self.config['resample_rate'], imputation=self.config['imputation'])
        loader_test = DataLoader(ds_test, batch_size=self.config['batch_size'], num_workers=6)
        return loader_test

    def val_dataloader(self):
        ds_val = PatientDataset(self.data_folder, self.X_val, self.y_val,
                                resample_rate=self.config['resample_rate'], imputation=self.config['imputation'])
        loader_val = DataLoader(ds_val, batch_size=self.config['batch_size'], num_workers=6)
        return loader_val


def train_model(config, data_folder=None, max_epochs=100, num_gpus=1, checkpoint_dir=None):
    model = FullNet(config, data_folder)

    metrics = {"loss": "val_loss",
               "acc": "val_acc",
               "recall": "val_recall",
               "precision": "val_precision",
               "roc": "val_roc",
               "f1": "val_f1",
               }

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/home/dhm/workspace/conference_projects/MICCAI2022_AKF_feature_importance/models/LSTM_torch',
        filename='epoch{epoch:02d}-val_loss{val_loss:.2f}',
        auto_insert_metric_name=False
    )

    callbacks = [
        checkpoint_callback,
        TuneReportCallback(metrics, on="validation_end"),
        TuneReportCheckpointCallback(
            metrics={
                "loss": "val_loss",
                "acc": "val_acc"
            },
            filename="checkpoint",
            on="validation_end"),
    ]

    trainer = Trainer(
        max_epochs=max_epochs,
        gpus=num_gpus,
        progress_bar_refresh_rate=0,
        callbacks=callbacks,
        # resume_from_checkpoint=os.path.join(checkpoint_dir, "checkpoint"),
    )
    trainer.fit(model)


if __name__=='__main__':
    n_layer = 2
    n_hidden = 64
    n_hidden_dense = 32
    resample_rate = 60
    imputation = 'zero'

    for pred in [12]:
        root_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src')
        data_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src/ts_aligned')
        data_split = np.load(os.path.join(root_folder, 'data_split.npy'), allow_pickle=True).item()


        config = {
            "n_input_static": 4,
            "n_input_seq": 33,
            "n_classes": 1,
            "n_hidden": n_hidden,
            "n_layers": n_layer,
            "hidden_dense": n_hidden_dense,
            "lr": 1e-3,
            "batch_size": 64,
            "dropout": 0.75,
            "dropout_dense": 0.5,
            'resample_rate': resample_rate,
            'imputation': imputation,
        }

        model_type = 'LSTM_torch'
        model_name = f"{n_layer}layer"
        model_save_path = f'/home/dhm/workspace/conference_projects/MICCAI2022_AKF_feature_importance/models/{model_type}/{model_name}'
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
                dirpath=f'/home/dhm/workspace/conference_projects/MICCAI2022_AKF_feature_importance/models/{model_type}/{model_name}/version_{version}',
                filename=modelfilename+'epoch{epoch:02d}-val_roc{val_roc:.2f}',
                auto_insert_metric_name=False
            ),
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=30,
            )
        ]

        model = FullNet(config, data_folder, data_split)

        logger = TensorBoardLogger(
            f"/home/dhm/workspace/conference_projects/MICCAI2022_AKF_feature_importance/models/{model_type}",
            name=model_name,
            version=version)

        trainer = Trainer(
            max_epochs=250,
            gpus=1,
            callbacks=callbacks,
            logger=logger,
            # resume_from_checkpoint=os.path.join(checkpoint_dir, "checkpoint"),
        )
        trainer.fit(model)


    ###############################################################################################################
    # Ray Tune
    ###############################################################################################################

    # checkpoint_dir = '/home/dhm/workspace/conference_projects/MICCAI2022_AKF_feature_importance/models/LSTM_torch'
    # max_epochs = 50
    #
    # config = {
    #     "n_input_static": 4,
    #     "n_input_seq": 25,
    #     "n_classes": 1,
    #     "n_hidden": tune.grid_search([16, 32, 64]),
    #     "n_layers": tune.grid_search([1, 2, 3]),
    #     "lr": tune.grid_search([1e-4, 1e-3]),
    #     "batch_size": tune.grid_search([32, 64, 128]),
    #     "dropout": tune.grid_search([0.5, 0.75]),
    # }
    #
    # scheduler = ASHAScheduler(
    #     max_t=max_epochs,
    #     grace_period=1,
    #     reduction_factor=2)
    #
    # trainable = tune.with_parameters(
    #     train_model,
    #     data_folder=data_folder,
    #     max_epochs=max_epochs,
    #     num_gpus=1,
    #     # checkpoint_dir=checkpoint_dir,
    # )
    #
    # analysis = tune.run(
    #     trainable,
    #     resources_per_trial={
    #         "cpu": 6,
    #         "gpu": 1
    #     },
    #     metric="loss",
    #     mode="min",
    #     config=config,
    #     scheduler=scheduler,
    #     name="tune_lstm"
    # )
    #
    # print(analysis.best_config)

    # best_trial = analysis.best_trial  # Get best trial
    # best_config = analysis.best_config  # Get best trial's hyperparameters
    # best_logdir = analysis.best_logdir  # Get best trial's logdir
    # best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
    # best_result = analysis.best_result  # Get best trial's last results

