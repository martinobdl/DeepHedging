import torch
from path_generator import BS_Generator, DscGenerator
from torch.utils.data import DataLoader
import torch.nn as nn
import pytorch_lightning as pl
from flags import FLAGS
import dataset
import numpy as np
import pnl_loss
import ptf_model


def contract(x):
    return np.maximum(x[1, -1] - FLAGS.SPOT, 0).astype(np.float32)


class VanillaModel(nn.Module):
    def __init__(self):
        super(VanillaModel, self).__init__()

        self.layer_dim = 2
        inner_dim = 16

        self.net = nn.Sequential(
                                nn.Linear(self.layer_dim+1, inner_dim),
                                nn.ReLU(),
                                nn.Linear(inner_dim, inner_dim),
                                nn.ReLU(),
                                nn.Linear(inner_dim, inner_dim),
                                nn.ReLU(),
                                nn.Linear(inner_dim, self.layer_dim-1),
                                nn.Sigmoid())

    def forward(self, x):
        out = torch.zeros_like(x[:, 1:, :])
        T = x.shape[2]
        for i in range(x.shape[2]-1):
            t = (T-i)*torch.ones_like(x[:, 0:1, i])
            inpt = torch.cat((x[:, :, i], t), dim=1)
            w1 = self.net(inpt)
            out[:, :, i] = w1
        return out


class LitModel(pl.LightningModule):

    def __init__(self, model, generators, ptf_function, loss):
        super().__init__()
        self.model = model
        self.loss = loss
        self.ptf_function = ptf_function
        assert len(generators) == 2
        self.generatorDsc = generators[0]
        self.generatorStock = generators[1]
        self.pnl_function = pnl_loss.loss_PnL()

    def forward(self, x):
        return self.ptf_function(x, self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        PnL = -self.pnl_function(y_hat, y)
        tensorboard_logs = {'train_loss': loss, 'train_PnL': PnL}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        PnL = -self.pnl_function(y_hat, y)
        return {'val_loss': loss, 'val_PnL': PnL}

    def train_dataloader(self):

        return DataLoader(dataset.AssetsDataset(self.generatorStock, self.generatorDsc, contract),
                          batch_size=FLAGS.BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs])
        avg_PnL = torch.stack([x['val_PnL'] for x in outputs])
        self.logger.experiment.add_histogram('PnL', values=avg_PnL, global_step=self.trainer.global_step)
        tensorboard_logs = {'val_loss': avg_loss.mean(), 'val_PnL': avg_PnL.mean()}
        return {'val_loss': avg_loss.mean(), 'log': tensorboard_logs}

    def val_dataloader(self):

        return DataLoader(dataset.AssetsDataset(self.generatorStock, self.generatorDsc, contract),
                          batch_size=512,
                          shuffle=False,
                          num_workers=4)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.005)


if __name__ == "__main__":

    generatorStock = BS_Generator(n_paths=FLAGS.SAMPLES)
    generatorDsc = DscGenerator(n_paths=FLAGS.SAMPLES)

    df = dataset.AssetsDataset(generatorStock, generatorDsc, contract)
    dl = DataLoader(df,
                    batch_size=3,
                    shuffle=True,
                    num_workers=0)
    x, y = next(iter(dl))
    model = VanillaModel()
    y_hat = model(x)
    y_hat = ptf_model.ptf_function(x, y_hat)
    assert torch.mean(torch.sum(y_hat[:, :, 1:] * x[:, :, 1:], dim=1)
                      - torch.sum(y_hat[:, :, :-1] * x[:, :, 1:], dim=1)) < 1e-7
    # assert torch.mean((y_hat[:, 0, -1]*x[:, 0, -1] - ptf_value(x, y_hat)[:, -1])**2) < 1e-7
