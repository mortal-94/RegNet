import torch
import torchvision
import typing
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from conv_rnns import ConvGRUCell, ConvLSTMCell

from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer


learning_rate = 3e-4
max_epochs = 10
batch_size = 32

class rnn_regulated_block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, rnn_cell, identity_block=None, stride=1):
        super(rnn_regulated_block, self).__init__()
        #print(f'In channels {in_channels} | Intermediate channels: {intermediate_channels} ')
        self.identity_block = identity_block
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU()

        self.rnn_cell = rnn_cell
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)

        self.conv3 = nn.LazyConv2d(intermediate_channels, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm2d(intermediate_channels)

        self.conv4 = nn.Conv2d(intermediate_channels, intermediate_channels * 4, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm2d(intermediate_channels * 4)
    def forward(self, x:torch.Tensor, state:typing.Tuple) -> typing.Tuple:
        y = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if isinstance(self.rnn_cell, ConvLSTMCell):
            c, h = self.rnn_cell(x, state)
        else:
            c = None; h = self.rnn_cell(x, state[1])

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #print(f'Block running. x.shape : {x.shape}, h shape: {h.shape}')
        x = torch.cat([x, h], dim=1)


        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)


        x = self.conv4(x)
        x = self.bn4(x)

        if self.identity_block is not None:
            x += self.identity_block(y)
        return c, h, self.relu(x)

class RegNet(pl.LightningModule):
    def __init__(self, in_dim:int, classes:int=3, cell_type:str='gru', layers:typing.List=[3, 4, 6, 3]):
        super(RegNet, self).__init__()
        self.layers = layers
        self.classes = classes
        self.intermediate_channels = 64
        self.conv1 = nn.Conv2d(in_dim, self.intermediate_channels, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.intermediate_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d((3, 3) , padding=1, stride=2)
        self.cell = ConvGRUCell if cell_type == 'gru' else ConvLSTMCell
        regulated_blocks = []
        for layer in range(len(layers)):
            stride = 1 if layer < 1 else 2
            channels = self.intermediate_channels if layer < 1 else self.intermediate_channels // 2
            h_channels = 64
            identity_block = None
            if layer > 1:
                identity_block = nn.Sequential(
                    nn.Conv2d(self.intermediate_channels, channels * 4, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(channels * 4)
                )

            regulated_blocks.append(rnn_regulated_block(
                self.intermediate_channels, channels,
                self.cell(channels,h_channels , kernel_size=3),
                identity_block, stride
            ))

            for block in range(layers[layer] - 1):
                self.intermediate_channels = channels * 4 if block < 1 else self.intermediate_channels
                conv_lstm = self.cell(channels, h_channels , kernel_size=3)
                regulated_blocks.append(rnn_regulated_block(
                        self.intermediate_channels, channels, conv_lstm
                    )
                )

        self.state_avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.regulated_blocks = nn.ModuleList(regulated_blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.output = nn.Linear(2048, classes)

    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        c, h = torch.zeros(x.shape), torch.zeros(x.shape)
        for i, block in enumerate(self.regulated_blocks):
            #print(f'Block: {i}, x.shape: {x.shape}, h.shape {h.shape}')
            c, h, x = block(x, (c, h))
            if h.shape[-1] != x.shape[-1]:
                h = self.state_avg_pool(h)
                if c is not None:
                    c = self.state_avg_pool(c)

        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.output(x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=learning_rate)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return { "loss" : loss }

    def train_dataloader(self):
        train_ds = torchvision.datasets.CIFAR10(
            './dataset', True, transform=torchvision.transforms.ToTensor(), download=True)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
        return train_dl

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return { "val_loss" : loss }

    def val_dataloader(self):
        val_ds = torchvision.datasets.CIFAR10(
            './dataset', False, transform=torchvision.transforms.ToTensor(), download=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)
        return val_dl


if __name__  == "__main__":
    model = RegNet(3, 10, 'lstm')
    output = model(torch.randn(4, 3, 224, 224))
    #print(output.shape)
    trainer =  Trainer(fast_dev_run=True)
    trainer.fit(model)


