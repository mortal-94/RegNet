import torch
import torch.nn as nn
import torchmetrics as tm
import torch.nn.functional as F

import argparse

import pytorch_lightning as pl
from torch.functional import Tensor
from typing import Tuple, Dict, List
from conv_rnns import ConvGRUCell, ConvLSTMCell


from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from cifar100_datamodule import Cifar100DataModule

class SELayer(nn.Module):
    def __init__(self, in_dim:int, reduction_factor:int=8) -> None:
        super(SELayer, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sequential= nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction_factor, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // reduction_factor, in_dim, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x:Tensor):
        B, C, _, _ = x.shape
        y = self.global_avg_pool(x).view(B, C)
        y = self.sequential(y).view(B, C, 1, 1)
        x = x * y.expand_as(x)
        return x



class rnn_regulated_block(nn.Module):
    def __init__(self, h_dim, in_channels, intermediate_channels, rnn_cell, identity_block=None, stride=1):
        super(rnn_regulated_block, self).__init__()
        #print(f'In channels {in_channels} | Intermediate channels: {intermediate_channels} ')
        self.stride = stride
        self.h_dim = h_dim
        self.identity_block = identity_block
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU()

        self.rnn_cell = rnn_cell
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, 
                               kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)

        #Multiply intermediate_channels by 2, torch.cat([hidden_state, x])
        self.conv3 = nn.Conv2d(h_dim + intermediate_channels, intermediate_channels, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm2d(intermediate_channels)

        self.conv4 = nn.Conv2d(intermediate_channels, intermediate_channels * 4, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(intermediate_channels * 4)

        self.se_layer = SELayer(intermediate_channels * 4, reduction_factor=8)
        
        downsample_dim = h_dim if isinstance(rnn_cell, ConvGRUCell) else h_dim * 2
        #Cell state dim remains constant but aspect ratio of the feature map is variable
        self.downsample_state = nn.LazyConv2d(downsample_dim, kernel_size=3, stride=stride, padding=1)


    def forward(self, x:torch.Tensor, state:Tuple) -> Tuple:
        y = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        c, h = self.rnn_cell(x, state)
        
        #print(f'Block running {x.shape}')

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = torch.cat([x, h], dim=1)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x = self.se_layer(x)

        if self.identity_block is not None:
            y = self.identity_block(y)
            if c is not None:
                s = torch.cat([c, h], dim=1)
                s = self.downsample_state(s)
                c, h = torch.split(s, self.h_dim, dim=1)
            else:
                h = self.downsample_state(h)

        x += y

        return c, h, self.relu(x)


class RegNet(pl.LightningModule):
    def __init__(self, regulated_block:nn.Module, in_dim:int, h_dim:int, intermediate_channels:int,
                 classes:int=3, cell_type:str='gru', layers:List=[3, 3, 3], config=None):
        super(RegNet, self).__init__()
        self.layers = layers
        self.classes = classes
        self.intermediate_channels = intermediate_channels
        self.h_dim = h_dim
        self.cell_type = cell_type
        #self.conv1 = nn.Conv2d(in_dim, self.intermediate_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(in_dim, self.intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.intermediate_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d((3, 3) , padding=1, stride=2)
        self.cell = ConvGRUCell if cell_type == 'gru' else ConvLSTMCell

        self.rnn_cells = nn.ModuleList()
        self.regulated_blocks = nn.ModuleList()
        num_layers = len(layers)
        
        
        c_in = self.intermediate_channels
        
        for layer in range(num_layers):
            self.rnn_cells.append(self.cell(c_in, h_dim, kernel_size=3))
            c_in = c_in * 4 if layer == 0 else c_in * 2

        for layer in range(num_layers):
            stride = 2
            channels = self.intermediate_channels // 2

            if layer < 1:
                stride = 1
                channels = self.intermediate_channels


            identity_block = nn.Sequential(
                nn.Conv2d(self.intermediate_channels, channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * 4)
            )

            self.regulated_blocks.append(
                regulated_block(
                    self.h_dim, self.intermediate_channels, channels,
                    self.cell(channels, h_dim , kernel_size=3),
                    identity_block, stride
                )
            )

            self.intermediate_channels = channels * 4

            for block in range(layers[layer] - 1):
                self.regulated_blocks.append(
                    regulated_block(
                        self.h_dim, self.intermediate_channels, channels,
                        self.cell(channels, h_dim, kernel_size=3)
                    )
                )   
            
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.output = nn.LazyLinear(classes)

        self.val_accuracy = tm.Accuracy(task="multiclass", num_classes=self.classes)
        self.test_accuracy = tm.Accuracy(task="multiclass", num_classes=self.classes)
        self.train_accuracy = tm.Accuracy(task="multiclass", num_classes=self.classes)

        self.config = config
        self.validation_step_outputs = []
        
        self.save_hyperparameters()


    def forward(self, x) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.max_pool(x)
        B, _, H, W = x.shape
        
        h = torch.zeros(B, self.h_dim, H, W)
        c = torch.zeros(B, self.h_dim, H, W) if self.cell_type != 'gru' else None
        
        c, h = self.rnn_cells[0](x, (c, h))
        
        layer_idx = 0
        block_sum = 0
      
        for i, block in enumerate(self.regulated_blocks):
            c, h, x = block(x, (c, h))
            block_sum += 1
            if layer_idx < len(self.layers) - 1 and block_sum == self.layers[layer_idx]:
                #print(f'Block {i}, {x.shape}, {h.shape}, {block_sum}')
                c, h = self.rnn_cells[layer_idx + 1](x, (c, h))
                layer_idx += 1
                block_sum = 0

        x = self.avg_pool(x)
        x = self.flatten(x)
        
        return self.output(x)


    def configure_optimizers(self):
        
        if self.config is not None:
            learning_rate = self.config['lr']
            weight_decay = self.config['weight_decay']

        optimizer= SGD(self.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=30)
        return { "optimizer": optimizer, "lr_scheduler": lr_scheduler,"monitor":  "val_accuracy"}


    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        outputs = torch.argmax(outputs, dim=-1)
        accuracy = self.train_accuracy(outputs, labels)
        return { "loss" : loss, "accuracy" : accuracy }


    def on_train_epoch_end(self):
        self.log('train_accuracy', self.train_accuracy, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        outputs = torch.argmax(outputs, dim=-1)
        accuracy = self.val_accuracy(outputs, labels)
        self.validation_step_outputs.append(loss)
        return { "val_loss" : loss }


    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('val_loss', avg_loss)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True)


    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        outputs = torch.argmax(outputs, dim=-1)
        accuracy = self.test_accuracy(outputs, labels)
        return { "test_loss" : loss }


    def test_epoch_end(self, outputs):
        self.log('test_accuracy', self.test_accuracy, prog_bar=True)
        
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


if __name__  == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='Number of epochs to train', type=int, default=30)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=128)
    parser.add_argument('--checkpoint', help='checkpoint path', type=str, default=None)
    

    args = parser.parse_args()

    cfm = Cifar100DataModule(batch_size=args.batch_size)

    model = None
    if args.checkpoint:
        model = RegNet.load_from_checkpoint(args.checkpoint)
    else:
        model = RegNet(rnn_regulated_block,
                       in_dim=3,
                       h_dim=16,
                       intermediate_channels=64,
                       classes=cfm.num_classes,
                       cell_type='lstm',
                       layers=[1, 1, 3]
                      )


    ### Log metric progression
    logger = TensorBoardLogger('logs', name='regnet_logs')

    ### Callbacks
    stop_early = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
    last_chkpt_path = 'checkpoints/regnet.ckpts'
    checkpoint = ModelCheckpoint(
        dirpath= last_chkpt_path, monitor='val_accuracy', mode='max',
        filename='{epoch}-{val_accuracy:.2f}', verbose=True, save_top_k=1
    )


    trainer = Trainer(
        accelerator="auto", fast_dev_run=False, logger=logger,
        max_epochs=args.epochs, callbacks=[checkpoint],
    )

    trainer.fit(model, cfm)


