"""
Author: Md Mostafizur Rahman
File: CNN design for the MNIST dataset 

"""

import os
import torch
from torchvision import models
from torchsummary import summary
from torch import nn
import torch.nn.functional as F

#project modules
from .. import config
from . import preprocess

model_checkpoint_dir = os.path.join(config.checkpoints_path(), "baseline.h5")
save_model_dir = os.path.join(config.output_path(), "baseline.h5")

def get_model():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
        
            self.conv_block = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2) 
            )
        
            self.linear_block = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(128*7*7, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, 10)
            )
        
        def forward(self, x):
            x = self.conv_block(x)
            x = x.view(x.size(0), -1)
            x = self.linear_block(x)
        
            return x
    return Net()
    




# def read_model():
#     model = load_model(save_model_dir)
#     return model
    
def save_model_checkpoints():
    return torch.save({
            'epoch': config.nb_epochs,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_checkpoint_dir)
    
# def set_early_stopping():
#     return EarlyStopping(monitor='val_loss', 
#                          patience=15, 
#                          verbose=2, 
#                          mode='auto'
#                          )

if __name__ == "__main__":
    conv_model = get_model()
    summary(conv_model, (1, 28, 28))
    

 
