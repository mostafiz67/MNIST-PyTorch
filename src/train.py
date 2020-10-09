"""
Author: Md Mostafziur Rahman
File: Traning a CNN architecture using the MNIST dataset
"""

import os
import torch
import numpy as np
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
import torchvision.datasets as datasets

# module packages
from .. import config
from . import preprocess, my_model

# Loding train Data
train_loader = preprocess.load_train_data()

# Loding Model
model = my_model.get_model()

# Define a Loss function and optimizer

optimizer = optim.Adam(params=model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()

# Train the network
for epoch in range(config.nb_epochs):
    running_loss = 0.0
    for batch_idx, (train_img, train_labels) in enumerate(train_loader):
        
        # get the inputs; data is a list of [inputs, labels]
        train_img = train_img.unsqueeze(1)
        train_img, train_labels = train_img, train_labels
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        output = model(train_img)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        # print every 2000 mini-batches
        if (batch_idx + 1)% 5000 == 4999:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch, batch_idx + 1, running_loss / 5000))
            running_loss = 0.0
print('Finished Training')


# Save Model 
model_checkpoint_dir = os.path.join(config.checkpoints_path(), "baseline.h5")
torch.save(model.state_dict(), model_checkpoint_dir)

# #Compile
# model.compile(keras.optimizers.Adam(config.lr),
#               keras.losses.categorical_crossentropy,
#               metrics = ['accuracy'])

# Check point
# model_cp = my_model.save_model_checkpoints()
# early_stopping = my_model.set_early_stopping()

# #model training
# model.fit(train_data, train_labels,
#           batch_size = config.batch_size,
#           epochs = config.nb_epochs,
#           verbose=2,
#           shuffle = True,
#           callbacks = [early_stopping, model_cp],
#           validation_split = 0.2)
