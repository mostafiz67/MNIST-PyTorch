"""
Author: Md Mostafizur Rahman
File: Preprocessing MNIST train and test datasets

"""

import os, cv2
import numpy as np
import pandas as pd 
import torch
import torchvision
import matplotlib.pyplot as plt


from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from .. import config




def load_train_data():
    
    # Loding CSV file in a dataframe with values
    train_img_df = pd.read_csv(os.path.join(config.dataset_path(), "mnist_train.csv"))
    
    #Getting image pixels values and labels from dataset 
    print("Getting image pixels values and labels from dataset! ...")
    train_images = (train_img_df.iloc[:, 1:].values).astype('float32')
    train_labels = train_img_df.iloc[:, 0].values
    
    #Reshape the images
    train_images = train_images.reshape(train_images.shape[0], 28, 28)
    
    #Ploting some traing images
    # for i in range(6, 9):
    #     plt.subplot(330 + (i+1))
    #     plt.imshow(train_images[i].squeeze(), cmap = plt.get_cmap('gray'))
    #     plt.title(train_labels[i])
    # plt.show()
    
    #Converting Images to tensor
    train_images_tensor = torch.tensor(train_images)/255.0
    train_labels_tensor = torch.tensor(train_labels)
    train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)
    
    #Train DataLoder Generator
    train_loader = DataLoader(train_tensor, batch_size= config.batch_size, 
                              num_workers = 2,shuffle= True)
    
    #Plot some sample images using the data generator
    #Commenting for testing
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     img_grid = make_grid(data[0:8,].unsqueeze(1), nrow=8)
    #     img_target_labels = target[0:8,].numpy()
    #     break

    # plt.imshow(img_grid.numpy().transpose((1,2,0)))
    # plt.rcParams['figure.figsize'] = (10, 2)
    # plt.title(img_target_labels, size=16)
    # plt.show()
    
    return train_loader

#Loadnig Test Data
def load_test_data():
    # Loding CSV file in a dataframe with values
    test_img_df = pd.read_csv(os.path.join(config.dataset_path(), "mnist_test.csv"))
    
    #Getting image pixels values and labels from dataset 
    print("Getting image pixels values from dataset! ...")
    test_images = (test_img_df.iloc[:,:].values).astype('float32')
    
    #Reshape the images
    test_images = test_images.reshape(test_images.shape[0], 28, 28)
    
    #Ploting some traing images
    # for i in range(6, 9):
    #     plt.subplot(330 + (i+1))
    #     plt.imshow(test_images[i].squeeze(), cmap = plt.get_cmap('gray'))
    # plt.show()
    
    #Converting Images to tensor
    test_tensor = torch.tensor(test_images)/255.0
    
    #Train DataLoder Generator
    test_loader = DataLoader(test_tensor, batch_size= config.batch_size, 
                              num_workers = 2,shuffle= True)
    
    print(len(test_loader))
    #Plot some sample images using the data generator
    # for data in test_loader:
    #     img_grid = make_grid(data[0:8,].unsqueeze(1), nrow=8)
    #     break

    # plt.imshow(img_grid.numpy().transpose((1,2,0)))
    # plt.rcParams['figure.figsize'] = (10, 2)
    # plt.show()
    
    return test_loader



    
if __name__ == "__main__":
    load_test_data()
