"""
Author: Md Mostafizur Rahman
File: Testing and submission Kaggle MNIST
"""

import os, torch
import pandas as pd
import numpy as np

# project modules
from .. import config
from . import preprocess, my_model

#Loading Model
model = my_model.get_model()
model.load_state_dict(torch.load(os.path.join(config.output_path(), "baseline.h5")))

#loading test data
test_data = preprocess.load_test_data()
print(len(test_data))

final_output = []
test_preds = torch.LongTensor()
for i, data in enumerate(test_data):
    data = data.unsqueeze(1)
    output = model(data)
    preds = output.cpu().data.max(1, keepdim=True)[1]
    test_preds = torch.cat((test_preds, preds), dim=0)

print((test_preds))

submission_df = pd.read_csv('Dropbox/MNIST-PT/submission/sample_submission.csv')
submission_df.head()
submission_df['Label'] = test_preds.numpy().squeeze()
submission_df.head()

