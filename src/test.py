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
for i, data in enumerate(test_data):
    data = data.unsqueeze(1)
    output = model(data)
    
    final_output.append(output)
    # preds = final_output.cpu().data.max(1, keepdim=True)[1]
result = torch.cat(final_output, dim=1) 
    # 

    
print(len(result))

# submission_df = pd.read_csv(os.path.join(config.submission_path(), "sample_submission.csv"))
# submission_df['Label'] = preds.numpy().squeeze()
# submission_df.head()