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
    output = model(data).cpu().detach().numpy()
    data = None

    final_output.append(output)
result = np.concatenate(final_output)

print((result))

# submission_df = pd.read_csv(os.path.join(config.submission_path(), "sample_submission.csv"))
# submission_df['Label'] = result.tolist()
# submission_df.head()

# submission_df = pd.read_csv(os.path.join(config.submission_path(), "sample.csv"))


# pd.DataFrame(result).to_csv("Dropbox/MNIST-PT/submission/sample.csv")

