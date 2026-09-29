'''
Generates dataset for ML Training
- Solutions of randomly generated versions of a shock tube problem
- Will automatically save and create a folder named 'data' in the current working directory
    - Generates a Train/Test/Validation Split
        - Default Ratio is 80/20/0
            - User may modify it here
- Saves outputs as .npy files
    - Outputs are numerically solved Q at every timestep (N_iter, 3, N_cells)
        Q = Conservative State Matrix: Shape (3, N_cells)
'''

import numpy as np
import os
from run import random_Run

# Dataset configuration parameters defined here
    #NOTE: MODIFY random_Run() IN run.py TO MODIFY SOLVER PARAMETERS
 
N_samples = 20

percent_train = 0.8
percent_test = 0.2
percent_validate = 0.0

# Save path (DO NOT MODIFY)
data_path = os.path.join(os.getcwd(), "generated_data")
train_save_path = os.path.join(data_path, "train")
test_save_path = os.path.join(data_path, "test")
validate_save_path = os.path.join(data_path, "validate")



def generate_Euler_Dataset():

    # Create data storage folders if missing
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for path in [train_save_path, test_save_path, validate_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    N_train = int(N_samples*percent_train)
    N_test = int(N_samples*percent_test)
    N_validate = int(N_samples*percent_validate)

    for i in range(N_train):
        print('============================================')
        print(f"Generating train data {i+1}/{N_train}")
        print('============================================')
        train_result = random_Run()
        Q_toSave = np.array(train_result.Q_hist)
        np.save(os.path.join(train_save_path, f"train_{i}.npy"), Q_toSave)
        print('============================================')
        print(f"Saving train data {i+1}/{N_train}")
        print('============================================')

    for i in range(N_test):
        print('============================================')
        print(f"Generating test data {i+1}/{N_test}")
        print('============================================')
        test_result = random_Run()
        Q_toSave = np.array(test_result.Q_hist)
        np.save(os.path.join(test_save_path, f"test_{i}.npy"), Q_toSave)
        print('============================================')
        print(f"Saving test data {i+1}/{N_train}")
        print('============================================')
    for i in range(N_validate):
        print('============================================')
        print(f"Generating validation data {i+1}/{N_validate}")
        print('============================================')
        validate_result = random_Run()
        Q_toSave = np.array(validate_result.Q_hist)
        np.save(os.path.join(validate_save_path, f"validate_{i}.npy"), Q_toSave)
        print('============================================')
        print(f"Saving validation data {i+1}/{N_train}")
        print('============================================')





if __name__ == '__main__':
    generate_Euler_Dataset()