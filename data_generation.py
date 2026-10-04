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
from Sod_Shock_Tube_Validation import test_Sod_Shock
from infrastructure.results import EulerResults

# Dataset configuration parameters defined here
    #NOTE: MODIFY random_Run() IN run.py TO MODIFY SOLVER PARAMETERS
 
N_samples = 32

percent_train = 0.66
percent_test = 0.33/2
percent_validate = 0.33/2

# Save path (DO NOT MODIFY)
data_path = os.path.join(os.getcwd(), "generated_data")
train_save_path = os.path.join(data_path, "train")
test_save_path = os.path.join(data_path, "test")
validate_save_path = os.path.join(data_path, "validate")



def generate_Euler_Dataset():
    def craft_ML_data(results : EulerResults, i : int, set_type : str):
        if set_type == 'train':
            save_path = train_save_path
        elif set_type == 'test':
            save_path = test_save_path
        elif set_type == 'validate':
            save_path = validate_save_path

        Q_toSave = np.array(results.Q_hist)
        np.save(os.path.join(save_path, f"{set_type}_{i}.npy"), Q_toSave)

        init_cond = np.array([results.Q_L, results.Q_R, results.x_split*np.ones_like(results.Q_L)])
        np.save(os.path.join(save_path, f"{set_type}_init_{i}.npy"), init_cond)

    # Create data storage folders if missing
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for path in [train_save_path, test_save_path, validate_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    N_train = int(N_samples*percent_train)
    N_test = int(N_samples*percent_test)
    N_validate = int(N_samples*percent_validate)

    print('============================================')
    print(f"Generating Cannonical Sod Shock Tube data")
    print('============================================')
    sod_result = test_Sod_Shock()
    craft_ML_data(sod_result, -1, 'train')
    print('============================================')
    print(f"Saving Sod Shock Tube data")
    print('============================================')

    for i in range(N_train):
        print('============================================')
        print(f"Generating train data {i+1}/{N_train}")
        print('============================================')
        train_result = random_Run()
        craft_ML_data(train_result, i, 'train')
        print('============================================')
        print(f"Saving train data {i+1}/{N_train}")
        print('============================================')
    for i in range(N_test):
        print('============================================')
        print(f"Generating test data {i+1}/{N_test}")
        print('============================================')
        test_result = random_Run()
        craft_ML_data(test_result, i, 'test')
        print('============================================')
        print(f"Saving test data {i+1}/{N_test}")
        print('============================================')
    for i in range(N_validate):
        print('============================================')
        print(f"Generating validation data {i+1}/{N_validate}")
        print('============================================')
        validate_result = random_Run()
        craft_ML_data(validate_result, i, 'validate')
        print('============================================')
        print(f"Saving validation data {i+1}/{N_validate}")
        print('============================================')





if __name__ == '__main__':
    generate_Euler_Dataset()