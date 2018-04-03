"""
Function moves data files from their base directory to "train" or "eval" 
subdirectories
"""
from glob import glob
import os
import numpy as np
import pandas as pd


if os.path.exists("D:/data/PAC_Data/PAC_data/"):
    base = "D:/data/PAC_Data/PAC_data/"
elif os.path.exists("/media/Data/Ian/Data/PAC_Data"):
    base = "/media/Data/Ian/Data/PAC_Data"
    
data = pd.read_csv(os.path.join(base, "PAC2018_Covariates_Upload.csv"))
# number of smaller class
N = int(np.sum(data.Label==2)*.2)

first_class = np.where(data.Label==1)[0][0:N]
second_class = np.where(data.Label==2)[0][0:N]
first_class_files = data.iloc[first_class,0]
second_class_files = data.iloc[second_class,0]

# move files
train_loc = os.path.join(base, 'train')
eval_loc = os.path.join(base, 'eval')
# make training and eval directories if necessary
os.makedirs(train_loc, exist_ok=True)
os.makedirs(eval_loc, exist_ok=True)

# move eval files
for filey in first_class_files:
    os.rename(os.path.join(base, filey+'.nii'),
              os.path.join(eval_loc, filey+'.nii'))
for filey in second_class_files:
    os.rename(os.path.join(base, filey+'.nii'),
              os.path.join(eval_loc, filey+'.nii'))
# move the rest to training
for filey in glob(os.path.join(base, '*.nii')):
    name = os.path.basename(filey)
    os.rename(filey, os.path.join(train_loc, name))