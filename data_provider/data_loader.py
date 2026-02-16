import os
import numpy as np
import pandas as pd
import glob
import re
import torch
import pdb
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
from utils.augmentation import run_augmentation_single
warnings.filterwarnings('ignore')

class OHT_fire_Loader(Dataset):
    def __init__(self, args, root_path, flag = None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.seq_len = args.seq_len

        self.datas_df, self.labels_df = self.load_data(root_path, flag)

    def load_data(self, root_path, flag):
        if flag == 'test' or flag == 'TEST':
            file_list = glob.glob(os.path.join(root_path, 'Test/Data/*.csv'))
            data_li = []
            label_li = []
            for file_dir in file_list:
                df = pd.read_csv(file_dir)
                self.ts_target_col = ['NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4', \
                        'ex_temperature', 'ex_humidity', 'ex_illuminance']
                self.label_col = ['tagging_state']
                if self.args.multimodal == True:
                    self.img_target_col = ['value_TGmx', 'X_Tmax', 'Y_Tmax']
                for i in range(0, len(df) - self.args.seq_len + 1, 1):
                    sub_df = df.iloc[i : i + self.args.seq_len, :]
                    data_li.append(sub_df[self.ts_target_col].values)
                    label_li.append(sub_df[self.label_col].values[-1][0])
        else :
            file_list = glob.glob(os.path.join(root_path, 'Train/Data/*.csv'))
            data_li = []
            label_li = []
            for file_dir in file_list:
                df = pd.read_csv(file_dir)
                self.ts_target_col = ['NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4', \
                        'ex_temperature', 'ex_humidity', 'ex_illuminance']
                self.label_col = ['tagging_state']
                if self.args.multimodal == True:
                    self.img_target_col = ['value_TGmx', 'X_Tmax', 'Y_Tmax']
                for i in range(0, len(df) - self.args.seq_len + 1, self.args.stride):
                    sub_df = df.iloc[i : i + self.args.seq_len, :]
                    data_li.append(sub_df[self.ts_target_col].values)
                    label_li.append(sub_df[self.label_col].values[-1][0])

        return data_li, label_li
        
    def __getitem__(self, index):
        data = self.datas_df[index]
        label = self.labels_df[index]
        return torch.tensor(data), torch.tensor(label)
    
    def __len__(self):
        return len(self.datas_df)
