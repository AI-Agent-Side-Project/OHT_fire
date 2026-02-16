import os
import numpy as np
import pandas as pd
import glob
import re
import torch
import pdb
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
from utils.augmentation import run_augmentation_single
warnings.filterwarnings('ignore')

class OHT_fire_Loader(Dataset):
    def __init__(self, args, root_path, flag = None, scaler = None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.seq_len = args.seq_len
        self.min = np.array([27.0, 18.0, 9.0, 6.0, 1.8, 0.55, 0.2, 0.29, 22.0, 35.0, 511.0])
        self.max = np.array([85.1, 22.0, 14.0, 10.0, 168.76, 173.36, 171.73, 170.4, 23.0, 36.0, 530.0])
        self.mean = np.array([31.915677, 19.995953, 11.837508, 8.001406, 6.191396, 13.110211, 4.784334, 2.550489, 22.500381, 35.502464, 520.499026])
        self.std = np.array([8.053964, 0.570940, 0.649219, 0.746791, 17.315621, 28.457335,17.689338, 12.563485, 0.500004, 0.499998, 5.765662])
        self.eps = 1e-8
        self.datas_df, self.labels_df = self.load_data(root_path, flag)

    def load_data(self, root_path, flag):
        self.ts_target_col = ['NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4', \
                'ex_temperature', 'ex_humidity', 'ex_illuminance']
        self.label_col = ['tagging_state']
        if self.args.multimodal == True:
            self.img_target_col = ['value_TGmx', 'X_Tmax', 'Y_Tmax']

        if flag == 'test' or flag == 'TEST':
            file_list = glob.glob(os.path.join(root_path, 'Test/Data/*.csv'))
            data_li = []
            label_li = []
            for file_dir in file_list:
                df = pd.read_csv(file_dir)
                for i in range(0, len(df) - self.args.seq_len + 1, 1):
                    sub_df = df.iloc[i : i + self.args.seq_len, :]
                    
                    raw_data = sub_df[self.ts_target_col].values
                    if self.args.norm == 'std':
                        normalized_data = (raw_data - self.mean) / (self.std + self.eps)
                    elif self.args.norm == 'minmax':
                        normalized_data = (raw_data - self.min) / (self.max - self.min + self.eps)
                    else:
                        normalized_data = raw_data   
                    data_li.append(normalized_data)
                    label_li.append(sub_df[self.label_col].values[-1][0])
        else :
            file_list = glob.glob(os.path.join(root_path, 'Train/Data/*.csv'))
            data_li = []
            label_li = []
            for file_dir in file_list:
                df = pd.read_csv(file_dir)
                # Train 데이터로 scaler fit (모든 train 파일의 데이터로 fit)
                self.scaler.fit(df[self.ts_target_col].values)
                for i in range(0, len(df) - self.args.seq_len + 1, self.args.stride):
                    sub_df = df.iloc[i : i + self.args.seq_len, :]
                    # Train 데이터에 scaler 적용 (transform)
                    if self.args.norm == 'std':
                        normalized_data = (raw_data - self.mean) / (self.std + self.eps)
                    elif self.args.norm == 'minmax':
                        normalized_data = (raw_data - self.min) / (self.max - self.min + self.eps)
                    else:
                        normalized_data = raw_data   
                    data_li.append(normalized_data)
                    label_li.append(sub_df[self.label_col].values[-1][0])

        return data_li, label_li
        
    def __getitem__(self, index):
        data = self.datas_df[index]
        label = self.labels_df[index]
        return torch.tensor(data), torch.tensor(label)
    
    def __len__(self):
        return len(self.datas_df)
