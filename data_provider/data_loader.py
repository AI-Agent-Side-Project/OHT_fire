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
        self.scaler = scaler  # Train에서 fit한 scaler를 받음
        self.datas_df, self.labels_df = self.load_data(root_path, flag)

    def load_data(self, root_path, flag):
        # Test 모드이고 scaler가 없으면 에러 발생
        if (flag == 'test' or flag == 'TEST') and self.scaler is None:
            raise ValueError("Test 모드에서는 train에서 fit한 scaler를 전달해야 합니다.")
        
        # Train 모드이고 scaler가 없으면 새로 생성
        if (flag != 'test' and flag != 'TEST') and self.scaler is None:
            if self.args.norm_method == 'std':
                self.scaler = StandardScaler()
            elif self.args.norm_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.args.norm_method == 'robust':
                self.scaler = RobustScaler()

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
                    # Test 데이터에 train scaler 적용 (transform만)
                    normalized_data = self.scaler.transform(sub_df[self.ts_target_col].values)
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
                    normalized_data = self.scaler.transform(sub_df[self.ts_target_col].values)
                    data_li.append(normalized_data)
                    label_li.append(sub_df[self.label_col].values[-1][0])

        return data_li, label_li
        
    def __getitem__(self, index):
        data = self.datas_df[index]
        label = self.labels_df[index]
        return torch.tensor(data), torch.tensor(label)
    
    def __len__(self):
        return len(self.datas_df)
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
