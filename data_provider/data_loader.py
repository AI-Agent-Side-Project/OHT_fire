import os
import numpy as np
import pandas as pd
import glob
import re
import torch
import pdb
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
from utils.augmentation import run_augmentation_single
warnings.filterwarnings('ignore')

# Global scaler cache for consistency between train/test
_scaler_cache = {}

class OHT_fire_Loader(Dataset):
    def __init__(self, args, root_path, flag=None, scaler=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.seq_len = args.seq_len
        self.scaler = scaler
        self.ts_target_col = ['NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4',
                              'ex_temperature', 'ex_humidity', 'ex_illuminance']
        self.label_col = ['tagging_state']
        if self.args.multimodal:
            self.img_target_col = ['value_TGmx', 'X_Tmax', 'Y_Tmax']
        
        # Load and normalize data
        self.datas_df, self.labels_df, self.file_mapping = self.load_data(root_path, flag)


    def load_data(self, root_path, flag):
        """
        Load and normalize data with sklearn scalers
        """
        if flag == 'test' or flag == 'TEST':
            file_list = sorted(glob.glob(os.path.join(root_path, 'Test/Data/*.csv')))
            stride = 1  # Stride=1 for test data
        else:
            file_list = sorted(glob.glob(os.path.join(root_path, 'Train/Data/*.csv')))
            stride = self.args.stride
        
        data_li = []
        label_li = []
        file_mapping = []  # Track which file each sample comes from
        
        # Initialize or load scaler
        if self.scaler is None:
            if self.args.norm == 'std':
                self.scaler = StandardScaler()
            elif self.args.norm == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.args.norm == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = None
        
        # For training data, fit the scaler first
        if flag != 'test' and flag != 'TEST' and self.scaler is not None:
            print("Fitting scaler on training data...")
            all_data = []
            for file_dir in file_list:
                df = pd.read_csv(file_dir)
                raw_data = df[self.ts_target_col].values
                all_data.append(raw_data)
            all_data = np.vstack(all_data)
            self.scaler.fit(all_data)
            print(f"Scaler fitted on {len(all_data)} samples")
        
        # Load and normalize data
        for file_idx, file_dir in enumerate(file_list):
            df = pd.read_csv(file_dir)
            file_name = os.path.basename(file_dir)
            
            for i in range(0, len(df) - self.seq_len + 1, stride):
                sub_df = df.iloc[i : i + self.seq_len, :]
                raw_data = sub_df[self.ts_target_col].values
                
                # Normalize data
                if self.scaler is not None:
                    normalized_data = self.scaler.transform(raw_data)
                else:
                    normalized_data = raw_data
                
                data_li.append(normalized_data)
                label_li.append(sub_df[self.label_col].values[-1][0])
                file_mapping.append({
                    'file_name': file_name,
                    'file_idx': file_idx,
                    'start_idx': i,
                    'end_idx': i + self.seq_len
                })
        
        print(f"Loaded {len(data_li)} samples from {len(file_list)} files")
        return data_li, label_li, file_mapping
    
    def inverse_transform(self, normalized_data):
        """Inverse transform normalized data back to original scale"""
        if self.scaler is not None:
            return self.scaler.inverse_transform(normalized_data)
        return normalized_data
    
    def get_scaler(self):
        """Get the scaler object"""
        return self.scaler
        
    def __getitem__(self, index):
        data = self.datas_df[index]
        label = self.labels_df[index]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.datas_df)
