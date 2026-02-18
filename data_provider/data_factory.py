from data_provider.data_loader import OHT_fire_Loader
from torch.utils.data import DataLoader

data_dict = {'OHT_fire' : OHT_fire_Loader}

# Global scaler cache for consistency between train/test
_scaler_cache = {}

def data_provider(args, flag, scaler=None):
    """
    Data provider with shared scaler
    
    Args:
        args: Configuration arguments
        flag: 'TRAIN' or 'TEST'
        scaler: Shared scaler object (optional)
    
    Returns:
        data_set: Dataset object
        data_loader: DataLoader object
        scaler: Scaler object (for sharing across train/test)
    """
    Data = data_dict[args.data]
    
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    
    if args.task_name == 'classification':
        # Create dataset with shared scaler
        data_set = Data(
            args=args,
            root_path=args.root_path,
            flag=flag,
            scaler=scaler
        )
        
        # Get scaler from dataset (important for train to share with test)
        scaler = data_set.get_scaler()
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader, scaler