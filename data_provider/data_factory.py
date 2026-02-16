from data_provider.data_loader import OHT_fire_Loader
from torch.utils.data import DataLoader

data_dict = {'OHT_fire' : OHT_fire_Loader}

# Train 데이터에서 fit한 scaler를 저장할 캐시
_scaler_cache = {}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path = args.root_path,
            flag = flag
        )

        data_loader = DataLoader(
            data_set,
            batch_size = batch_size,
            shuffle = shuffle_flag,
            num_workers = args.num_workers,
            drop_last = drop_last,
        )
        return data_set, data_loader