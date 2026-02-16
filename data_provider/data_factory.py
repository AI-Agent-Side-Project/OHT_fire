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
        
        # Train 모드: scaler 없이 로드 (새로 생성되고 fit됨)
        if flag == 'train':
            data_set = Data(
                args = args,
                root_path = args.root_path,
                flag = flag,
                scaler = None
            )
            # Train scaler 캐시에 저장
            _scaler_cache[args.data] = data_set.scaler
        # Test 모드: train에서 fit한 scaler 전달
        else:
            scaler = _scaler_cache.get(args.data)
            if scaler is None:
                raise ValueError("Test 모드 실행 전에 Train 모드를 먼저 실행하여 scaler를 초기화하세요.")
            data_set = Data(
                args = args,
                root_path = args.root_path,
                flag = flag,
                scaler = scaler
            )

        data_loader = DataLoader(
            data_set,
            batch_size = batch_size,
            shuffle = shuffle_flag,
            num_workers = args.num_workers,
            drop_last = drop_last,
        )
        return data_set, data_loader