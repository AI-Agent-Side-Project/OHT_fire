import argparse
import os
import torch
import torch.backends
import random
import numpy as np
import pdb
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--multimodal', action='store_true', help='use multimodal data', default=False)
    parser.add_argument('--task_name', type = str, default = 'classification')
    parser.add_argument('--is_training', type = int, default = 1, help = 'status')
    parser.add_argument('--model', type = str, default = 'TimesNet')
    parser.add_argument('--norm', type = str, default = 'minmax')

    # data loader
    parser.add_argument('--date', type = str, default = 'ETTh1')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')


    # GPU
    parser.add_argument('--use_gpu', type = bool, default = False)
    parser.add_argument('--gpu', type = int, default = 0)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)

    # optimization
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    # data loader
    parser.add_argument('--stride' , type=int, default=4, help='data stride')
    parser.add_argument('--seq_len', type=int, default=16, help='input sequence length')
    parser.add_argument('--data', type=str, default='OHT_fire', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./database/OHT/', help='root path of the data file')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    
    # XAI options
    parser.add_argument('--use_xai', action='store_true', help='use SHAP-based XAI analysis', default=False)
    parser.add_argument('--xai_num_samples', type=int, default=100, help='number of samples for XAI analysis')

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            print('Using MPS')
        else:
            args.device = torch.device("cpu")
            print('Using CPU')
        
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]



    Exp = Exp_Classification
    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_dm{}_dff{}_topk{}_sl{}_n{}'.format('TSC', args.d_model, args.d_ff, args.top_k, args.seq_len, args.norm)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            
            # XAI analysis after testing
            if args.use_xai:
                print('>>>>>>>XAI analysis : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                test_data, test_loader = exp._get_data(flag='TEST')
                exp.xai_shap(setting, test_data, test_loader, num_samples=args.xai_num_samples)
            
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        ii = 0
        setting = '{}_dm{}_dff{}_topk{}_sl{}_n{}'.format('TSC', args.d_model, args.d_ff, args.top_k, args.seq_len, args.norm)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        
        # XAI analysis after testing
        if args.use_xai:
            print('>>>>>>>XAI analysis : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            test_data, test_loader = exp._get_data(flag='TEST')
            exp.xai_shap(setting, test_data, test_loader, num_samples=args.xai_num_samples)
        
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()