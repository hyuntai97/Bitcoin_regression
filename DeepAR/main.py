from dataloader import get_dataloader
from model import *
from train import ModelTrain
from evaluate import ModelTest
from utils import generate_serial_number

import torch
import argparse
import os 
import json
import pickle


def config_args(parser):    
    # directory
    parser.add_argument('--datadir', type=str, default='../dataset', help='data directory')
    parser.add_argument('--logdir',type=str, default='./logs', help='logs directory')
    parser.add_argument('--savedir',type=str, default='../save', help='save directory')
    
    # data
    parser.add_argument('--dataname', type=str, default='upbit_ohlcv_1700.csv', help='dataset name')
    parser.add_argument('--target_feature', type=str, default='open', help='the target feature')
    parser.add_argument('--input_window', type=int, default=50, help='input window size')
    parser.add_argument('--output_window', type=int, default=25, help='output window size')
    parser.add_argument('--stride', type=int, default=1, help='stride size')
    parser.add_argument('--train_rate', type=float, default=0.8, help='train rate')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch')

    # train options
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--embedding_size', type=int, default=10, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=50, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--likelihood', type=str, default='g', help='likelihood option')
    parser.add_argument('--n_samples', type=int, default=20, help='number of gaussian samples')
    parser.add_argument('--metric', type=str, default='MAPE', help='test metric')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # set arguments
    parser = argparse.ArgumentParser(description='time series forcasting')
    args = config_args(parser)

    # save directory
    SN = generate_serial_number()
    SAVE_DIR = os.path.join(f'{args.logdir}/{args.dataname}_{args.epochs}_{args.output_window}')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # save arguments
    json.dump(vars(args), open(os.path.join(SAVE_DIR,'arguments.json'),'w'), indent=4)

    # set seed


    # dataloader
    train_dataloader, test_dataloader, yscaler = get_dataloader(target_feature=args.target_feature,
                                                        data_root=args.datadir,                                        
                                                        data_name=args.dataname,
                                                        batch_size=args.batch_size,
                                                        input_window=args.input_window,
                                                        output_window=args.output_window,
                                                        train_rate=args.train_rate,
                                                        stride=args.stride)  

    # build model
    model = DeepAR(input_size=2,
                   embedding_size=args.embedding_size, 
                   hidden_size=args.hidden_size,
                   num_layers=args.num_layers,
                   likelihood=args.likelihood)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # train
    modeltrain = ModelTrain(
        model, 
        train_dataloader,
        args.epochs,
        optimizer
    )

    # save history 
    pickle.dump(modeltrain.history, open(os.path.join(SAVE_DIR, 'train_history.pkl'),'wb'))
    # save model 
    torch.save(modeltrain.model.state_dict(), os.path.join(SAVE_DIR, 'model.pth'))         

    # test
    modeltest = ModelTest(
        modeltrain.model,
        test_dataloader,
        yscaler,
        args.n_samples, 
        args.metric,
        args.input_window,
        args.output_window,
        SAVE_DIR)
                        
    # save history 
    pickle.dump(modeltest.history, open(os.path.join(SAVE_DIR, 'test_history.pkl'),'wb'))
    
    