import os
import argparse

from torch.backends import cudnn
from utils.utils import *
import pandas as pd
from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    return solver

class AnomalyTransformer:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, lr: float = 1e-4, num_epochs: int = 10, k: int = 3, win_size: int = 100, input_c: int = 38, output_c: int = 38, batch_size: int = 256, pretrained_model: str = None, model_save_path: str = 'checkpoints', anormly_ratio: float = 4.00) -> None:
        cudnn.benchmark = True
        if (not os.path.exists(model_save_path)):
            mkdir(model_save_path)
        self.solver = Solver(train, test, lr, num_epochs, k, win_size, input_c, output_c, batch_size, pretrained_model, model_save_path, anormly_ratio)
    
    def fit(self):
        self.solver.train()
        pass
    
    def get_associations(self):
        s = self.solver.test()
        print(s)
        return s
    
    def score(self):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
