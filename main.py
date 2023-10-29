"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/10/25 13:34
@File : main.py
"""
from __init__ import *
from train import *
from inference import *

parser = argparse.ArgumentParser(description='Replication of Reimaging Price Trend')
parser.add_argument('-s', '--settings', type=str, required=True, metavar='',
                    help='Location of your Training configs file')
parser.add_argument('-t', '--train', action='store_true',
                    help='Turn on Training mode')
parser.add_argument('-i', '--inference', action='store_true',
                    help='Turn on Inference/Testing mode')
args = parser.parse_args()


if __name__ == '__main__':
    settings = yaml.load(open(args.settings, 'r'), Loader=yaml.FullLoader)

    if args.train:
        train_main(settings)
    elif args.inference:
        inference_main(settings)
    else:
        print('===========')

