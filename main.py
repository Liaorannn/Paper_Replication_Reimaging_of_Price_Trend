"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/10/25 13:34
@File : main.py
"""
import argparse

from __init__ import *
from train import *

parser = argparse.ArgumentParser(description='Replication of Reimaging Price Trend')
parser.add_argument('-s', '--settings', type=str, required=True, metavar='',
                    help='Location of your Training configs file')
args = parser.parse_args()


if __name__ == '__main__':
    settings = yaml.load(open(args.settings, 'r'), Loader=yaml.FullLoader)

    train_main(settings)


