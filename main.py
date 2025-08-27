# coding=utf-8
# When: 2025/8/19 10:05
# Who: Yuzhen Shen (yuzhenshen@nnu.edu.cn)

import argparse
import torch
from PSONet import PSONetRunner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data', help='data path')
    parser.add_argument('--dcva_path', default='deep feature of DCVA', help='deep features path')
    parser.add_argument('--save_path', default='result', help='result save path')
    parser.add_argument('--data1_name', default='I1', help='name of data1')
    parser.add_argument('--data2_name', default='I2', help='name of data2')
    parser.add_argument('--gt_name', default='gt', help='name of ground truth')
    parser.add_argument('--dcva1_name', default='deepfea1', help='name of deep feature1')
    parser.add_argument('--dcva2_name', default='deepfea2', help='name of deep feature2')
    parser.add_argument('--patch_size', default=15, type=int, help='patch size')
    # Net parameters.
    parser.add_argument('--learn_rate', default=0.001, help='learning rate of net')
    parser.add_argument('--in_channels', default=6, help='RGB or RGBNIR?')
    parser.add_argument('--epoch', default=10, type=int, help='epochs run for net [default: 10]')
    parser.add_argument('--device', default='cpu', help='processor')
    parser.add_argument('--Train', default=True, help='Train?')
    parser.add_argument('--Test', default=True, help='Test?')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    runner = PSONetRunner(args, device)
    runner.run()

if __name__=="__main__":
    main()