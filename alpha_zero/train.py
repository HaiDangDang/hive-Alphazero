#!/usr/bin/env python

from alpha_zero.alpha_net import ChessNet, train
import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp

def train_chessnet(net_to_train="current_net_trained8_iter1.pth.tar",save_as="current_net_trained8_iter1.pth.tar"):
    # gather data
    datasets = []

    for i in range(7):
        data_path = f"./datasets/iter{0}/"
        for idx,file in enumerate(os.listdir(data_path)):
            filename = os.path.join(data_path,file)
            with open(filename, 'rb') as fo:
                datasets.extend(pickle.load(fo, encoding='bytes'))
    #
    # data_path = "./datasets/iter1/"
    # for idx,file in enumerate(os.listdir(data_path)):
    #     filename = os.path.join(data_path,file)
    #     with open(filename, 'rb') as fo:
    #         datasets.extend(pickle.load(fo, encoding='bytes'))

    datasets = np.array(datasets)
    mp.set_start_method("spawn", force=True)

    # train net
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    current_net_filename = os.path.join("./model_data/",\
                                    net_to_train)
    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['state_dict'])

    processes2 = []
    for i in range(1):
        p2 = mp.Process(target=train, args=(net, datasets, 0, 12, i))
        p2.start()
        processes2.append(p2)
    for p2 in processes2:
        p2.join()

    # train(net,datasets)
    # save results
    torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",\
                                    save_as))

if __name__=="__main__":
    train_chessnet()