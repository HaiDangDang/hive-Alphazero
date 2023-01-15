import json
import copy
import numpy as np
import os
from glob import glob
from collections import deque
from random import shuffle
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp
from hive_engine.config import DISCOUNTED_REWARD


import torch
import ujson
import torch.nn as nn
from alpha_zero.alpha_net import ChessNet, train


def read_game_data_from_file(path):
    try:
        with open(path, "rt") as f:
            return ujson.load(f)
    except Exception as e:
        print(e)



def get_game_data_filenames():
    # datapath = "../datapath"
    os.chdir("/media/winter/5B1DBA0733F1427F")

    datapath = './datasets'

    # datapath = "../dataSelf"

    play_data_filename_tmpl = "play_%s.json"

    pattern = os.path.join(datapath, play_data_filename_tmpl % "*")
    files = list(sorted(glob(pattern)))
    return files

def load_data(filename):
    data = read_game_data_from_file(filename)
    dat_x = []
    counter_index = 0
    if data is None:
        print(filename)
    else:
        for state_fen, policy, value, game_lens in data:
            step = game_lens[1]
            game_lens = game_lens[0]
            state = np.array(state_fen)

            policy = np.array(policy, dtype=np.float32)
            if step == game_lens:
                value = value
            else:
                value = (value * DISCOUNTED_REWARD ** (game_lens - step))

            # action = np.argmax(policy)
            # ret = np.zeros(len(policy))
            # ret[action] = 1.0
            dat_x.append([state, policy, value])
            counter_index += 1
    return dat_x

filenames = deque(get_game_data_filenames())
# filename = filenames.popleft()
#
# data = read_game_data_from_file(filename)

dataset = []
futures = deque()

max_workers = 200
lens = 0
with ProcessPoolExecutor(max_workers=16) as executor:
    for _ in range(max_workers):
        if len(filenames) == 0:
            break
        filename = filenames.popleft()
        futures.append(executor.submit(load_data, filename))
    while futures :
        dataset.extend(futures.popleft().result())
        # lens += len(futures.popleft().result())
        print(len(dataset))

        if len(filenames) > 0:
            if len(dataset) <= 9000000:
                filename = filenames.popleft()
                futures.append(executor.submit(load_data, filename))
            else:
                if len(futures) < 2:
                    print(len(filenames))
                    filename = filenames.popleft()
                    futures.append(executor.submit(load_data, filename))

dataset = np.array(dataset, dtype=object)
# x = []
# for d in dataset:
#     a = np.sum(d[1])
#     if a != 1:
#         if a not in x:
#             x.append(a)
# print(len(dataset))
# dataset[0][1].shape
net = ChessNet()
cuda = torch.cuda.is_available()
if cuda:
    net.cuda()
net.share_memory()
net.eval()
# save_as = "iter_1_6_TS5.pth.tar"
# current_net_filename = os.path.join("./model_data/", \
#                                     save_as)
# checkpoint = torch.load(current_net_filename)
# net.load_state_dict(checkpoint['state_dict'])

# processes2 = []
# for i in range(3):
#     p2 = mp.Process(target=train, args=(net, dataset, 0, 8, i))
#     p2.start()
#     processes2.append(p2)
# for p2 in processes2:
#     p2.join()
save_as = "iter_4_6.pth.tar"
torch.save({'state_dict': net.state_dict()},
           save_as)

train(net, dataset, 0, 4, 0)

save_as = "iter_5_6.pth.tar"
# current_net_filename = os.path.join("./model_data/", \
#                                     save_as)
torch.save({'state_dict': net.state_dict()},
           save_as)

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")