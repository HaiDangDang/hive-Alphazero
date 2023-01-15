"""
Holds the worker which trains the chess model using self play data.
"""
import os
import torch
import numpy as np
import multiprocessing as mp
from threading import Thread
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager


from collections import deque
from datetime import datetime
from time import time
from copy import deepcopy

from hive_engine.config import MAX_PROCESS, SEARCH_THREADS, MAX_GAME_LENGTH
from settings import WIDTH, HEIGHT, PIECE_WHITE, PIECE_BLACK

from woker.solo_play import HivePlayer
from woker.sl import write_game_data_to_file
from woker.api_hive import HiveModelAPI

from hive_engine.env_hive import GamePlay

from alpha_zero.alpha_net import ChessNet


def get_pipes(net, num=1):
    api = HiveModelAPI(net)
    api.start()
    return [api.create_pipe() for _ in range(num)]

# noinspection PyAttributeOutsideInit
class SelfPlayWorker:
    def __init__(self):
        self.current_model = self.load_model()
        self.m = Manager()
        self.cur_pipes = self.m.list([get_pipes(self.current_model, SEARCH_THREADS) for _ in range(MAX_PROCESS)])
        self.buffer = []
        self.running = True
    def start(self):
        """
        Do self play and write the data to the appropriate file.
        """
        self.buffer = []
        self.win_lose = []

        nb_game_in_file = 100
        futures = deque()
        start_time = time()
        game_lens = []
        with ProcessPoolExecutor(max_workers=MAX_PROCESS) as executor:
            for game_idx in range(MAX_PROCESS * 2):
                futures.append(executor.submit(self_play_buffer, cur=self.cur_pipes))
            game_idx = 0
            while True:
                game_idx += 1
                data, value_white = futures.popleft().result()
                game_lens.append(len(data))
                self.buffer += data
                self.win_lose += value_white

                if (game_idx % nb_game_in_file) == 0:
                    print(f"SAVE {game_idx}")
                    self.flush_buffer()

                if (game_idx % 10) == 0:
                    win_rate = len(np.where(np.array(self.win_lose) == 1)[0]) / len(self.win_lose)
                    _, counter_white = np.unique(self.win_lose, return_counts=True)
                    print(f" Total_game {len(self.win_lose)} --- "
                          f" Mean_game_len {np.round(np.mean(game_lens), 2)} --- "
                          f" White_Win % {np.round(win_rate, 2)} --- ")
                    print(f"Counter {counter_white} {_}")

                if game_idx <= 12000 and self.running:
                    futures.append(executor.submit(self_play_buffer, cur=self.cur_pipes)) # Keep it going
                else:
                    break

        if len(data) > 0:
            self.flush_buffer()

    def load_model(self) -> ChessNet:
        net = ChessNet()
        cuda = torch.cuda.is_available()
        if cuda:
            net.cuda()
        net.share_memory()
        net.eval()
        save_as = "iter_1_6_TS5.pth.tar"
        current_net_filename = os.path.join("./model_data/", \
                                            save_as)
        checkpoint = torch.load(current_net_filename)
        net.load_state_dict(checkpoint['state_dict'])

        return net

    def flush_buffer(self):
        """
        Flush the play data buffer and write the data to the appropriate location
        """

        datapath = "../dataSelf"

        play_data_filename_tmpl = "play_%s.json"
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(datapath, play_data_filename_tmpl% game_id)
        thread = Thread(target=write_game_data_to_file, args=(path, self.buffer))
        thread.start()
        self.buffer = []



def self_play_buffer(cur) -> list:
    board = GamePlay(HEIGHT_MAP=HEIGHT - 100, WIDTH_MAP=WIDTH - 500)
    pipes = cur.pop()

    white = HivePlayer(pipes=pipes)
    black = HivePlayer(pipes=pipes)

    state_policy_player = []
    black_count = 0
    white_count = 0
    e = 0.7
    while not board.game_is_over():
        if board.state.player() == 0:
            action, policy = white.action(board)
            player = 'W'
            white_count += 1
            counter = white_count
        else:
            action, policy = black.action(board)
            player = 'B'
            black_count += 1
            counter = black_count

        if board.state.turn <= 2:
            action = np.random.choice(board.actions())

        sum_all = policy[1]
        policy = policy[0]

        error = e - int(board.state.turn+1)/2 * 0.15
        actions = board.actions()

        if error >= 0.1 and len(actions) != 0:
            p = deepcopy(policy[actions])
            # noise = np.random.normal(0, error, len(actions))
            # policy[actions] += noise
            # policy[actions] = policy[actions] - np.min(policy[actions])

            noise = np.random.dirichlet([0.5] * len(actions))
            p = (1 - error) * np.array(p) + error * noise
            p /= p.sum()
            action = np.random.choice(board.actions(), p=p)

        state = board.encode_board(player)
        state_policy_player.append([state.tolist(), policy, player, counter])
        board.move(action)
        if board.state.turn >= MAX_GAME_LENGTH:
            break

    if board.game_is_over():
        if board.state.winner == PIECE_WHITE:  # black wins
            value_white = 1
        elif board.state.winner == PIECE_BLACK:  # white wins
            value_white = -1
        else:
            value_white = 0
    else:
        value_white = 0
    print(value_white, board.state.turn)
    white.finish_game(value_white)
    black.finish_game(-value_white)

    data = []
    game_lens = 0
    for state, policy, player, counter in state_policy_player:
        if player == "W":
            value = value_white
            game_lens = white_count
        elif player == "B":
            value = value_white * -1
            game_lens = black_count

        if value_white == 0:
            value = -1
        data.append([state,  policy, value, [game_lens, counter]])

    cur.append(pipes)
    return data, [value_white]

def main(worker):
    worker.start()


if __name__ == '__main__':
    os.chdir("../")
    import sys
    sys.setrecursionlimit(20000)
    print(sys.getrecursionlimit())

    mp.set_start_method('spawn')

    print(os.getcwd())
    worker = SelfPlayWorker()
    try:
        main(worker)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            worker.running = False
            sys.exit(0)
        except SystemExit:
            os._exit(0)


