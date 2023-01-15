import os
import numpy as np
import pickle
import multiprocessing as mp
import sys
import json
from collections import deque
from json import JSONEncoder

from datetime import datetime
from time import time
import ujson

from logging import getLogger

from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from threading import Thread


import pieces
from hive_engine.env_hive import GamePlay
from hive_engine.config import MAX_MAP_FULL, index_char, index_number, BOT_WEIGHT, DISCOUNTED_REWARD
from settings import WIDTH, HEIGHT, PIECE_WHITE, PIECE_BLACK

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
logger = getLogger(__name__)


def decode_piece(piece):
    if piece == "Q":
        return str(pieces.Queen) + "0"
    elif piece[0] == "G":
        return str(pieces.Grasshopper) + str(int(piece[1]) - 1)
    elif piece[0] == "B":
        return str(pieces.Beetle) + str(int(piece[1]) - 1)
    elif piece[0] == "S":
        return str(pieces.Spider) + str(int(piece[1]) - 1)
    elif piece[0] == "A":
        return str(pieces.Ant) + str(int(piece[1]) - 1)


piece_types = ["G1", "G2", "G3", "A1", "A2", "A3", "S1", "S2", "B1", "B2", "Q"]


def write_game_data_to_file(path, data):
    # try:
    #     with open(path + '.pkl', 'wb') as f:
    #         pickle.dump(data, f)
    # except Exception as e:
    #     print(e)
    try:
        # np.save(path, data)
        with open(path, "wt") as f:
            ujson.dump(data, f)
    except Exception as e:
        print(e)

def open_game(path):
    try:
        with open(path + '.pkl', 'rb') as f:
            b = pickle.load(f)

    except Exception as e:
        print(e)
    return b

class SupervisedLearningWorker:
    def __init__(self):
        self.buffer = []

    def start(self, index):
        self.buffer = []
        self.game_names = []
        self.lens_all = 0
        self.idx = 0
        games = self.get_games_from_all_files()
        start_with = 10000 * index
        end_with = 10000 * (index + 1)
        games = deque(games[start_with: end_with])
        print(len(games), f"from {start_with} to {end_with}")
        start_time = time()
        print("START")
        while len(games) != 0:
            g_c = [games.popleft() for _i in range(min(2000, len(games)))]
            with ProcessPoolExecutor(max_workers=60) as executor:
                for res in as_completed([executor.submit(get_buffer, game) for game in g_c]):
                    self.idx += 1
                    if self.idx % 10 == 0:
                        print(self.idx, time() - start_time)
                        start_time = time()
                    data, game_name = res.result()
                    self.save_data(data, [game_name])

            if len(self.buffer) > 0:
                self.flush_buffer()
        end_time = time()
        print(end_time - start_time)
        print(self.lens_all)

    def get_games_from_all_files(self):
        games = []
        database = "../datahist"
        x = []
        for idx, file in enumerate(os.listdir(database)):
            file_name = os.path.join(database, file)
            with open(file_name, 'rb') as f:
                data_boards = pickle.load(f)
            x.append(data_boards)
        games.extend(x)
        # print(len(games))
        games = games
        return games

    def save_data(self, data, game_name):
        self.buffer += data
        self.game_names += game_name
        sl_nb_game_in_file = 100

        if self.idx % sl_nb_game_in_file == 0:
            self.flush_buffer()

    def flush_buffer(self):
        datapath = "../datapath"
        os.chdir("/media/winter/5B1DBA0733F1427F")
        datapath = './datasets'

        play_data_filename_tmpl = "play_%s.json"
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(datapath, play_data_filename_tmpl% game_id)
        logger.info(f"save play data to {path}")
        thread = Thread(target=write_game_data_to_file, args=(path, self.buffer))
        thread.start()

        datapath = './dataname'
        path = os.path.join(datapath, play_data_filename_tmpl% game_id)
        thread = Thread(target=write_game_data_to_file, args=(path, self.game_names))

        thread.start()
        self.lens_all += len(self.buffer)
        self.buffer = []

def get_buffer(game) -> list:
    board = GamePlay(HEIGHT_MAP=HEIGHT - 100, WIDTH_MAP=WIDTH - 500)
    state_policy_player = []
    black_count = 0
    white_count = 0
    black_bot = False
    white_bot = True
    for i in range(len(game)):

        current_step = game[i]
        piece, x, y, player, bot = current_step[0], current_step[1], current_step[2], current_step[3], current_step[4]

        if ((board.player() == 1 and player == "W") or
            (board.player() == 0 and player == "B")):
            # state_policy_player = []
            # break
            board.skip_turn()
        if player == "W":
            white_count += 1
            counter = white_count
            if bot == 1:
                white_bot = True
        else:
            black_count += 1
            counter = black_count
            if bot == 1:
                black_bot = True

        y = index_number.index(y)
        x = index_char.index(x)
        end_tile = board.board_matrix[x, y]
        piece = decode_piece(piece)
        board_action = board.actions()
        action_list = {}
        action_list[piece] = [end_tile]

        action = board.encode_action(action_list)

        if action[0] not in board_action:
            print("CCC")
            state_policy_player = []
            break
        policy = np.zeros(MAX_MAP_FULL * MAX_MAP_FULL * 11)

        weight = 1
        if bot == 1:
            weight = BOT_WEIGHT
        # elif white_bot or black_bot:
        #     weight = 0.8
        policy[action] = weight

        state = board.encode_board(player)
        state_policy_player.append([state.tolist(), policy, player, counter])
        board.move(action[0], with_skip=False)

    check_bot_win = False
    if board.game_is_over():
        if board.state.winner == PIECE_WHITE:  # black wins
            value_white = 1
            if white_bot:
                check_bot_win = True
        elif board.state.winner == PIECE_BLACK:  # white wins
            value_white = -1
            if black_bot:
                check_bot_win = True
        else:
            value_white = 0
    else:
        value_white = 0
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
            value = 0
        # if check_bot_win:
        #     policy *= 0.7
        data.append([state,  policy.tolist(), value, [game_lens, counter]])
    # print(data)
    return data, game


# SupervisedLearningWorker().start()


def start(index):
    return SupervisedLearningWorker().start(index)


if __name__ == "__main__":
    os.chdir("../")
    print(os.getcwd())

    mp.set_start_method('spawn')
    sys.setrecursionlimit(10000)
    index = 4
    start(index)

