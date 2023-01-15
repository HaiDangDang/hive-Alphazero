"""
Holds the worker which trains the chess model using self play data.
"""
import os
import torch
import pygame as pg
import numpy as np
from tile import draw_drag

import multiprocessing as mp
from threading import Thread
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager

import time
from copy import deepcopy
from collections import deque
from datetime import datetime

from hive_engine.config import MAX_PROCESS, SEARCH_THREADS, MAX_GAME_LENGTH, ACTION_SPACE
from settings import WIDTH, HEIGHT, PIECE_WHITE, PIECE_BLACK
from hive_engine.env_hive import GamePlay

from woker.solo_play import HivePlayer
from woker.sl import write_game_data_to_file
from woker.api_hive import HiveModelAPI


from alpha_zero.alpha_net import ChessNet
from move_checker import is_valid_move
from inventory_frame import Inventory_Frame
from settings import BACKGROUND

def get_pipes(net, num=1):
    api = HiveModelAPI(net)
    api.start()
    return [api.create_pipe() for _ in range(num)]

# noinspection PyAttributeOutsideInit
class SelfPlayWorker:
    def __init__(self):
        self.path_main = "iter_5_6.pth.tar"
        self.model_1 = self.load_model(self.path_main)
        self.model_2 = self.load_model("iter_1_6.pth.tar")

        self.m = Manager()
        self.max_process = 20
        self.cur_pipes_1 = self.m.list([get_pipes(self.model_1, SEARCH_THREADS) for _ in range(int(self.max_process))])
        self.cur_pipes_2 = self.m.list([get_pipes(self.model_2, 1) for _ in range(int(self.max_process))])

        self.buffer = []
        self.win_lose = []
        self.state_keys = []
    def start(self, main_player='W', render=False):
        if render:
            self.max_process = 1
        """
        Do self play and write the data to the appropriate file.
        """
        self.buffer = []
        futures = deque()
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=self.max_process) as executor:
            for game_idx in range(self.max_process * 2):

                futures.append(
                    executor.submit(self_play_buffer, cur_1=self.cur_pipes_1,
                                    cur_2=self.cur_pipes_2,
                                    main_player=main_player, render=render))
            game_idx = 0
            while True:
                game_idx += 1
                data, value_white, state_key = futures.popleft().result()
                self.win_lose += value_white
                self.buffer += [data]
                self.state_keys += [state_key]
                if (game_idx % 10) == 0:
                    end_time = time.time()
                    win_rate = len(np.where(np.array(self.win_lose) == 1)[0]) / len(self.win_lose)

                    _, counter_white = np.unique(self.win_lose, return_counts=True)

                    print(f" Model {self.path_main} --- "
                          f" Total_game {len(self.win_lose)} --- "
                          f" Mean_game_len {np.round(np.mean(self.buffer), 2)} --- "
                          f" White_Win % {np.round(win_rate, 2)} --- "
                          f" Main_player {main_player}",
                          f"Numbers_of_total_game {np.round(len(np.unique(self.state_keys))/len(self.win_lose), 2)}")
                    print(f"Counter {counter_white} {_}")
                    start_time = time.time()
                if game_idx <= 6000:
                    futures.append(
                        executor.submit(self_play_buffer, cur_1=self.cur_pipes_1,
                                        cur_2=self.cur_pipes_2,
                                        main_player=main_player, render=render))
                else:
                    break

    def load_model(self, path) -> ChessNet:
        net = ChessNet()
        cuda = torch.cuda.is_available()
        if cuda:
            net.cuda()
        net.share_memory()
        net.eval()
        current_net_filename = os.path.join("./model_data/", \
                                            path)
        checkpoint = torch.load(current_net_filename)
        net.load_state_dict(checkpoint['state_dict'])

        return net

tau_decay_rate= 0.9
def apply_temperature(policy, turn):
    tau = np.power(tau_decay_rate, turn)
    if tau < 0.1:
        tau = 0
    if tau == 0:
        action = np.argmax(policy)
        ret = np.zeros(len(policy))
        ret[action] = 1.0
        return ret
    else:
        ret = np.power(policy, 1/tau)
        ret /= np.sum(ret)
        return ret

def self_play_buffer(cur_1, cur_2, main_player='W', render=False) -> list:
    board = GamePlay(HEIGHT_MAP=HEIGHT - 100, WIDTH_MAP=WIDTH - 500)

    if render:
        pg.font.init()
        screen = pg.display.set_mode((WIDTH, HEIGHT))
        background = pg.Surface(screen.get_size())

        pg.display.set_caption('Hive')
        icon = pg.image.load('images/icon.png')
        pg.display.set_icon(icon)

        white_inventory = Inventory_Frame((0, 158), 0, white=True, training=False)
        black_inventory = Inventory_Frame((700, 158), 1, white=False, training=False)
        board.state.running = True
        board.state.main_loop = True
    pipes_1 = cur_1.pop()
    pipes_2 = cur_2.pop()
    if main_player == "W":
        white = HivePlayer(pipes=pipes_1, reward=True)
        black = HivePlayer(pipes=pipes_2, reward=True)
    else:
        white = HivePlayer(pipes=pipes_2, reward=True)
        black = HivePlayer(pipes=pipes_1, reward=True)

    while not board.game_is_over():
        if board.state.player() == 0:
            policy_v, value = white.expand_and_evaluate(board)
            actions = board.actions()
            if len(actions) == 0:
                action = -1
            else:
                policy = policy_v[actions]
                # p = apply_temperature(policy, int(board.state.turn+1)/2)
                # my_action = int(np.random.choice(range(len(policy)), p=p))
                #
                # action = actions[my_action]
                action = np.argmax(policy)
                action = actions[action]
                # if main_player == 'W':
                #     action, policy = white.action(deepcopy(board))
                if board.state.turn <= 4:
                    action = np.random.choice(board.actions())
        else:
            policy_v, value = black.expand_and_evaluate(board)
            actions = board.actions()
            if len(actions) == 0:
                action = -1
            else:
                policy = policy_v[actions]
                #
                # p = apply_temperature(policy, int(board.state.turn+1)/2)
                # my_action = int(np.random.choice(range(len(policy)), p=p))
                # action = actions[my_action]
                action = np.argmax(policy)
                action = actions[action]
                if main_player == 'B':
                    action, policy = black.action(board)
                    main_player = 'W'
                if board.state.turn <= 4:
                    action = np.random.choice(board.actions())
                # print(np.max(policy[0]))
        # time.sleep(1)
        # else:
        #     if board.state.player() == 1:
        #         action, policy = black.action(deepcopy(board))
        #         player = 'B'
        #         policy = policy[0]
        #         black_count += 1
        #         counter = black_count
        #
        #     else:
        #         # print(len(board.history_white), board.state.player())
        #         policy_v, value = white.expand_and_evaluate(deepcopy(board))
        #         actions = board.actions()
        #         if len(actions) == 0:
        #             action = -1
        #         else:
        #             white_count += 1
        #             counter = white_count
        #
        #             policy = policy_v[actions]
        #             action = np.argmax(policy)
        #             action = actions[action]
        #             player = 'W'
        #             policy = policy_v.tolist()
        #             # if board.state.turn <= 4:
        #             #     action = np.random.choice(board.actions())

        # print(len(state_policy_player))
        state = board.encode_board()
        # board.move(action)
        if board.state.turn >= MAX_GAME_LENGTH:
            break

        if render:
            # time.sleep(0.5)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    board.state.quit()
                    # break
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        board.state.quit()
                        break
                if event.type == pg.MOUSEBUTTONDOWN:
                    board.state.click()
                if event.type == pg.MOUSEBUTTONUP:
                    board.state.unclick()
                    if board.state.moving_piece and board.state.is_player_turn():
                        old_tile = next(tile for tile in
                                        board.state.board_tiles if tile.has_pieces()
                                        and tile.pieces[-1]
                                        == board.state.moving_piece)
                        new_tile = next((tile for tile in
                                         board.state.board_tiles
                                         if tile.under_mouse(pos)), None)
                        if is_valid_move(board.state, old_tile, new_tile):
                            state = board.encode_board()
                            old_tile.move_piece(new_tile)
                            main_player = 'B'
                            if board.state.player() == 1:
                                for piece, value in board.black_pieces_set.items():
                                    tile = value[0]
                                    if tile == old_tile:
                                        board.black_pieces_set[piece] = [new_tile, 0, value[2]]
                                        print("ccc")
                                        if old_tile.axial_coords != (99, 99):
                                            board.latest_pos['B'] = (piece, old_tile)
                                        break
                            else:
                                for piece, value in board.white_pieces_set.items():
                                    tile = value[0]
                                    if tile == old_tile:
                                        board.white_pieces_set[piece] = [new_tile, 0, value[2]]

                                        if old_tile.axial_coords != (99, 99):
                                            board.latest_pos['W'] = (piece, old_tile)
                                        break
                            board.state.next_turn()
                            board.human_play()
                    board.state.remove_moving_piece()

            pos = pg.mouse.get_pos()
            background.fill(BACKGROUND)
            white_inventory.draw(background, pos)
            black_inventory.draw(background, pos)
            for tile in board.state.board_tiles:
                if board.state.clicked:
                    tile.draw(background, pos, board.state.clicked)

                    if tile.under_mouse(pos) and board.state.moving_piece \
                            is None and tile.has_pieces():
                        board.state.add_moving_piece(tile.pieces[-1])
                else:
                    tile.draw(background, pos)
            if board.state.moving_piece:
                draw_drag(background, pos, board.state.moving_piece)
            board.state.turn_panel.draw(background, board.state.turn)
            screen.blit(background, (0, 0))
            pg.display.flip()

    state_key = board.state_key
    if board.game_is_over():
        if board.state.winner == PIECE_WHITE:  # black wins
            value_white = 1
        elif board.state.winner == PIECE_BLACK:  # white wins
            value_white = -1
        else:
            value_white = 0
    else:
        value_white = 0
    # print(value_white, board.state.turn, main_player)
    white.finish_game(value_white)
    black.finish_game(-value_white)

    data = [board.state.turn]

    cur_1.append(pipes_1)
    cur_2.append(pipes_2)
    if main_player == 'B':
        value_white *= -1
    return data, [value_white], state_key

def main(worker, render=False):
    worker.start(main_player="B", render=render)


if __name__ == '__main__':
    os.chdir("../")
    import sys
    sys.setrecursionlimit(20000)
    print(sys.getrecursionlimit())
    mp.set_start_method('spawn')

    print(os.getcwd())
    worker = SelfPlayWorker()
    main(worker, render=True)
