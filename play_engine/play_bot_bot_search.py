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

from hive_engine.config import MAX_PROCESS, SEARCH_THREADS, MAX_GAME_LENGTH
from settings import WIDTH, HEIGHT, PIECE_WHITE, PIECE_BLACK
from hive_engine.env_hive import GamePlay

from woker.solo_play import HivePlayer
from woker.sl import write_game_data_to_file
from woker.api_hive import HiveModelAPI


from alpha_zero.alpha_net import ChessNet
from move_checker import is_valid_move
from inventory_frame import Inventory_Frame
from settings import BACKGROUND


MAX_PROCESS = 1
def get_pipes(net, num=1):
    api = HiveModelAPI(net)
    api.start()
    return [api.create_pipe() for _ in range(num)]

# noinspection PyAttributeOutsideInit
class SelfPlayWorker:
    def __init__(self):
        self.current_model_TS = self.load_model("iter_4_4_TS.pth.tar")
        self.current_model = self.load_model("iter_0_4.pth.tar")

        self.m = Manager()
        self.cur_pipes_TS = self.m.list([get_pipes(self.current_model_TS, SEARCH_THREADS) for _ in range(MAX_PROCESS)])
        self.cur_pipes = self.m.list([get_pipes(self.current_model, SEARCH_THREADS) for _ in range(MAX_PROCESS)])

        self.buffer = []
        self.win_lose = []

    def start(self, main_player='W', render=False):
        """
        Do self play and write the data to the appropriate file.
        """
        self.buffer = []
        nb_game_in_file = 50
        futures = deque()
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=MAX_PROCESS) as executor:
            for game_idx in range(MAX_PROCESS * 2):
                # if game_idx % 2 == 0:
                #     main_player = 'B'
                # else:
                #     main_player = 'W'
                futures.append(executor.submit(self_play_buffer, cur=self.cur_pipes_TS, bot_2=self.cur_pipes,
                                               main_player=main_player, render=render))
            game_idx = 0
            while True:
                game_idx += 1
                data, value_white = futures.popleft().result()
                self.win_lose += value_white

                if game_idx <= 6000:
                    futures.append(executor.submit(self_play_buffer, cur=self.cur_pipes_TS, bot_2=self.cur_pipes,
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

def self_play_buffer(cur, bot_2, main_player='W', render=False) -> list:
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

    pipes = cur.pop()
    white = HivePlayer(pipes=pipes, reward=True)

    bot_ = bot_2.pop()
    black = HivePlayer(pipes=bot_, reward=True)
    state_policy_player = []

    while not board.game_is_over():
        if render:
            time.sleep(2)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    board.state.quit()
                    break
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
        # print(board.white_action_list)
        if main_player == 'W':
            if board.state.player() == 0:
                action, policy = white.action(deepcopy(board))
                player = 'W'
                policy = policy[0]
                # print("Search")
            else:
                policy_v, value = black.expand_and_evaluate(deepcopy(board))
                actions = board.actions()
                if len(actions) == 0:
                    action = -1
                else:
                    policy = policy_v[actions]
                    action = np.argmax(policy)
                    action = actions[action]
                    player = 'B'
                    policy = policy_v.tolist()

                    if board.state.turn <= 4:
                        action = np.random.choice(board.actions())
        else:
            if board.state.player() == 1:
                action, policy = black.action(deepcopy(board))
                player = 'B'
                policy = policy[0]
            else:
                # print(len(board.history_white), board.state.player())

                policy_v, value = white.expand_and_evaluate(deepcopy(board))
                actions = board.actions()
                if len(actions) == 0:
                    action = -1
                else:
                    policy = policy_v[actions]
                    action = np.argmax(policy)
                    action = actions[action]
                    player = 'W'
                    policy = policy_v.tolist()
                    if board.state.turn <= 4:
                        action = np.random.choice(board.actions())

        # print(len(state_policy_player))
        state = board.encode_board(player)
        if player == main_player:
            state_policy_player.append([state.tolist(), policy, player])

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
    print(value_white, board.state.turn, main_player)
    white.finish_game(value_white)
    black.finish_game(-value_white)

    data = []
    for state, policy, player in state_policy_player:
        if player == "W":
            value = value_white
        elif player == "B":
            value = value_white * -1
        if value_white == 0:
            value = -1
        data.append([state,  policy, value])

    cur.append(pipes)
    bot_2.append(bot_)

    return data, [value_white]

def main(worker, render=False):
    worker.start(main_player="W", render=render)


if __name__ == '__main__':
    os.chdir("../")
    import sys
    sys.setrecursionlimit(20000)
    print(sys.getrecursionlimit())

    mp.set_start_method('spawn')

    print(os.getcwd())
    worker = SelfPlayWorker()
    try:
        main(worker, render=True)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            worker.flush_buffer()
            sys.exit(0)
        except SystemExit:
            os._exit(0)



