#!/usr/bin/python
# -*- coding: utf-8 -*-
from tile import Tile, initialize_grid, draw_drag
from move_checker import is_valid_move, game_is_over, \
    player_has_no_moves
from menus import start_menu, end_menu, no_move_popup
from game_state import Game_State
from inventory_frame import Inventory_Frame
from turn_panel import Turn_Panel
from settings import BACKGROUND, WIDTH, HEIGHT, PIECE_WHITE, PIECE_BLACK, BLUE
from pieces import Queen, Grasshopper, Spider, Beetle, Ant
import numpy as np
import time
from copy import deepcopy
from hive_engine.config import MAX_MAP_FULL, STATE_FEATURES
import copy


from scipy.ndimage import label

board_matrix = np.zeros((MAX_MAP_FULL, MAX_MAP_FULL), dtype=str)
len_board_matrix = np.vectorize(len)


piece_sets = {'W': {"Q": (-1, -1),
                    "A1": (-1, -1), "A2": (-1, -1), "A3": (-1, -1),
                    "B1": (-1, -1), "B2": (-1, -1),
                    "G1": (-1, -1), "G2": (-1, -1), "G3": (-1, -1),
                    "S1": (-1, -1), "S2": (-1, -1)},
              'B': {"q": (-1, -1),
                    "a1": (-1, -1), "a2": (-1, -1), "a3": (-1, -1),
                    "b1": (-1, -1), "b2": (-1, -1),
                    "g1": (-1, -1), "g2": (-1, -1), "g3": (-1, -1),
                    "s1": (-1, -1), "s2": (-1, -1)}}

board_matrix[0,0] = "G"
x,y = label(mylen(board_matrix))

x.legal_action = []

str_dict = {str(Queen): "Q",
                str(Beetle): "B",
                str(Spider): "S", str(Grasshopper): "G", str(Ant): "A"}

class Hive_Board():

    def __init__(self, HEIGHT_MAP, WIDTH_MAP, COPY_ROOT=False):

        self.board_matrix = np.zeros((MAX_MAP_FULL, MAX_MAP_FULL))
        self.pieces_set  =

    def game_is_over(self):
        return game_is_over(self.state)

    def new_game(self):
        # self.state = Game_State(initialize_grid(HEIGHT - 300, WIDTH - 150, radius=20), render=True)
        self.state = Game_State(initialize_grid(self.HEIGHT_MAP, self.WIDTH_MAP, radius=20), render=True)

        self.next_move_tiles = []
        for tile in self.state.board_tiles:
            if tile.color == BLUE:
                self.next_move_tiles.append(tile)
                break
        state_key = ""
        for tile in self.state.board_tiles:
            if tile.has_pieces():
                piece_type = type(tile.pieces[-1])
                if tile.pieces[-1].color == PIECE_WHITE:
                    for i in range(3):
                        key = str(piece_type) + str(i)
                        if key not in self.white_pieces_set:
                            self.white_pieces_set[key] = [tile, 0]
                            break
                else:
                    for i in range(3):
                        key = str(piece_type) + str(i)
                        if key not in self.black_pieces_set:
                            self.black_pieces_set[key] = [tile, 0]
                            break

            if tile.axial_coords != (99,99):
                self.board_matrix[tile.index_xy[0], tile.index_xy[1]] = tile
                state_key += "."

        self.state_key = state_key
        self.encoded_action = self.pre_actions()

    def move(self, move, with_skip=True):

        key_list = list(self.white_pieces_set.keys())

        map = np.zeros(MAX_MAP_FULL * MAX_MAP_FULL * 11)

        map[move] = 1
        map = map.reshape((MAX_MAP_FULL, MAX_MAP_FULL,11))
        poss = np.where(map == 1)
        piece = key_list[poss[2][-1]]
        end_tile = self.board_matrix[poss[0][-1], poss[1][-1]]

        if self.state.player() == 0:
            start_tile = self.white_pieces_set[piece][0]
            self.white_pieces_set[piece] = [end_tile, len(end_tile.pieces)]
            if start_tile.axial_coords != (99, 99):
                self.latest_pos['W'] = (piece, start_tile)
        else:
            start_tile = self.black_pieces_set[piece][0]
            self.black_pieces_set[piece] = [end_tile, len(end_tile.pieces)]
            if start_tile.axial_coords != (99, 99):
                self.latest_pos['B'] = (piece, start_tile)

        self.state.add_moving_piece(start_tile.pieces[-1])
        # check = is_valid_move(self.state, start_tile, end_tile)
        # if not check:
        #     print(move)
        #     print(piece)
        #     print(start_tile.pieces)
        #     print(end_tile.pieces)
        #     print(start_tile)
        #     print(end_tile)
        #     if self.state.player() == 0:
        #         actions_list = self.get_actions(self.white_pieces_set)
        #     else:
        #         actions_list = self.get_actions(self.black_pieces_set)
        #
        #     print(actions_list)
        # assert check
        self.state.remove_moving_piece()

        start_tile.move_piece(end_tile)
        self.state.next_turn()

        self.next_move_tiles = []
        state_key = ""
        for tile in self.state.board_tiles:
            if tile.axial_coords != (99, 99):
                if tile.has_pieces():
                    for adjacent_tile in tile.adjacent_tiles:
                        if not adjacent_tile.has_pieces() and adjacent_tile not in self.next_move_tiles:
                            self.next_move_tiles.append(adjacent_tile)
                    for piece in tile.pieces:
                        piece_type = type(piece)
                        piece_type = str_dict[str(piece_type)]
                        if piece.color == PIECE_BLACK:
                            piece_type = piece_type.lower()
                        state_key += piece_type
                else:
                    state_key += "."
        self.state_key = state_key

        self.encoded_action = self.pre_actions()
        if with_skip:
            if not self.added:
                self.state.next_turn()
                self.encoded_action = self.pre_actions()

        # if move == 2992:
        #     self.state.next_turn()

    def actions(self):
        return self.encoded_action

    def human_play(self):
        self.next_move_tiles = []
        for tile in self.state.board_tiles:
            if tile.axial_coords != (99, 99) and tile.has_pieces():
                for adjacent_tile in tile.adjacent_tiles:
                    if not adjacent_tile.has_pieces() and adjacent_tile not in self.next_move_tiles:
                        self.next_move_tiles.append(adjacent_tile)
        self.encoded_action = self.pre_actions()

    def pre_actions(self):
        self.added = False
        if self.state.player() == 0:
            actions_list = self.get_actions(self.white_pieces_set)
        else:
            actions_list = self.get_actions(self.black_pieces_set)

        return self.encode_action(actions_list)


    def get_actions(self, piece_set):
        actions_list = {}
        for piece, value in piece_set.items():
            tile = value[0]
            level = value[1]
            if level + 1 == len(tile.pieces):
                tiles = []
                actions_list[piece] = []
                self.state.add_moving_piece(tile.pieces[-1])
                if (tile.axial_coords == (99, 99) or
                        piece[:-1] not in [str(Beetle)]):
                    for new_tile in self.next_move_tiles:
                        if is_valid_move(self.state, tile, new_tile):
                            if new_tile not in tiles:
                                actions_list[piece].append(new_tile)
                                tiles.append(new_tile)
                                self.added = True
                elif (tile.axial_coords != (99, 99) and
                      piece[:-1] in [str(Beetle)]):
                    for adjacent_tile in tile.adjacent_tiles:
                        if adjacent_tile not in tiles:
                            if is_valid_move(self.state, tile, adjacent_tile):
                                actions_list[piece].append(adjacent_tile)
                                tiles.append(adjacent_tile)
                                self.added = True

                self.state.remove_moving_piece()
        return actions_list

    def encode_action(self, action_list):
        action_on_board = np.zeros([MAX_MAP_FULL, MAX_MAP_FULL, 11])
        # action_skip = np.zeros([1])
        key_list = list(self.white_pieces_set.keys())
        # if noTurn:
        #     action_skip = np.ones([1])
        #
        #     return [2992]
        # else:
        for piece, tiles in action_list.items():
            index = np.where(np.array(key_list) == piece)[0][0]
            for tile in tiles:
                x, y = np.where(self.board_matrix == tile)
                action_on_board[x[-1], y[-1], index] = 1
        encoded_action = action_on_board.reshape(-1)
        encoded_action = np.where(encoded_action == 1)[0]
        # if len(encoded_action) == 0:
        #     print("ENDING")
        return encoded_action.tolist()

    def encode_board(self, player="N"):
        max_column = MAX_MAP_FULL
        max_row = MAX_MAP_FULL
        encoder_dict = {str(Queen) + "0": 0,
                        str(Beetle) + "0": 1, str(Beetle) + "1": 2,
                        str(Spider) + "0": 3 , str(Spider)+ "1": 4 ,
                        str(Grasshopper) + "0": 5, str(Grasshopper) + "1": 6, str(Grasshopper) + "2": 7,
                        str(Ant) + "0": 8, str(Ant) + "1": 9, str(Ant) + "2": 10}

        boards = np.zeros([max_row, max_column, STATE_FEATURES])
        #0 - 11 : white pieces
        #12 - 23 : black pieces
        #32: players
        #33-34 : beetle lvl 3
        #35-36 : beetle lvl 4
        #37-38 : beetle lvl 5
        #39: turn
        boards[:, :, 31] = self.state.turn
        if player == "N":
            if self.state.player() == 0:
                player = "W"
            else:
                player = "B"
        if player == "W":
            white_set = self.white_pieces_set
            black_set = self.black_pieces_set
            not_player = 'B'
        else:
            white_set = self.black_pieces_set
            black_set = self.white_pieces_set
            not_player = 'W'

        if len(self.latest_pos[player]) != 0:
            tile = self.latest_pos[player][1]
            piece = self.latest_pos[player][0]
            poss = np.where(self.board_matrix == tile)
            boards[poss[0][0], poss[1][0], encoder_dict[piece]] = 0.5

        if len(self.latest_pos[not_player]) != 0:
            tile = self.latest_pos[not_player][1]
            piece = self.latest_pos[not_player][0]
            poss = np.where(self.board_matrix == tile)
            boards[poss[0][0], poss[1][0], encoder_dict[piece]] = 0.5

        for piece, value in white_set.items():
            tile = value[0]
            if tile.axial_coords != (99, 99):

                poss = np.where(self.board_matrix == tile)
                boards[poss[0][0], poss[1][0], encoder_dict[piece]] = 1
                boards[poss[0][0], poss[1][0], 11] = 1
                boards[poss[0][0], poss[1][0], 30] = 1
                if piece[:-1] == str(Queen):
                    for adjacent_tile in tile.adjacent_tiles:
                        poss_2 = np.where(self.board_matrix == adjacent_tile)
                        boards[poss_2[0][0], poss_2[1][0], 32] = 1

                if piece[:-1] == str(Beetle):
                    if len(tile.pieces) >= 3:
                        if tile.pieces[2] == PIECE_WHITE:
                            boards[poss[0][0], poss[1][0], 24] = 1
                    if len(tile.pieces) >= 4:
                        if tile.pieces[3] == PIECE_WHITE:
                            boards[poss[0][0], poss[1][0], 25] = 1
                    if len(tile.pieces) == 5:
                        if tile.pieces[-1] == PIECE_WHITE:
                            boards[poss[0][0], poss[1][0], 26] = 1

        extra_black = 12
        for piece, value in black_set.items():
            tile = value[0]
            if tile.axial_coords != (99, 99):
                poss = np.where(self.board_matrix == tile)
                boards[poss[0][0], poss[1][0], encoder_dict[piece] + extra_black] = 1
                boards[poss[0][0], poss[1][0], 11 + extra_black] = 1
                boards[poss[0][0], poss[1][0], 30] = 1
                if piece[:-1] == str(Queen):
                    for adjacent_tile in tile.adjacent_tiles:
                        poss_2 = np.where(self.board_matrix == adjacent_tile)
                        boards[poss_2[0][0], poss_2[1][0], 33] = 1

                if piece[:-1] == str(Beetle):
                    if len(tile.pieces) >= 3:
                        if tile.pieces[2] == PIECE_BLACK:
                            boards[poss[0][0], poss[1][0], 27] = 1
                    if len(tile.pieces) >= 4:
                        if tile.pieces[3] == PIECE_BLACK:
                            boards[poss[0][0], poss[1][0], 28] = 1
                    if len(tile.pieces) == 5:
                        if tile.pieces[-1] == PIECE_BLACK:
                            boards[poss[0][0], poss[1][0], 29] = 1

        for i in range(len(self.history)):
            boards[:, :, 34+i*2: 36+i*2] = self.history[i]
            if i == 4:
                break
        self.history.insert(0, boards[:, :, [11, 11 + extra_black]])
        self.history = self.history[:6]
        return boards

    def state_to_str(self):
        state_key = ""
        for tile in self.state.board_tiles:
            if tile.has_pieces():
                for piece in tile.piece:
                    piece_type = type(piece)
                    piece_type = str(piece_type)[0]
                    state_key += piece_type
            else:
                state_key += "0"
        return state_key

    def copy(self):

        env = copy.copy(self)
        tiles = [copy.copy(x) for x in self.state.board_tiles]
        state_new = Game_State(tiles=tiles, render=False)
        env.state = copy.copy(self.state)
        env.white_pieces_set = copy.copy(self.white_pieces_set)
        env.black_pieces_set = copy.copy(self.black_pieces_set)
        env.state = state_new
        # print("CCc")
        return env

    def turn(self):
        return self.state.turn

    def player(self):
        return self.state.player()

    def skip_turn(self):
        self.state.next_turn()
        self.encoded_action = self.pre_actions()

