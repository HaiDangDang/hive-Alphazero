#!/usr/bin/python
# -*- coding: utf-8 -*-
from tile import Tile, initialize_grid, draw_drag
from move_checker import is_valid_move, game_is_over, \
    move_does_not_break_hive
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


str_dict = {str(Queen): "Q",
            str(Beetle): "B",
            str(Spider): "S",
            str(Grasshopper): "G",
            str(Ant): "A"}
class GamePlay():

    def __init__(self, HEIGHT_MAP, WIDTH_MAP, second_force=False):

        # pg.font.init()
        self.next_move_tiles = []
        # self.state = Game_State(initialize_grid(HEIGHT - 300, WIDTH - 150, radius=20), render=True)
        # self.state = Game_State(initialize_grid(HEIGHT_MAP, WIDTH_MAP, radius=20), render=True)

        self.board_matrix = np.empty((MAX_MAP_FULL, MAX_MAP_FULL), dtype=object)
        self.white_pieces_set = {}
        self.black_pieces_set = {}
        self.encoded_action = None

        self.history_black = []
        self.history_white = []
        self.added = False
        self.latest_pos = {'B': [], 'W': []}

        self.state_key = ""

        self.HEIGHT_MAP = HEIGHT_MAP
        self.WIDTH_MAP = WIDTH_MAP
        self.stack_moves = []
        self.black_action_list = None
        self.white_action_list = None

        self.add_history = True
        self.pieces_keys = {}

        self.state_final = []
        self.new_game()
        self.second_force = True

    def game_is_over(self):
        return game_is_over(self.state)

    def new_game(self):
        # self.state = Game_State(initialize_grid(HEIGHT - 300, WIDTH - 150, radius=20), render=True)
        self.state = Game_State(initialize_grid(self.HEIGHT_MAP, self.WIDTH_MAP, radius=30), render=True)

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
                            self.white_pieces_set[key] = [tile, 0, tile.pieces[-1]]
                            self.pieces_keys[tile.pieces[-1]] = str_dict[str(piece_type)] + str(i)
                            break
                else:
                    for i in range(3):
                        key = str(piece_type) + str(i)
                        if key not in self.black_pieces_set:
                            self.black_pieces_set[key] = [tile, 0, tile.pieces[-1]]
                            self.pieces_keys[tile.pieces[-1]] = str(str_dict[str(piece_type)] + str(i)).lower()
                            break

            if tile.axial_coords != (99,99):
                self.board_matrix[tile.index_xy[0], tile.index_xy[1]] = tile
                state_key += "."

        state_key += str(self.state.player())
        self.state_key = state_key

        self.encoded_action = self.pre_actions()
        self.state_final = self.make_state_value()

    def move(self, move, with_skip=False):
        if move == -1:
            self.state.next_turn()
            self.state_key = self.state_key[:-1]
            self.state_key += str(self.state.player())
            # print("cccc")
        else:
            key_list = list(self.white_pieces_set.keys())
            map = np.zeros(MAX_MAP_FULL * MAX_MAP_FULL * 11)

            map[move] = 1
            map = map.reshape((MAX_MAP_FULL, MAX_MAP_FULL,11))
            poss = np.where(map == 1)
            # print(poss)
            piece = key_list[poss[2][-1]]
            end_tile = self.board_matrix[poss[0][-1], poss[1][-1]]

            if self.state.player() == 0:
                start_tile = self.white_pieces_set[piece][0]
                piece_type = self.white_pieces_set[piece][2]
                self.white_pieces_set[piece] = [end_tile, len(end_tile.pieces), piece_type]

            else:
                start_tile = self.black_pieces_set[piece][0]
                piece_type = self.black_pieces_set[piece][2]

                self.black_pieces_set[piece] = [end_tile, len(end_tile.pieces), piece_type]


            self.state.add_moving_piece(start_tile.pieces[-1])
            # check = is_valid_move(self.state, start_tile, end_tile)
            # if not check:
                # print(move)
                # print(piece)
                # print(start_tile.axial_coords)
                # print(end_tile.pieces)
                # print(start_tile)
                # print(end_tile)
                # if self.state.player() == 0:
                #     actions_list = self.get_actions(self.white_pieces_set)
                # else:
                #     actions_list = self.get_actions(self.black_pieces_set)
                # print(end_tile in self.white_action_list[piece], end_tile in self.black_action_list[piece])
                # print(move in self.encoded_action)
                # print("FAIL", piece, start_tile.core_index, end_tile.core_index, self.state.player())
            # assert check
            self.state.remove_moving_piece()
            self.add_history = True
            start_tile.move_piece(end_tile)
            self.state.next_turn()

            self.next_move_tiles = []
            state_key = ""
            for tile in self.state.board_tiles:
                if tile.axial_coords != (99, 99):
                    if tile.has_pieces():
                        for adjacent_tile in tile.adjacent_tiles:
                            if not adjacent_tile.has_pieces() and adjacent_tile not in self.next_move_tiles:
                                if self.state.turn == 2:
                                    if adjacent_tile.core_index == ('M', '13'):
                                        self.next_move_tiles.append(adjacent_tile)
                                else:
                                    self.next_move_tiles.append(adjacent_tile)
                        for piece in tile.pieces:
                            state_key += self.pieces_keys[piece]
                    else:
                        state_key += "."
            state_key += str(self.state.player())

            self.state_key = state_key

        self.encoded_action = self.pre_actions()
        self.state_final = self.make_state_value()

        # if with_skip:
        #     if not self.added:
        #         print("ccc")
        #         self.state.next_turn()
        #         self.encoded_action = self.pre_actions()

        # if move == 2992:
        #     self.state.next_turn()

    def actions(self):
        return self.encoded_action

    def human_play(self):
        self.add_history = True
        self.next_move_tiles = []
        for tile in self.state.board_tiles:
            if tile.axial_coords != (99, 99) and tile.has_pieces():
                for adjacent_tile in tile.adjacent_tiles:
                    if not adjacent_tile.has_pieces() and adjacent_tile not in self.next_move_tiles:
                        self.next_move_tiles.append(adjacent_tile)
        self.encoded_action = self.pre_actions()
        self.state_final = self.make_state_value()

    def pre_actions(self):
        self.added = False
        if self.state.player() == 0:
            actions_list = self.get_actions(self.white_pieces_set)
            self.white_action_list = actions_list
        else:
            actions_list = self.get_actions(self.black_pieces_set)
            self.black_action_list = actions_list

        return self.encode_action(actions_list)

    def get_actions(self, piece_set):
        actions_list = {}
        on_board = []
        for piece, value in piece_set.items():
            tile = value[0]
            level = value[1]
            if level + 1 == len(tile.pieces):
                tiles = []
                actions_list[piece] = []
                self.state.add_moving_piece(tile.pieces[-1])
                if tile.axial_coords == (99, 99):
                    if piece[:-1] not in on_board:
                        on_board.append(piece[:-1])
                        for new_tile in self.next_move_tiles:
                            if is_valid_move(self.state, tile, new_tile):
                                if new_tile not in tiles:
                                    actions_list[piece].append(new_tile)
                                    tiles.append(new_tile)
                                    self.added = True
                else:
                    if move_does_not_break_hive(self.state, tile):
                        if piece[:-1] not in [str(Beetle), str(Queen)]:
                            for new_tile in self.next_move_tiles:
                                if is_valid_move(self.state, tile, new_tile,
                                                 check_move_break_hive=False):
                                    if new_tile not in tiles:
                                        actions_list[piece].append(new_tile)
                                    tiles.append(new_tile)
                                    self.added = True
                        else:
                            for adjacent_tile in tile.adjacent_tiles:
                                if adjacent_tile not in tiles:
                                    if is_valid_move(self.state, tile, adjacent_tile,
                                                     check_move_break_hive=False):
                                        actions_list[piece].append(adjacent_tile)
                                        tiles.append(adjacent_tile)
                                        self.added = True
                self.state.remove_moving_piece()
        # if self.added == False:
        #     actions_list = {}
        #     on_board = []
        #     for piece, value in piece_set.items():
        #         tile = value[0]
        #         level = value[1]
        #         if level + 1 == len(tile.pieces):
        #             tiles = []
        #             actions_list[piece] = []
        #             self.state.add_moving_piece(tile.pieces[-1])
        #             if tile.axial_coords == (99, 99):
        #                 if piece[:-1] not in on_board:
        #                     for new_tile in self.next_move_tiles:
        #                         if is_valid_move(self.state, tile, new_tile):
        #                             if new_tile not in tiles:
        #                                 actions_list[piece].append(new_tile)
        #                                 tiles.append(new_tile)
        #                                 self.added = True
        #             else:
        #                 if move_does_not_break_hive(self.state, tile):
        #                     if piece[:-1] not in [str(Beetle), str(Queen)]:
        #                         for new_tile in self.next_move_tiles:
        #                             if is_valid_move(self.state, tile, new_tile,
        #                                              check_move_break_hive=False):
        #                                 if new_tile not in tiles:
        #                                     actions_list[piece].append(new_tile)
        #                                 tiles.append(new_tile)
        #                                 self.added = True
        #                     else:
        #                         for adjacent_tile in tile.adjacent_tiles:
        #                             if adjacent_tile not in tiles:
        #                                 if is_valid_move(self.state, tile, adjacent_tile,
        #                                                  check_move_break_hive=False):
        #                                     actions_list[piece].append(adjacent_tile)
        #                                     tiles.append(adjacent_tile)
        #                                     self.added = True
        #             self.state.remove_moving_piece()
        #     if self.added:
        #         print("CCCCCCCCCCC")
        # print(on_board)
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

        return encoded_action.tolist()

    def encode_board(self, player="N"):
        if player == "N":
            if self.state.player() == 0:
                player = "W"
            else:
                player = "B"

        if player in self.state_final:
            return self.state_final[player]
        else:
            print("ERROR", self.state.turn, player)
            return self.state_final[player]
            return self.make_state_value(player)[player]

    def make_state_value(self, player="N"):
        max_column = MAX_MAP_FULL
        max_row = MAX_MAP_FULL
        encoder_dict = {str(Queen) + "0": 0,
                        str(Beetle) + "0": 1, str(Beetle) + "1": 2,
                        str(Spider) + "0": 3 , str(Spider)+ "1": 4 ,
                        str(Grasshopper) + "0": 5, str(Grasshopper) + "1": 6, str(Grasshopper) + "2": 7,
                        str(Ant) + "0": 8, str(Ant) + "1": 9, str(Ant) + "2": 10}

        boards = np.zeros([max_row, max_column, STATE_FEATURES])
        # boards[:, :, 56] = self.state.player()
        boards[:, :, 31] = self.state.turn
        if player == "N":
            if self.state.player() == 0:
                player = "W"
            else:
                player = "B"

        if player == "W":
            white_set = self.white_pieces_set
            black_set = self.black_pieces_set
            history = self.history_white
            white_list = self.white_action_list
        else:
            white_set = self.black_pieces_set
            black_set = self.white_pieces_set
            history = self.history_black
            white_list = self.black_action_list

        lock_black, queen_move_black = self.mini_black_actions(player)
        # lock_black, queen_move_black = [], {}

        for piece, value in white_set.items():
            tile = value[0]
            if tile.axial_coords != (99, 99):
                poss = np.where(self.board_matrix == tile)
                boards[poss[0][0], poss[1][0], encoder_dict[piece]] = 1
                boards[poss[0][0], poss[1][0], 11] = 1
                boards[poss[0][0], poss[1][0], 30] = 1
                if piece[:-1] == str(Queen):

                    index_counter = 0
                    for adjacent_tile in tile.adjacent_tiles:
                        poss_2 = np.where(self.board_matrix == adjacent_tile)
                        if adjacent_tile.has_pieces():
                            boards[poss_2[0][0], poss_2[1][0], 32] = 1
                        else:
                            if adjacent_tile in queen_move_black:
                                for poss_3 in queen_move_black[adjacent_tile]:
                                    boards[poss_3[0][0], poss_3[1][0], 44 + index_counter] = 1
                        index_counter += 1

                if white_list is not None:
                    if piece in white_list:
                        if len(white_list[piece]) == 0:
                            boards[poss[0][0], poss[1][0], 34] = 1
                    else:
                        boards[poss[0][0], poss[1][0], 34] = 1

                if piece[:-1] == str(Beetle):
                    if len(tile.pieces) >= 3:
                        if tile.pieces[2] == value[2]:
                            boards[poss[0][0], poss[1][0], 24] = 1
                    if len(tile.pieces) >= 4:
                        if tile.pieces[3] == value[2]:
                            boards[poss[0][0], poss[1][0], 25] = 1
                    if len(tile.pieces) == 5:
                        if tile.pieces[-1] == value[2]:
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

                    index_counter = 0
                    for adjacent_tile in tile.adjacent_tiles:
                        poss_2 = np.where(self.board_matrix == adjacent_tile)
                        if adjacent_tile.has_pieces():
                            boards[poss_2[0][0], poss_2[1][0], 33] = 1
                        else:
                            if white_list is not None:
                                for piece_2, actions in white_list.items():
                                    if adjacent_tile in actions:
                                        tile_2 = white_set[piece_2][0]
                                        if tile_2.axial_coords != (99, 99):
                                            poss_3 = np.where(self.board_matrix == tile_2)
                                            # print(piece_2, tile_2.axial_coords, player)
                                            boards[poss_3[0][0], poss_3[1][0], 50 + index_counter] = 1
                        index_counter += 1

                if piece in lock_black:
                    boards[poss[0][0], poss[1][0], 35] = 1

                if piece[:-1] == str(Beetle):
                    if len(tile.pieces) >= 3:
                        if tile.pieces[2] == value[2]:
                            boards[poss[0][0], poss[1][0], 27] = 1
                    if len(tile.pieces) >= 4:
                        if tile.pieces[3] == value[2]:
                            boards[poss[0][0], poss[1][0], 28] = 1
                    if len(tile.pieces) == 5:
                        if tile.pieces[-1] == value[2]:
                            boards[poss[0][0], poss[1][0], 29] = 1

        for i in range(len(history)):
            boards[:, :, 36+i*2: 38+i*2] = history[i]
            if i == 3:
                break

        if player == "W":
            if self.add_history:
                self.add_history = False
                self.history_white.insert(0, boards[:, :, [11, 11 + extra_black]])
                self.history_white = self.history_white[:6]
        else:
            if self.add_history:
                self.add_history = False
                self.history_black.insert(0, boards[:, :, [11, 11 + extra_black]])
                self.history_black = self.history_black[:6]

        return {player: boards}

    def mini_black_actions(self, player):
        if player == "W":
            black_set = self.black_pieces_set
            white_set = self.white_pieces_set

        else:
            black_set = self.white_pieces_set
            white_set = self.black_pieces_set
        lock = []
        queen_move = {}
        queen_tile = white_set[str(Queen) + "0"][0]
        for piece, value in black_set.items():
            tile = value[0]
            if tile.axial_coords != (99, 99):
                level = value[1]
                if level + 1 == len(tile.pieces):
                    self.state.add_moving_piece(tile.pieces[-1])
                    tile.remove_piece()
                    check_remove = move_does_not_break_hive_local(self.state)
                    tile.add_piece(value[2])
                    if check_remove:
                        for adjacent_tile in queen_tile.adjacent_tiles:
                            if not adjacent_tile.has_pieces():
                                if is_valid_move(self.state, tile, adjacent_tile,
                                                 check_move_break_hive=False):
                                    poss = np.where(self.board_matrix == tile)
                                    if len(poss) != 0:
                                        if adjacent_tile not in queen_move:
                                            queen_move[adjacent_tile] = []
                                        queen_move[adjacent_tile].append(poss)
                    else:
                        lock.append(piece)
                    self.state.remove_moving_piece()
                else:
                    lock.append(piece)

        return lock, queen_move

    def turn(self):
        return self.state.turn

    def player(self):
        return self.state.player()

    def skip_turn(self):
        self.state.next_turn()
        self.encoded_action = self.pre_actions()
        self.state_final = self.make_state_value()

    def decode_action(self, action):
        key_list = list(self.white_pieces_set.keys())

        map = np.zeros(MAX_MAP_FULL * MAX_MAP_FULL * 11)
        map[action] = 1
        map = map.reshape((MAX_MAP_FULL, MAX_MAP_FULL, 11))
        poss = np.where(map == 1)
        end_tile = self.board_matrix[poss[0][-1], poss[1][-1]]
        piece = key_list[poss[2][-1]]
        return piece, end_tile.core_index

def move_does_not_break_hive_local(state):
    tile_list = state.get_tiles_with_pieces()
    visited = []
    queue = []
    if len(tile_list) == 0:
        return False
    visited.append(tile_list[0])
    queue.append(tile_list[0])

    while queue:
        current_tile = queue.pop(0)

        for neighbor_tile in [x for x in current_tile.adjacent_tiles
                              if x.has_pieces()]:
            if neighbor_tile not in visited:
                visited.append(neighbor_tile)
                queue.append(neighbor_tile)

    if len(visited) != len(tile_list):
        return False
    else:
        return True
