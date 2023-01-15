import gym
from gym import spaces

import pygame as pg
from tile import Tile, initialize_grid, draw_drag
from move_checker import is_valid_move, game_is_over, \
    player_has_no_moves
from menus import start_menu, end_menu, no_move_popup
from game_state import Game_State
from inventory_frame import Inventory_Frame
from turn_panel import Turn_Panel
from settings import BACKGROUND, WIDTH, HEIGHT
import numpy as np
from copy import deepcopy
"""
x0
000
x0

00
x00
00
"""
class HiveBoard():
    def __init__(self, height=4, weight=7):
        self.board = np.zeros((height, weight)).astype(str)
        self.board[self.board == "0.0"] = " "

        self.encoder_dict = {"Q":0, "A":1, "B":2, "S":3, "G":4,
                             "q":5, "a":6, "b":7, "s":8, "g":9}

        self.white_pieces = {"Q":1, "A":3, "B":2, "S":2, "G":3}
        self.black_pieces = {"q":1, "a":3, "b":2, "s":2, "g":3}

        self.center_row = int(height/2) + 1
        self.center_column = int(weight/2) + 1

        self.even_neighbor = [[0, -1], [0, 1],[-1, -1], [-1, -2],
                                [1, -1], [1, -2]]
        self.odd_neighbor = [[0, -1], [0, 1],[-1, -1], [-1, 0],
                                [1, -1], [1, 0]]


        self.center_point = tuple((self.center_row, self.center_column))
        self.tile_has_piece = []
        self.valid_tiles = [self.center_point]

        # self.find_valid_tile(self.center_row, self.center_column)

        self.running = True
        self.menu_loop = True
        self.main_loop = False
        self.end_loop = False
        self.play_new_game = False
        self.move_popup_loop = False

        #

        self.moving_piece = None
        self.turn = 1

        # other

        self.winner = None

    def find_valid_tiles(self):
        valid_tiles = []
        for tile in self.tile_has_piece:
            if tile[0]%2 == 0:
                next_tiles = np.array(self.even_neighbor) + list(tile)
            else:
                next_tiles = np.array(self.odd_neighbor) + list(tile)

            next_tiles = next_tiles[(next_tiles >= 0).all(axis=1)]

            for tile_valid in next_tiles:
                t = tuple(tile_valid)
                if t not in self.tile_has_piece:
                    valid_tiles.append(t)
        self.valid_tiles = valid_tiles

    def start_game(self):
        self.menu_loop = False
        self.main_loop = True

    def end_game(self):
        self.main_loop = False
        self.end_loop = True

    def new_game(self):
        self.main_loop = True
        self.end_loop = False

        self.turn = 1

    # def get_tiles_with_pieces(self, include_inventory=False):
    #     tiles = []
    #     for tile in self.init_board:
    #         if include_inventory:
    #             if tile != " ":
    #                 tiles.append(tile)
    #         elif tile != " " and type(tile) is not Inventory_Tile:
    #             tiles.append(tile)
    #     return tiles

    def remove_piece(self, pos):
        self.tile_has_piece.remove(pos)
        piece = self.board[pos[0], pos[1]]
        if len(piece) == 1:
            self.board[pos[0], pos[1]]= " "
        else:
            self.board[pos[0], pos[1]] = self.board[pos[0], pos[1]][1:]

    def add_piece(self, pos, piece):
        self.tile_has_piece.append(pos)
        if len(piece) == 1:
            self.board[pos[0], pos[1]] = piece
        else:
            self.board[pos[0], pos[1]] = piece + self.board[pos[0], pos[1]]

    def move_does_not_break_hive(self):
        move_set = []
        for tile in self.tile_has_piece:
            temp_tile_set = deepcopy(self.tile_has_piece)
            temp_tile_set.remove(tile)

            visited = [temp_tile_set[0]]
            queue = [temp_tile_set[0]]

            while queue:
                current_tile = queue.pop(0)

                if current_tile[0]%2 == 0:
                    next_tiles = np.array(self.even_neighbor) + list(current_tile)
                else:
                    next_tiles = np.array(self.odd_neighbor) + list(current_tile)

                next_tiles = next_tiles[(next_tiles >= 0).all(axis=1)]

                for neighbor_tile in next_tiles:
                    if neighbor_tile in temp_tile_set:
                        if neighbor_tile not in visited:
                            visited.append(neighbor_tile)
                            queue.append(neighbor_tile)

            if len(visited) == len(temp_tile_set):
                move_set.append(tile)



x = HiveBoard()
a = x.valid_tiles
(12,10) in a
a = tuple((1,2))
a = list(a)