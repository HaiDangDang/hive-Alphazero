#!/usr/bin/python
# -*- coding: utf-8 -*-
import pygame as pg
from tile import Tile, initialize_grid, draw_drag
from move_checker import is_valid_move, game_is_over, \
    player_has_no_moves
from menus import start_menu, end_menu, no_move_popup
from game_state import Game_State
from inventory_frame import Inventory_Frame
from turn_panel import Turn_Panel
import sys

from settings import BACKGROUND, WIDTH, HEIGHT, PIECE_WHITE, PIECE_BLACK, BLUE
from pieces import Queen, Grasshopper, Spider, Beetle, Ant
import numpy as np
import pickle
import string
import os

from hive_engine.config import MAX_MAP_FULL, MAX_MAP_HAFT, index_char, index_number

database = "../datahist"


piece_types = ["G1", "G2", "G3", "A1", "A2", "A3", "S1", "S2", "B1", "B2", "Q", ]


# for idx, file in enumerate(os.listdir(database)):
#
#     file_name = os.path.join(database, file)
#     with open(file_name, 'rb') as f:
#         data_boards = pickle.load(f)

index = np.random.choice(len(os.listdir(database)))
# index = 2
file_name = os.path.join(database, os.listdir(database)[index])
with open(file_name, 'rb') as f:
    data_boards = pickle.load(f)

txt_file = open(os.path.join("../database", os.listdir(database)[index][:-3] + "txt"), 'r', encoding="ISO-8859-1")
contents = [line for line in txt_file.readlines()]

pg.font.init()
# Create the screen

screen = pg.display.set_mode((WIDTH, HEIGHT))
background = pg.Surface(screen.get_size())

# Title and Icon

pg.display.set_caption('Hive')
icon = pg.image.load('images/icon.png')
pg.display.set_icon(icon)

current_step = data_boards[0]
piece, x, y, player = current_step[0], current_step[1], current_step[2], current_step[3]
shift_up = index_char.index(x) - index_char.index("N")

y = index_number.index(y) + 1
x = index_char.index(x)

state = Game_State(initialize_grid(HEIGHT - 100, WIDTH - 500, radius=20, center_tile=[y,x]), render=True)
white_inventory = Inventory_Frame((0, 158), 0, white=True)
black_inventory = Inventory_Frame((700, 158), 1, white=False)

shift_top_bottom = ['Z', 'A']
shift_left_right = ['1', '26']


boards_W = {}
boards_B = {}
for p in piece_types:
    boards_B[p] = None
    boards_W[p] = None


board_matrix = np.empty((MAX_MAP_FULL, MAX_MAP_FULL), dtype=object)

for tile in state.board_tiles:
    if tile.has_pieces():
        piece_type = type(tile.pieces[-1])
        if tile.pieces[-1].color == PIECE_WHITE:
            for i in range(3):
                key = str(piece_type)[15] + str(i + 1)
                if piece_type == Queen:
                    key = str(piece_type)[15]
                if boards_W[key] is None:
                    boards_W[key] = tile
                    break
        else:
            for i in range(3):
                key = str(piece_type)[15] + str(i +1)
                if piece_type == Queen:
                    key = str(piece_type)[15]
                if boards_B[key] is None:
                    boards_B[key] = tile
                    break

    if tile.axial_coords != (99,99):
        board_matrix[tile.index_xy[0], tile.index_xy[1]] = tile

# top = 25
# bottom = 0
#
# for i in range(abs(shift_up)):
#     store_old = []
#
#     for j in range(26):
#         for columns in range(26):
#             if j == bottom:
#                 store_old.append(board_matrix[j, columns].coords)
#             if j != top:
#                 board_matrix[j, columns].change_coords(board_matrix[(j + 1)%26, columns].coords)
#             else:
#                 board_matrix[j, columns].change_coords(store_old[columns])



# axial_distance(board_matrix[12, 12].axial_coords, board_matrix[13, 14].axial_coords)



state.running = True
state.main_loop = True
counter = 0
player = "W"
# current_step = data_boards[0]
# piece, x, y = current_step[0], current_step[1], current_step[2]
# start_x = ""
# start_y = ""
while state.running:
    pos = pg.mouse.get_pos()
    background.fill(BACKGROUND)
    # white_inventory.draw(background, pos)
    # black_inventory.draw(background, pos)
    for tile in state.board_tiles:
        if state.clicked:
            tile.draw(background, pos, state.clicked)
            if tile.under_mouse(pos) and state.moving_piece \
                    is None and tile.has_pieces():
                state.add_moving_piece(tile.pieces[-1])
        else:
            tile.draw(background, pos)
    if state.moving_piece:
        draw_drag(background, pos, state.moving_piece)
    state.turn_panel.draw(background, state.turn)
    screen.blit(background, (0, 0))
    pg.display.flip()

    events = pg.event.get()
    for event in events:
        if event.type == pg.KEYUP:
            current_step = data_boards[counter]
            print(player, current_step)
            piece, x, y, player_ = current_step[0], current_step[1], current_step[2], current_step[3]
            y = index_number.index(y)
            x = index_char.index(x)
            end_tile = board_matrix[x, y]

            player = player_
            if player == "W":
                start_tile = boards_W[piece]
            else:
                start_tile = boards_B[piece]
            state.add_moving_piece(start_tile.pieces[-1])
            check = is_valid_move(state, start_tile, end_tile)
            # if not check:
            print(check)
            state.remove_moving_piece()
            start_tile.move_piece(end_tile)
            state.next_turn()

            if player == "W":
                boards_W[piece] = end_tile
                player = "B"
            else:
                boards_B[piece] = end_tile
                player = "W"
            counter += 1




for idx, file in enumerate(os.listdir(database)):
    print(idx)
    if idx <= 0:
        continue
    file_name = os.path.join(database, file)
    with open(file_name, 'rb') as f:
        data_boards = pickle.load(f)
    current_step = data_boards[0]
    piece, x, y, player = current_step[0], current_step[1], current_step[2], current_step[3]
    y = index_number.index(y) + 1
    x = index_char.index(x)

    state = Game_State(initialize_grid(HEIGHT - 100, WIDTH - 500, radius=20, center_tile=[y, x]), render=True)
    white_inventory = Inventory_Frame((0, 158), 0, white=True)
    black_inventory = Inventory_Frame((700, 158), 1, white=False)

    boards_W = {}
    boards_B = {}
    for p in piece_types:
        boards_B[p] = None
        boards_W[p] = None


    board_matrix = np.empty((MAX_MAP_FULL, MAX_MAP_FULL), dtype=object)

    for tile in state.board_tiles:
        if tile.has_pieces():
            piece_type = type(tile.pieces[-1])
            if tile.pieces[-1].color == PIECE_WHITE:
                for i in range(3):
                    key = str(piece_type)[15] + str(i + 1)
                    if piece_type == Queen:
                        key = str(piece_type)[15]
                    if boards_W[key] is None:
                        boards_W[key] = tile
                        break
            else:
                for i in range(3):
                    key = str(piece_type)[15] + str(i + 1)
                    if piece_type == Queen:
                        key = str(piece_type)[15]
                    if boards_B[key] is None:
                        boards_B[key] = tile
                        break

        if tile.axial_coords != (99, 99):
            board_matrix[tile.index_xy[0], tile.index_xy[1]] = tile

    player = "W"
    check = True
    for i in range(len(data_boards)):
        current_step = data_boards[i]
        piece, x, y, player = current_step[0], current_step[1], current_step[2], current_step[3]
        y = index_number.index(y)
        x = index_char.index(x)

        end_tile = board_matrix[x, y]

        if player == "W":
            start_tile = boards_W[piece]
        else:
            start_tile = boards_B[piece]

        state.add_moving_piece(start_tile.pieces[-1])
        check = is_valid_move(state, start_tile, end_tile)
        if not check:
            break
        state.remove_moving_piece()
        start_tile.move_piece(end_tile)
        state.next_turn()

        if player == "W":
            boards_W[piece] = end_tile
            player = "B"
        else:
            boards_B[piece] = end_tile
            player = "W"

    if not check:
        break