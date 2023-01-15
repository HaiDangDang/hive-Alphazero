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


database = "../datahist"


piece_types = ["G1", "G2", "G3", "A1", "A2", "A3", "S1", "S2", "B1", "B2", "Q", ]

index_char = list(string.ascii_uppercase)
len(index_char)
number = np.arange(0, 27, 1)
index_number = list(map(str, number))

# for idx, file in enumerate(os.listdir(database)):
#
#     file_name = os.path.join(database, file)
#     with open(file_name, 'rb') as f:
#         data_boards = pickle.load(f)
index = np.random.choice(len(os.listdir(database)))

file_name = os.path.join(database, os.listdir(database)[index])
with open(file_name, 'rb') as f:
    data_boards = pickle.load(f)
pg.font.init()
# Create the screen

screen = pg.display.set_mode((WIDTH, HEIGHT))
background = pg.Surface(screen.get_size())

# Title and Icon

pg.display.set_caption('Hive')
icon = pg.image.load('images/icon.png')
pg.display.set_icon(icon)

state = Game_State(initialize_grid(HEIGHT - 250, WIDTH - 100, radius=20), render=True)

boards_W = {}
boards_B = {}
for p in piece_types:
    boards_B[p] = None
    boards_W[p] = None

i = 0
j = 0
board_matrix = np.empty((26, 26), dtype=object)

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
        board_matrix[i,j] = tile
        j += 1
        if j == 26:
            i += 1
            j = 0
white_inventory = Inventory_Frame((0, 158), 0, white=True)
black_inventory = Inventory_Frame((700, 158), 1, white=False)


state.running = True
state.main_loop = True
counter = 0
player = "W"
# current_step = data_boards[0]
# piece, x, y = current_step[0], current_step[1], current_step[2]
# start_x = ""
# start_y = ""
current_board = GamePlay()
#
# while state.running:
#     pos = pg.mouse.get_pos()
#     background.fill(BACKGROUND)
#     white_inventory.draw(background, pos)
#     black_inventory.draw(background, pos)
#     for tile in state.board_tiles:
#         if state.clicked:
#             tile.draw(background, pos, state.clicked)
#             if tile.under_mouse(pos) and state.moving_piece \
#                     is None and tile.has_pieces():
#                 state.add_moving_piece(tile.pieces[-1])
#         else:
#             tile.draw(background, pos)
#     if state.moving_piece:
#         draw_drag(background, pos, state.moving_piece)
#     state.turn_panel.draw(background, state.turn)
#     screen.blit(background, (0, 0))
#     pg.display.flip()
#
#     events = pg.event.get()
#     for event in events:
#         if event.type == pg.KEYUP:
#             board_state = copy.deepcopy(current_board.encode_board())
#
#             best_move, root, leaf = UCT_search(current_board, 50,chessnet)
#
#             current_board = do_decode_n_move_pieces(current_board,best_move) # decode move and move piece(s)
#
#             state = current_board.state


white_inventory = Inventory_Frame((0, 158), 0, white=True)
black_inventory = Inventory_Frame((700, 158), 1, white=False)
# print(len(state.board_tiles))
current_board.state.menu_loop = False
current_board.state.main_loop = True
state = current_board.state
# state.running = True
# state.main_loop = True
while state.running:
    while state.menu_loop:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                state.quit()
                break
            start_menu(screen, state, event)

    while state.move_popup_loop:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                state.quit()
                break
            no_move_popup(screen, background, state, event)

    while state.main_loop:
        if state.player() == 0:
            board_state = copy.deepcopy(current_board.encode_board())

            best_move, root, leaf = UCT_search(current_board, 50,chessnet)

            current_board = do_decode_n_move_pieces(current_board,best_move) # decode move and move piece(s)

            state = current_board.state

        pos = pg.mouse.get_pos()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                state.quit()
                break
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    state.quit()
                    break
            if event.type == pg.MOUSEBUTTONDOWN:
                state.click()
            if event.type == pg.MOUSEBUTTONUP:
                state.unclick()
                if state.moving_piece and state.is_player_turn():
                    old_tile = next(tile for tile in
                                    state.board_tiles if tile.has_pieces()
                                    and tile.pieces[-1]
                                    == state.moving_piece)
                    new_tile = next((tile for tile in
                                     state.board_tiles
                                     if tile.under_mouse(pos)), None)
                    if is_valid_move(state, old_tile, new_tile):
                        old_tile.move_piece(new_tile)
                        state.next_turn()
                        current_board.state = state
                        current_board.human_play()
                        current_board.get_actions(current_board.white_pieces_set)

                        # for piece, value in current_board.black_pieces_set.items():
                        #     tile = value[0]
                        #     if tile == old_tile:
                        #         current_board.black_pieces_set[piece] = [new_tile,0]
                        # current_board
                        # current_board.state = state
                        #
                        # if player_has_no_moves(state):
                        #     if (state.turn != 7 and state.turn != 8):
                        #         state.open_popup()

                state.remove_moving_piece()

        # only animate once each loop

        background.fill(BACKGROUND)
        white_inventory.draw(background, pos)
        black_inventory.draw(background, pos)
        for tile in state.board_tiles:
            if state.clicked:
                tile.draw(background, pos, state.clicked)
                # if tile.under_mouse(pos):
                #     # print(tile.axial_coords)
                #     if tile.color != RED:
                #         tile.color = RED
                #     else:
                #         tile.color = WHITE
                #     total = 0
                #     for _t in state.board_tiles:
                #         if _t.color == RED:
                #             total+=1
                #     print(total)
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

        # if game_is_over(state):
        #     state.end_game()
    #     break
    # break
    # while state.end_loop:
    #     end_menu(screen, state, event)  # drawing takes precedence over the close window button
    #     for event in pg.event.get():
    #         if event.type == pg.QUIT:
    #             state.quit()
    #             break
