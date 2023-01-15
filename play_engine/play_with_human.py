
import torch
from alpha_zero.alpha_net import ChessNet, train
import pygame as pg
from settings import BACKGROUND, WIDTH, HEIGHT,RED, WHITE
from hive_engine.env_hive import GamePlay
import os
from tile import Tile, initialize_grid, draw_drag
from move_checker import is_valid_move, game_is_over, \
    is_straight_line, move_does_not_break_hive
from menus import start_menu, end_menu, no_move_popup
from game_state import Game_State
from inventory_frame import Inventory_Frame
import numpy as np
from stable_baselines3 import PPO

from copy import  deepcopy

# for tile in board.state.board_tiles:
#     if tile.core_index == ("R", "13"):
#         print(tile.index_xy)
#         break
#
# delta_1 = 4 - 7
# delta_2 = 10 - delta_1
# delta_x = min(delta_1, delta_2)
#
# delta_1 = 4 - 7
# delta_2 = 10 - delta_1
# delta_y = min(delta_1, delta_2)
#
# x = tile
# is_straight_line(x.index_xy, tile.index_xy)

import copy
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

pg.font.init()

screen = pg.display.set_mode((WIDTH, HEIGHT))
background = pg.Surface(screen.get_size())

# Title and Icon


pg.display.set_caption('Hive')
icon = pg.image.load('images/icon.png')
pg.display.set_icon(icon)

board = GamePlay(HEIGHT_MAP=HEIGHT - 100, WIDTH_MAP=WIDTH - 500)
# state = board.encode_board()
# board.player()
# for i in range(6):
#     print(np.sum(state[:,:,44+i]))
# board.mini_black_actions(player='W')
# black_set = board.white_pieces_set
#
# for piece, value in black_set.items():
#     tile = value[0]
#     if tile.axial_coords != (99, 99):
#         # board.state.add_moving_piece(tile.pieces[-1])
#         if move_does_not_break_hive(board.state, tile):
#             print(piece)
#         else:
#             print(piece)
#
#             0x7f9259b99220
#         # board.state.remove_moving_piece()
#
#             lock.append(piece)
# old_tile = tile
# temp_piece = old_tile.pieces[-1]
# old_tile.remove_piece()
# tile_list = board.state.get_tiles_with_pieces()
# visited = []
# queue = []
# # if len(tile_list) == 0:
# #     return False
# visited.append(tile_list[0])
# queue.append(tile_list[0])
#
# while queue:
#     current_tile = queue.pop(0)
#
#     for neighbor_tile in [x for x in current_tile.adjacent_tiles
#                           if x.has_pieces()]:
#         if neighbor_tile not in visited:
#             visited.append(neighbor_tile)
#             queue.append(neighbor_tile)
#
# if len(visited) != len(tile_list):
#     old_tile.add_piece(temp_piece)
#     return False
# else:
#     old_tile.add_piece(temp_piece)
#     return True
# x = env.encode_board()
# np.sum(x[:, :, 11])
# board.state.player()
# x = board.encode_board()
# np.sum(x[:, :, 11 + 12])

white_inventory = Inventory_Frame((0, 158), 0, white=True, training=False)
black_inventory = Inventory_Frame((700, 158), 1, white=False, training=False)
board.state.running = True
board.state.main_loop = True
# env = copy.deepcopy(board)
# board = copy.deepcopy(env)

while board.state.running:
    # while board.state.menu_loop:
    #     for event in pg.event.get():
    #         if event.type == pg.QUIT:
    #             board.state.quit()
    #             break
    #         start_menu(screen, board.state, event)
    #
    # while board.state.move_popup_loop:
    #     for event in pg.event.get():
    #         if event.type == pg.QUIT:
    #             board.state.quit()
    #             break
    #         no_move_popup(screen, background, board.state, event)

    while board.state.main_loop:
        # np.sum(board_state[:,:,29])
        if board.state.player() == 0:
            board_state = board.encode_board()

            board_state = board_state.transpose(2, 0, 1)
            board_state = torch.from_numpy(board_state).float().cuda()
            board_state = torch.unsqueeze(board_state, 0)

            policy, value = net(board_state)
            policy = policy.detach().cpu().numpy().reshape(-1)
            actions = board.actions()
            if len(actions) == 0:
                action = -1
            else:
                policy = policy[actions]
                action = np.argmax(policy)
                print(actions[action], value)
                board.move(actions[action])

        pos = pg.mouse.get_pos()
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
                    # print(move_does_not_break_hive(board.state, old_tile))
                    if is_valid_move(board.state, old_tile, new_tile):
                        board_state = board.encode_board()

                        board_state = board_state.transpose(2, 0, 1)
                        board_state = torch.from_numpy(board_state).float().cuda()
                        board_state = torch.unsqueeze(board_state, 0)
                        policy, value = net(board_state)
                        print(value, board.state.player())
                        old_tile.move_piece(new_tile)
                        if board.state.player() == 0:
                            for piece, value in board.white_pieces_set.items():
                                tile = value[0]
                                if tile == old_tile and board.state.moving_piece == value[2]:
                                    board.white_pieces_set[piece] = [new_tile, 0, value[2]]
                                    print("move ", piece)
                                    break
                        else:
                            for piece, value in board.black_pieces_set.items():
                                tile = value[0]
                                if tile == old_tile and board.state.moving_piece == value[2]:
                                    board.black_pieces_set[piece] = [new_tile, 0, value[2]]
                                    print("move ", piece)
                                    break
                        board.state.next_turn()
                        board.human_play()

                        # board.get_actions(board.white_pieces_set)

                        # if player_has_no_moves(board.state):
                        #     if (board.state.turn != 7 and board.state.turn != 8):
                        #         board.state.open_popup()
                board.state.remove_moving_piece()

        # only animate once each loop

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

        if game_is_over(board.state):
            board.state.end_game()
    #     break
    # break
    while board.state.end_loop:
        end_menu(screen, board.state, event)  # drawing takes precedence over the close window button
        for event in pg.event.get():
            if event.type == pg.QUIT:
                board.state.quit()
                break