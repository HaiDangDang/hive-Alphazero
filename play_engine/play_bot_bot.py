
import torch
from alpha_zero.alpha_net import ChessNet, train
import pygame as pg
from settings import BACKGROUND, WIDTH, HEIGHT,RED, WHITE
from hive_engine.env_hive import GamePlay
import os
from tile import Tile, initialize_grid, draw_drag
from move_checker import is_valid_move, game_is_over, \
    player_has_no_moves
from menus import start_menu, end_menu, no_move_popup
from game_state import Game_State
from inventory_frame import Inventory_Frame
import numpy as np
from alpha_zero.MCTS_chess import UCT_search

def load_bot(path):
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.share_memory()
    save_as = path
    current_net_filename = os.path.join("./model_data/", \
                                        save_as)
    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['state_dict'])
    return net
pg.font.init()

screen = pg.display.set_mode((WIDTH, HEIGHT))
background = pg.Surface(screen.get_size())

pg.display.set_caption('Hive')
icon = pg.image.load('images/icon.png')
pg.display.set_icon(icon)

board = GamePlay(HEIGHT_MAP=HEIGHT - 100, WIDTH_MAP=WIDTH - 500)

white_inventory = Inventory_Frame((0, 158), 0, white=True, training=False)
black_inventory = Inventory_Frame((700, 158), 1, white=False, training=False)
board.state.running = True
board.state.main_loop = True

bot_1 = load_bot("v2_4.pth.tar")
bot_2 = load_bot("self_iter0_Map5_F44.pth.tar")

while board.state.running:

    while board.state.main_loop:
        # events = pg.event.get()
        # for event in events:



        pos = pg.mouse.get_pos()
        for event in pg.event.get():
            if event.type == pg.KEYUP:
                board_state = board.encode_board()
                np.sum(board_state[:, :,11])
                board_state.shape
                board_state = board_state.transpose(2, 0, 1)
                board_state = torch.from_numpy(board_state).float().cuda()
                board_state = torch.unsqueeze(board_state, 0)
                if board.state.player() == 0:
                    policy, value = bot_1(board_state)
                else:
                    policy, value = bot_2(board_state)

                policy = policy.detach().cpu().numpy().reshape(-1)
                actions = board.actions()
                policy = policy[actions]
                action = np.argmax(policy)
                board.move(actions[action])
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
                        old_tile.move_piece(new_tile)
                        if board.state.player() == 1:
                            for piece, value in board.black_pieces_set.items():
                                tile = value[0]
                                if tile == old_tile:
                                    board.black_pieces_set[piece] = [new_tile, 0]
                                    print("ccc")
                                    if old_tile.axial_coords != (99, 99):
                                        board.latest_pos['B'] = (piece, old_tile)
                                    break
                        else:
                            for piece, value in board.white_pieces_set.items():
                                tile = value[0]
                                if tile == old_tile:
                                    board.white_pieces_set[piece] = [new_tile, 0]
                                    print("ccc")

                                    if old_tile.axial_coords != (99, 99):
                                        board.latest_pos['W'] = (piece, old_tile)
                                    break
                        board.state.next_turn()

                        board.human_play()

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