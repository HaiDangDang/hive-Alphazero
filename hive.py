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
from settings import BACKGROUND, WIDTH, HEIGHT,RED, WHITE
import sys
from hive_engine.config import MAX_MAP_FULL, MAX_MAP_HAFT


def Hive():

    pg.font.init()

    # Create the screen

    screen = pg.display.set_mode((WIDTH, HEIGHT))
    background = pg.Surface(screen.get_size())

    # Title and Icon

    pg.display.set_caption('Hive')
    icon = pg.image.load('images/icon.png')
    pg.display.set_icon(icon)

    state = Game_State(initialize_grid(HEIGHT - 100, WIDTH - 500, radius=20), render=True)

    # state.menu_loop = False

    # x = state.board_tiles
    # for i in x:
    #     print(i.axial_coords)
    # state = current_board.state
    white_inventory = Inventory_Frame((0, 158), 0, white=True)
    black_inventory = Inventory_Frame((700, 158), 1, white=False)
    # print(len(state.board_tiles))
    # state.menu_loop = False
    # state.main_loop = True

    # state.running = True
    # state.main_loop = True
    state = board.state
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
            # if state.player() == 0:
            #     board_state = copy.deepcopy(current_board.encode_board())
            #     # current_board.encode_action([0,0], True)
            #     # board_state.shape
            #     # # np.mean(board_state)
            #     # root.child_total_value
            #     # current_board.player()
            #     # start_time = time.time()
            #     # root.is_expanded
            #     best_move, root = UCT_search(current_board,50,chessnet)
            #     # print("--- %s seconds ---" % (time.time() - start_time))
            #
            #     # root.child_number_visits[root.child_number_visits!=0]
            #     # np.argmax(root.child_number_visits)
            #     # len(current_board.actions_move)
            #     # current_board.not_encode_a
            #     # current_board.state.winner
            #
            #     current_board = do_decode_n_move_pieces(current_board,best_move) #
            #     state = current_board.state

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
                            # for piece, value in current_board.black_pieces_set.items():
                            #     tile = value[0]
                            #     if tile == old_tile:
                            #         current_board.black_pieces_set[piece] = [new_tile,0]
                            # current_board
                            # current_board.state = state

                            if player_has_no_moves(state):
                                if (state.turn != 7 and state.turn != 8):
                                    state.open_popup()

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

    return state.play_new_game


def main():
    run_game = True
    while run_game:
        run_game = Hive()


if __name__ == '__main__':
    main()
