#!/usr/bin/python
# -*- coding: utf-8 -*-
import pygame as pg
from pieces import Queen, Grasshopper, Spider, Beetle, Ant
from tile import Inventory_Tile
from settings import PIECE_WHITE, PIECE_BLACK, WHITE, BLACK, PANEL, \
    WIDTH, HEIGHT

class Inventory_Frame:

    def __init__(
        self,
        pos,
        player,
        white=True,
        training=True
        ):
        left = pos[0]
        top = HEIGHT - pos[1]

        inventory_width = WIDTH / 2 - 100
        inventory_height = 160

        inner_left = left + 5
        inner_top = top + 5
        inner_width = inventory_width - 10
        inner_height = inventory_height - 10

        self.back_panel = pg.Rect(left, top, inventory_width,
                                  inventory_height)
        self.inner_panel = pg.Rect(inner_left, inner_top, inner_width,
                                   inner_height)

        title_height = inner_height / 10
        stock_height = inner_height * (9 / 10)
        stock_width = inner_width / 5

        self.tile_rects = []
        self.tiles = []
        fix_inner = inner_left
        self.piece_sets = []
        if white:
            self.color = PIECE_WHITE
        else:
            self.color = PIECE_BLACK
            inner_left += 270
        for i in range(0, 5):
            self.tile_rects.append(pg.Rect(fix_inner + i * stock_width
                                   + 2, inner_top + title_height + 2,
                                   stock_width - 4, stock_height - 4))

            if i == 0:
                tile_pos = (inner_left + i * stock_width + stock_width
                            / 2, inner_top + title_height
                            + stock_height / 2)
                Q = Queen(self.color)
                self.piece_sets.append(Q)
                self.tiles.append(Inventory_Tile(tile_pos, (99, 99),
                                  20, self.color,
                                  piece=Q))
            if i == 1:
                for j in range(1, 3):
                    tile_pos = (inner_left + i * stock_width
                                + stock_width / 2, inner_top
                                + title_height + j * stock_height / 3)
                    B = Beetle(self.color)
                    self.piece_sets.append(B)
                    self.tiles.append(Inventory_Tile(tile_pos, (99,
                            99), 20, self.color,
                            piece=B))
            if i == 2:
                for j in range(1, 3):
                    tile_pos = (inner_left + i * stock_width
                                + stock_width / 2, inner_top
                                + title_height + j * stock_height / 3)
                    S = Spider(self.color)
                    self.piece_sets.append(S)
                    self.tiles.append(Inventory_Tile(tile_pos, (99,
                            99), 20, self.color,
                            piece=S))
            if i == 3:
                for j in [25, 67, 109]:
                    tile_pos = (inner_left + i * stock_width
                                + stock_width / 2, inner_top
                                + title_height + j * stock_height / 135)
                    G = Grasshopper(self.color)
                    self.piece_sets.append(G)
                    self.tiles.append(Inventory_Tile(tile_pos, (99,
                            99), 20, self.color,
                            piece=G))
            if i == 4:
                for j in [25, 67, 109]:
                    tile_pos = (inner_left + i * stock_width
                                + stock_width / 2, inner_top
                                + title_height + j * stock_height / 135)
                    A = Ant(self.color)
                    self.piece_sets.append(A)
                    self.tiles.append(Inventory_Tile(tile_pos, (99,
                            99), 20, self.color, piece=A))
        for tile in self.tiles:
            for piece in tile.pieces:
                piece.update_pos(tile.coords)

        if training == False:
            FONT = pg.font.SysFont('Times New Norman', 24)
            if player == 0:
                self.font = FONT.render('Player 1 Inventory', True, WHITE)
            else:
                self.font = FONT.render('Player 2 Inventory', True, WHITE)
            self.title_rect = self.font.get_rect(center=(fix_inner
                    + inner_width / 2, inner_top + title_height / 2))

    def draw(self, background, pos):

        pg.draw.rect(background, BLACK, self.back_panel)
        pg.draw.rect(background, PANEL, self.inner_panel)
        pg.draw.rect(background, PANEL, self.title_rect)
        for i in range(0, len(self.tile_rects)):
            pg.draw.rect(background, self.color, self.tile_rects[i])
        # print(self.title_rect)
        background.blit(self.font, self.title_rect)
        pg.display.flip()
