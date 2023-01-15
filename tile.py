#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pygame as pg
# from pieces import Queen, Grasshopper, Spider, Beetle, Ant
from settings import WHITE, RED, BLUE, BLACK, PURPLE, PIECE_WHITE, PIECE_BLACK
import string
from hive_engine.config import MAX_MAP_FULL, MAX_MAP_HAFT, index_char, index_number

class Tile:
    def __init__(self, coord_pair, axial_coords, radius, color, piece=None, index_xy=[MAX_MAP_HAFT, MAX_MAP_HAFT]):
        self.coords = coord_pair
        self.axial_coords = axial_coords
        self.radius = radius
        self.hex = get_hex_points(coord_pair, radius)
        self.hex_select = get_hex_points(coord_pair, radius * 1.1)
        self.color = color
        self.adjacent_tiles = []
        self.index_xy = index_xy
        if piece:
            self.pieces = [piece]
        else:
            self.pieces = []
        self.core_index = (index_char[self.index_xy[0]], index_number[self.index_xy[1]])
        # print(self.core_index)

    def change_coords(self, coord_pair):
        self.coords = coord_pair
        self.hex = get_hex_points(coord_pair, self.radius)
        self.hex_select = get_hex_points(coord_pair, self.radius * 1.1)

    def draw(self, surface, pos, clicked=False):
        FONT = pg.font.SysFont('Times New Norman', 26, bold=True)
        # print(self.axial_coords )
        font = FONT.render(f"{index_char[self.index_xy[0]]}"
                           f"{index_number[self.index_xy[1]]}",
                           True, BLACK)
        if self.under_mouse(pos):
            if clicked:
                pg.draw.polygon(surface, RED, self.hex)
            else:
                pg.draw.polygon(surface, RED, self.hex_select)
                pg.draw.polygon(surface, self.color, self.hex)
        else:
            pg.draw.polygon(surface, self.color, self.hex)
        if self.has_pieces():
            self.pieces[-1].draw(surface, self.coords)
            FONT = pg.font.SysFont('Times New Norman', 26, bold=True)
            color = RED
            if self.pieces[-1].color == PIECE_WHITE:
                color = BLACK

            font = FONT.render(f"{index_char[self.index_xy[0]]}"
                               f"{index_number[self.index_xy[1]]}",
                               True, color)
        # print(self.axial_coords)
        surface.blit(font, (self.coords[0] - 24,self.coords[1]-30))

    def under_mouse(self, pos):
        if distance(self.coords, pos) < self.radius - 1:
            return True
        else:
            return False

    def add_piece(self, piece):
        self.pieces.append(piece)
        self.pieces[-1].update_pos(self.coords)
        self.color = self.pieces[-1].color

    def remove_piece(self):
        self.pieces.pop(-1)
        if self.has_pieces():
            self.color = self.pieces[-1].color
        elif type(self) is Inventory_Tile:
            pass
        else:
            self.color = WHITE

    def move_piece(self, new_tile):
        new_tile.add_piece(self.pieces[-1])
        self.remove_piece()

    def has_pieces(self):
        if len(self.pieces) > 0:
            return True
        else:
            return False

    def set_coords_inventory(self, coord_pair):
        self.coords = coord_pair

    def is_hive_adjacent(self, state):
        for tile in self.adjacent_tiles:
            if tile.has_pieces():
                return True
        return False

    def set_adjacent_tiles(self, board_tiles):  # tiles don't move, only pieces do
        # (q, r) = self.axial_coords
        # adjacent_tiles = []
        # for tile in board_tiles:
        #     if tile.axial_coords in [
        #         (q, r - 1),
        #         (q + 1, r - 1),
        #         (q + 1, r),
        #         (q, r + 1),
        #         (q - 1, r + 1),
        #         (q - 1, r),
        #         ]:
        #         adjacent_tiles.append(tile)
        (q, r) = self.index_xy
        adjacent_tiles = []
        for tile in board_tiles:
            if tuple(tile.index_xy) in [
                ((q - 1)%MAX_MAP_FULL, r),
                ((q + 1)%MAX_MAP_FULL, r),
                (q, (r + 1)%MAX_MAP_FULL),
                (q, (r - 1)%MAX_MAP_FULL),
                ((q - 1)%MAX_MAP_FULL, (r - 1)%MAX_MAP_FULL),
                ((q + 1)%MAX_MAP_FULL, (r + 1)%MAX_MAP_FULL),
                ]:
                adjacent_tiles.append(tile)
        self.adjacent_tiles = adjacent_tiles

class Inventory_Tile(Tile):

    def __init__(self, coord_pair, axial_coords, radius, color, piece):
        super().__init__(coord_pair, axial_coords, radius, color, piece)


class Start_Tile(Tile):

    def __init__(self, coord_pair, axial_coords, radius, color, piece, index_xy):
        super().__init__(coord_pair, axial_coords, radius, BLUE, piece, index_xy)


def distance(pair_one, pair_two):
    (x1, y1) = pair_one
    (x2, y2) = pair_two
    return np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def get_hex_points(coord_pair, radius):
    (x, y) = coord_pair

    return (  # has to be in counterclockwise order for drawing
        (x, y + radius),  # top
        (x - ((radius * np.sqrt(3))/2), y + (radius / 2)),  # top-left
        (x - ((radius * np.sqrt(3))/2), y - (radius / 2)),  # bottom-left
        (x, y - radius),  # bottom
        (x + ((radius * np.sqrt(3))/2), y - (radius / 2)),  # bottom-right
        (x + ((radius * np.sqrt(3))/2), y + (radius / 2))  # top-right
    )


def initialize_grid(height, width, radius, center_tile=[MAX_MAP_HAFT +1, MAX_MAP_HAFT]):
    hex_radius = radius

    pixel_y = list(range(height, height//2, -2 * hex_radius + 6))[-MAX_MAP_HAFT:]
    pixel_y = pixel_y + \
              list(range(pixel_y[-1] + -2 * hex_radius + 6, 0 + 50, -2 * hex_radius + 6))[:MAX_MAP_HAFT]


    pixel_x = list(range(0, width//2, 2 * hex_radius))[-MAX_MAP_HAFT:]
    pixel_x =  pixel_x + list(range(pixel_x[-1] + 2 * hex_radius , width , 2 * hex_radius))[:MAX_MAP_HAFT]
    #
    # print(len(pixel_y))
    # print(len(pixel_x))

    axial_r = list(range(len(pixel_y) // 2 - 1, -(1 * len(pixel_y)
                   // 2) - 1, -1))
    # print(axial_r)
    odd_y = pixel_y[1::2]
    tiles = []
    delta_x = pixel_x[1] - pixel_x[0]
    for k in range(MAX_MAP_FULL +1 ):
        pixel_x.append(pixel_x[-1] + delta_x)
    extra = 0
    max_x = MAX_MAP_FULL
    for j in range(0, len(pixel_y)):
        j = len(pixel_y) - j -1
        count_k = 1
        for k in range(0, len(pixel_x)):
            if k < extra:
                continue
            if count_k > max_x:
                break
            if count_k == center_tile[0] and j == center_tile[1]:  # middle tile

                tiles.append(Start_Tile((pixel_x[k] + hex_radius*(j%2), pixel_y[j]),
                             ((j + 1) // 2 + k - 16, axial_r[j]),
                             hex_radius + 1, WHITE, None, index_xy=[j, count_k - 1]))
            else:
                tiles.append(Tile((pixel_x[k] + hex_radius*(j%2), pixel_y[j]),
                                  ((j + 1) // 2 + k - 16, axial_r[j]),
                             hex_radius + 1, WHITE, index_xy=[j, count_k - 1]))
            count_k += 1
        extra += 1 * j%2

    for tile in tiles:
        tile.set_adjacent_tiles(tiles)

    return tiles

# def initialize_grid(height, width, radius):
#     hex_radius = radius
#
#     # location of the tiles in pygame/cartesian pixels
#     pixel_y = list(range(height + hex_radius, 0 + 400, -2 * hex_radius + 6))
#     pixel_x = list(range(870, width + hex_radius , 2 * hex_radius))
#
#     # pixel_y = list(range(height + hex_radius -400, 200, -2 * hex_radius + 6))
#     # # pixel_x = list(range(200, width + hex_radius -450, 2 * hex_radius))
#     # print(len(pixel_x), len(pixel_y))
#     # #
#     print(len(pixel_y))
#     print(len(pixel_x))
#     # axial hexagonal coordinates used for move finding
#
#     axial_r = list(range(len(pixel_y) // 2 - 1, -(1 * len(pixel_y)
#                    // 2) - 1, -1))
#     odd_y = pixel_y[1::2]
#     tiles = []
#     for j in range(0, len(pixel_y)):
#         for k in range(0, len(pixel_x)):
#             if pixel_y[j] in odd_y:
#                 if k == 8 and j == 7:  # middle tile
#                     tiles.append(Start_Tile((pixel_x[k] + hex_radius,
#                                  pixel_y[j]), ((j + 1) // 2 + k - 16,
#                                  axial_r[j]), hex_radius + 1, WHITE, None))
#                 else:
#                     tiles.append(Tile((pixel_x[k] + hex_radius,
#                                  pixel_y[j]), ((j + 1) // 2 + k - 16,
#                                  axial_r[j]), hex_radius + 1, WHITE))
#             else:
#                 if k == 8 and j == 7:  # middle tile
#                 # if pixel_x[k] == 440 and pixel_y[j] == 380:  # middle tile
#                     tiles.append(Start_Tile((pixel_x[k], pixel_y[j]),
#                                  ((j + 1) // 2 + k - 16, axial_r[j]),
#                                  hex_radius + 1, WHITE, None))
#                 else:
#                     tiles.append(Tile((pixel_x[k], pixel_y[j]), ((j
#                                  + 1) // 2 + k - 16, axial_r[j]),
#                                  hex_radius + 1, WHITE))
#
#     for tile in tiles:
#         tile.set_adjacent_tiles(tiles)
#
#     return tiles

def draw_drag(background, pos, piece=None):
    pg.draw.line(background, pg.Color('red'), pos, piece.old_pos)
